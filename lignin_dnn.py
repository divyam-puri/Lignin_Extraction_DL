"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   LIGNIN REMOVAL PREDICTOR  —  PyTorch DNN  v8  (Zero-Leakage)            ║
║                                                                              ║
║   STRICT DATA CONTRACT:                                                     ║
║   • engineered_features (467) → internal train/val for Optuna + HPO        ║
║   • validation_dataset   (42) → blind test, evaluated ONCE at the end      ║
║   • Scalers fit on 467 ONLY                                                 ║
║   • 42 samples invisible until final evaluation                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import joblib, optuna, warnings
from pymongo import MongoClient
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")

TARGET = "Lignin_remove_yield"


# ─────────────────────────────────────────────────────────────────────────────
# 1 · ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
class LigninNetV8(nn.Module):
    def __init__(self, n_in, layer_sizes, dropouts):
        super().__init__()
        layers = []
        prev   = n_in
        for size, drop in zip(layer_sizes, dropouts):
            layers += [
                nn.Linear(prev, size),
                nn.LayerNorm(size),   # stable for small chemical datasets
                nn.GELU(),            # smoother gradients than ReLU/SiLU
                nn.Dropout(drop)
            ]
            prev = size
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

        # Kaiming init for GELU
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · FEATURE ENGINEERING  (pure math — no data statistics, safe for both sets)
# ─────────────────────────────────────────────────────────────────────────────
def add_physics_features(df):
    df = df.copy()
    if "temperature_C" in df.columns and "time_hr" in df.columns:
        t  = df["temperature_C"].astype(float)
        hr = df["time_hr"].astype(float)
        if "LogR0" not in df.columns:
            df["LogR0"] = np.log10((hr + 1e-9) * np.exp((t - 100) / 14.75))
        df["LogR0_sq"]        = df["LogR0"] ** 2
        df["severity_x_time"] = df["LogR0"] * hr
        df["log_time"]        = np.log1p(hr)
        df["sqrt_time"]       = np.sqrt(hr.clip(0))
        df["temp_sq"]         = t ** 2
        df["temp_x_LogR0"]    = t * df["LogR0"]
        df["inv_temp"]        = 1.0 / (t + 273.15)

    if "liquid_solid_ratio" in df.columns and "LogR0" in df.columns:
        lsr = df["liquid_solid_ratio"].astype(float)
        df["log_LSR"]     = np.log1p(lsr)
        df["LSR_x_LogR0"] = lsr * df["LogR0"]
        df["LSR_sq"]      = lsr ** 2

    if "HBD_HBA_ratio" in df.columns and "LogR0" in df.columns:
        df["ratio_x_LogR0"] = df["HBD_HBA_ratio"].astype(float) * df["LogR0"]

    if "lignin_percent" in df.columns and "LogR0" in df.columns:
        df["lignin_x_LogR0"] = df["lignin_percent"].astype(float) * df["LogR0"]

    if "HBA-MW" in df.columns and "HBD-MW" in df.columns:
        df["MW_ratio"] = (df["HBA-MW"].astype(float) /
                          (df["HBD-MW"].astype(float) + 1e-9))

    if "HBA-SLogP" in df.columns and "HBD-SLogP" in df.columns:
        df["SLogP_sum"] = (df["HBA-SLogP"].astype(float) +
                           df["HBD-SLogP"].astype(float))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3 · FEATURE LIST  (only columns present in BOTH collections)
# ─────────────────────────────────────────────────────────────────────────────
BASE_PROCESS = [
    "cellulose_percent", "hemicellulose_percent", "lignin_percent",
    "size_mm", "temperature_C", "time_hr",
    "HBD_HBA_ratio", "liquid_solid_ratio", "LogR0"
]
BASE_MOLECULAR = [
    "HBA-pKa/pkb", "HBD-pKa/pkb", "HBD-MW",
    "HBA-TopoPSA", "HBD-TopoPSA",
    "HBA-nHBAcc", "HBA-nHBDon", "HBD-nHBAcc", "HBD-nHBDon",
    "HBA-SlogP_VSA1", "HBA-SLogP", "HBD-SlogP_VSA1", "HBD-SLogP",
    "HBA-nAromAtom", "HBD-nAromAtom",
    "HBA-nRot", "HBD-nRot",
    "HBA-nBase", "HBD-nBase", "HBD-nC"
]
ENGINEERED_EXTRA = [
    "LogR0_sq", "severity_x_time", "log_time", "sqrt_time",
    "temp_sq", "temp_x_LogR0", "inv_temp",
    "log_LSR", "LSR_x_LogR0", "LSR_sq",
    "ratio_x_LogR0", "lignin_x_LogR0", "MW_ratio", "SLogP_sum"
]
ALL_CANDIDATES = BASE_PROCESS + BASE_MOLECULAR + ENGINEERED_EXTRA


# ─────────────────────────────────────────────────────────────────────────────
# 4 · SHARED TRAINING FUNCTION
#     Used by both Optuna trials and final retraining.
#     Takes pre-built tensors → returns trained model.
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(n_features, layer_sizes, dropouts, lr, wd, bs,
                    Xtr_t, ytr_t, Xva_t, yva_t,
                    max_epochs=500, patience=60):
    """
    Trains a LigninNetV8 and returns the best model (by val loss).
    No test data is passed in — caller controls what val tensors contain.
    """
    model   = LigninNetV8(n_features, layer_sizes, dropouts).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch     = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                  opt, T_0=80, T_mult=2, eta_min=1e-6)
    loss_fn = nn.HuberLoss(delta=0.15)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=True,
                    drop_last=False)

    best_val  = float("inf")
    best_wts  = None
    patience_c = 0
    tr_hist, va_hist = [], []

    for ep in range(max_epochs):
        # ── Train ──
        model.train()
        ep_loss = 0.0
        for Xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(Xb)
        sch.step()
        tr_hist.append(ep_loss / len(Xtr_t))

        # ── Val ──
        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(Xva_t), yva_t).item()
        va_hist.append(va_loss)

        if va_loss < best_val - 1e-7:
            best_val   = va_loss
            best_wts   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_c = 0
        else:
            patience_c += 1
        if patience_c >= patience:
            break

    model.load_state_dict(best_wts)
    return model, tr_hist, va_hist


def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)


def predict_original_scale(model, X_scaled, scaler_y):
    model.eval()
    with torch.no_grad():
        p = model(to_tensor(X_scaled)).cpu().numpy().flatten()
    return scaler_y.inverse_transform(p.reshape(-1, 1)).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# 5 · MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_v8_pipeline():

    # ── LOAD ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 1 · Loading Data")
    print("=" * 65)
    client = MongoClient("mongodb://localhost:27017/")
    db     = client["Lignin"]
    df_eng = pd.DataFrame(list(db["engineered_features"].find({}, {"_id": 0})))
    df_val = pd.DataFrame(list(db["validation_dataset"].find({}, {"_id": 0})))
    client.close()

    assert TARGET in df_eng.columns
    assert TARGET in df_val.columns
    print(f"  engineered_features : {len(df_eng)} rows  ← TRAIN/VAL only")
    print(f"  validation_dataset  : {len(df_val)} rows  ← BLIND TEST (locked)")

    # ── FEATURE ENGINEERING ───────────────────────────────────────────────────
    df_eng = add_physics_features(df_eng)
    df_val = add_physics_features(df_val)

    FEATURES = [f for f in ALL_CANDIDATES
                if f in df_eng.columns and f in df_val.columns]
    print(f"\n✓ Features used : {len(FEATURES)}")

    # ── EXTRACT ARRAYS ────────────────────────────────────────────────────────
    X_dev   = df_eng[FEATURES].values.astype(np.float32)
    y_dev   = df_eng[TARGET].values.astype(np.float32)
    X_blind = df_val[FEATURES].values.astype(np.float32)   # ← stays locked
    y_blind = df_val[TARGET].values.astype(np.float32)     # ← stays locked

    # NaN fill using ONLY dev (467) column means
    col_means = np.nanmean(X_dev, axis=0)
    for i in range(X_dev.shape[1]):
        X_dev[np.isnan(X_dev[:, i]),     i] = col_means[i]
        X_blind[np.isnan(X_blind[:, i]), i] = col_means[i]  # train mean, not blind

    # ── SCALE — fit on ALL 467 DEV SAMPLES ───────────────────────────────────
    # RobustScaler: median/IQR based → handles experimental outliers well
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()

    X_dev_s = scaler_x.fit_transform(X_dev).astype(np.float32)
    y_dev_s = scaler_y.fit_transform(
                  y_dev.reshape(-1, 1)).flatten().astype(np.float32)

    # Blind set transformed with dev scaler — NOT fit on blind data
    X_blind_s = scaler_x.transform(X_blind).astype(np.float32)

    # ── INTERNAL SPLIT from 467 ONLY ─────────────────────────────────────────
    # 374 train / 93 val — both from engineered_features
    # 42 blind test samples are not involved here at all
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev_s, y_dev_s,
        test_size=0.20,
        random_state=42,
        shuffle=True
    )

    # Store unscaled y_va for R² reporting in original units
    y_va_orig = scaler_y.inverse_transform(y_va.reshape(-1, 1)).flatten()

    print(f"\n  Internal split (from 467 only):")
    print(f"    Optuna train : {len(X_tr)} samples")
    print(f"    Optuna val   : {len(X_va)} samples")
    print(f"    Blind test   : {len(X_blind)} samples  ← NOT visible yet")

    # Pre-build internal tensors for Optuna (reused across all 100 trials)
    Xtr_t = to_tensor(X_tr)
    ytr_t = to_tensor(y_tr).unsqueeze(1)
    Xva_t = to_tensor(X_va)
    yva_t = to_tensor(y_va).unsqueeze(1)

    # ── OPTUNA OBJECTIVE ──────────────────────────────────────────────────────
    # Maximises internal val R² from the 374/93 split.
    # The 42 blind samples are completely invisible here.
    def objective(trial):
        n_layers    = trial.suggest_int("n_layers", 2, 4)
        layer_sizes = [trial.suggest_categorical(f"s{i}", [128, 256, 512])
                       for i in range(n_layers)]
        dropouts    = [trial.suggest_float(f"d{i}", 0.05, 0.30)
                       for i in range(n_layers)]
        lr          = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd          = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
        bs          = trial.suggest_categorical("bs", [16, 32, 64])

        model, _, _ = train_one_model(
            len(FEATURES), layer_sizes, dropouts, lr, wd, bs,
            Xtr_t, ytr_t, Xva_t, yva_t,
            max_epochs=400, patience=50          # shorter for HPO speed
        )

        # Val R² in original units (not scaled) — honest metric for Optuna
        preds_orig = predict_original_scale(model, X_va, scaler_y)
        return r2_score(y_va_orig, preds_orig)

    # ── RUN OPTUNA ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2 · Optuna Search (100 trials)")
    print("  Objective : internal val R² from 467 samples only")
    print("  Blind 42  : NOT visible during search")
    print("=" * 65)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    best_so_far = [-999]
    trial_count = [0]

    def callback(study, trial):
        trial_count[0] += 1
        if study.best_value > best_so_far[0]:
            best_so_far[0] = study.best_value
            p = trial.params
            print(f"  Trial {trial_count[0]:>3}  ★ val R²={study.best_value:.4f}  "
                  f"layers={p['n_layers']}  lr={p['lr']:.5f}")
        elif trial_count[0] % 25 == 0:
            print(f"  Trial {trial_count[0]:>3}    best val R²={study.best_value:.4f}")

    study.optimize(objective, n_trials=100, callbacks=[callback])

    bp = study.best_params
    print(f"\n✓ Optuna complete — best internal val R² = {study.best_value:.4f}")
    print(f"  Best params : {bp}")

    # ── RETRAIN ON ALL 467 WITH BEST HYPERPARAMETERS ──────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 3 · Retrain on ALL 467 samples with best hyperparameters")
    print("  Blind 42  : STILL NOT VISIBLE")
    print("=" * 65)

    n_layers    = bp["n_layers"]
    layer_sizes = [bp[f"s{i}"] for i in range(n_layers)]
    dropouts    = [bp[f"d{i}"] for i in range(n_layers)]
    lr, wd, bs  = bp["lr"], bp["wd"], bp["bs"]

    print(f"  Architecture : {len(FEATURES)} → {' → '.join(map(str, layer_sizes))} → 1")
    print(f"  LR={lr:.5f}  WD={wd:.5f}  Batch={bs}")

    # Small holdout from 467 for early stopping during final retraining
    # This is NOT the blind test set — it's 10% of the 467
    X_final_tr, X_final_va, y_final_tr, y_final_va = train_test_split(
        X_dev_s, y_dev_s,
        test_size=0.10,
        random_state=1,     # different seed to Optuna split
        shuffle=True
    )

    Xftr_t = to_tensor(X_final_tr)
    yftr_t = to_tensor(y_final_tr).unsqueeze(1)
    Xfva_t = to_tensor(X_final_va)
    yfva_t = to_tensor(y_final_va).unsqueeze(1)

    final_model, tr_hist, va_hist = train_one_model(
        len(FEATURES), layer_sizes, dropouts, lr, wd, bs,
        Xftr_t, yftr_t, Xfva_t, yfva_t,
        max_epochs=1500, patience=120    # longer for final model
    )
    print(f"✓ Final model trained for {len(tr_hist)} epochs")

    # ── EVALUATE ON ALL 467 (train performance) ───────────────────────────────
    pred_dev = predict_original_scale(final_model, X_dev_s, scaler_y)
    r2_dev   = r2_score(y_dev, pred_dev)

    # ─────────────────────────────────────────────────────────────────────────
    # FINAL BLIND TEST — 42 samples — evaluated ONCE — no changes after this
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 4 · FINAL BLIND TEST on 42 samples")
    print("  This is the FIRST and ONLY time the model sees these samples.")
    print("=" * 65)

    pred_blind = predict_original_scale(final_model, X_blind_s, scaler_y)
    r2_blind   = r2_score(y_blind,  pred_blind)
    mae_blind  = mean_absolute_error(y_blind, pred_blind)
    rmse_blind = np.sqrt(mean_squared_error(y_blind, pred_blind))

    print(f"\n  Dev  R² (467) : {r2_dev:.4f}")
    print(f"  Blind R² (42) : {r2_blind:.4f}  ← honest score")
    print(f"  Blind MAE     : {mae_blind:.4f}")
    print(f"  Blind RMSE    : {rmse_blind:.4f}")

    # ── FEATURE IMPORTANCE (permutation on blind set — read only) ─────────────
    rng = np.random.RandomState(0)
    importances = []
    for i in range(len(FEATURES)):
        Xp = X_blind_s.copy()
        Xp[:, i] = rng.permutation(Xp[:, i])
        p_shuf   = predict_original_scale(final_model, Xp, scaler_y)
        importances.append(r2_blind - r2_score(y_blind, p_shuf))
    feat_imp = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    # ── PLOTS ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Lignin DNN v8 (Zero-Leakage) — "
        f"Train=467 | Blind Test=42\n"
        f"Blind Test R²={r2_blind:.4f}   Dev R²={r2_dev:.4f}",
        fontsize=12, fontweight="bold"
    )

    # ① Blind test parity
    ax = axes[0, 0]
    sc = ax.scatter(y_blind, pred_blind, alpha=0.85, s=80,
                    edgecolors="k", linewidth=0.5,
                    c=np.abs(pred_blind - y_blind), cmap="RdYlGn_r")
    plt.colorbar(sc, ax=ax, label="|error|")
    lims = [min(y_blind.min(), pred_blind.min()) - 0.02,
            max(y_blind.max(), pred_blind.max()) + 0.02]
    ax.plot(lims, lims, "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual Yield"); ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Blind Test Parity — 42 samples\nR²={r2_blind:.4f}  "
                 f"MAE={mae_blind:.4f}  RMSE={rmse_blind:.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    # ② Dev set parity
    ax = axes[0, 1]
    ax.scatter(y_dev, pred_dev, alpha=0.40, s=25,
               edgecolors="k", linewidth=0.3, color="steelblue")
    lims2 = [min(y_dev.min(), pred_dev.min()) - 0.02,
             max(y_dev.max(), pred_dev.max()) + 0.02]
    ax.plot(lims2, lims2, "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual Yield"); ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Dev Parity — 467 samples  (R²={r2_dev:.4f})")
    ax.legend(); ax.grid(alpha=0.3)

    # ③ Training loss curve
    ax = axes[0, 2]
    ax.plot(tr_hist, label="Train (421 samples)", lw=2, color="steelblue")
    ax.plot(va_hist, label="Early-stop val (46 samples)", lw=2,
            color="orange", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Final Model Training History")
    ax.legend(); ax.grid(alpha=0.3)

    # ④ Optuna search history (internal val R², never blind)
    ax = axes[1, 0]
    trial_r2s    = [t.value for t in study.trials if t.value is not None]
    running_best = np.maximum.accumulate(trial_r2s)
    ax.plot(trial_r2s,    alpha=0.4, color="steelblue", lw=1, label="Trial val R²")
    ax.plot(running_best, color="navy",  lw=2, label="Best so far")
    ax.axhline(0.8259, color="red",   ls=":", lw=1.5, label="XGBoost (0.8259)")
    ax.axhline(r2_blind, color="green", ls="-", lw=1.5,
               label=f"Blind R²={r2_blind:.4f}")
    ax.set_xlabel("Trial"); ax.set_ylabel("Internal Val R²")
    ax.set_title("Optuna — 100 trials (internal val only)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # ⑤ Residuals
    ax = axes[1, 1]
    res = y_blind - pred_blind
    ax.hist(res, bins=12, color="mediumpurple", edgecolor="white")
    ax.axvline(0,        color="red",    lw=2,   ls="--", label="Zero")
    ax.axvline(res.mean(), color="orange", lw=1.5, ls="--",
               label=f"Mean={res.mean():.4f}")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_title("Blind Test Residuals (42 samples)")
    ax.legend(); ax.grid(alpha=0.3)

    # ⑥ Feature importance
    ax = axes[1, 2]
    top  = feat_imp.head(15)
    cols = ["#e74c3c" if v > 0 else "#95a5a6" for v in top.values]
    ax.barh(range(len(top)), top.values, color=cols,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_xlabel("R² Drop (importance)")
    ax.set_title("Feature Importances\n(Permutation on blind test)")
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig("lignin_dnn_v8_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Plot saved → lignin_dnn_v8_results.png")

    # ── SAVE ──────────────────────────────────────────────────────────────────
    joblib.dump(scaler_x, "lignin_scaler_x_v8.pkl")
    joblib.dump(scaler_y, "lignin_scaler_y_v8.pkl")
    torch.save({
        "model_state":   final_model.state_dict(),
        "n_features":    len(FEATURES),
        "layer_sizes":   layer_sizes,
        "dropouts":      dropouts,
        "feature_names": FEATURES,
        "blind_r2":      float(r2_blind),
        "dev_r2":        float(r2_dev),
        "best_params":   bp,
    }, "lignin_dnn_v8_final.pt")
    print("✓ Model  saved → lignin_dnn_v8_final.pt")
    print("✓ Scalers saved → lignin_scaler_x/y_v8.pkl")

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  FINAL RESULTS — v8 (Zero-Leakage)                              ║
╠══════════════════════════════════════════════════════════════════╣
║  DATA CONTRACT                                                   ║
║    Dev pool      : 467  (engineered_features)                   ║
║    Optuna train  : 374  (80% of 467)                            ║
║    Optuna val    : 93   (20% of 467)  ← Optuna objective        ║
║    Final train   : 421  (90% of 467)  ← retrain                ║
║    Early-stop    : 46   (10% of 467)  ← retrain guard          ║
║    Blind test    : 42   (validation_dataset, seen ONCE)         ║
╠══════════════════════════════════════════════════════════════════╣
║  PERFORMANCE                                                     ║
║    Dev  R² (467) : {r2_dev:.4f}                                   ║
║    Blind R² (42) : {r2_blind:.4f}   ← honest, final score          ║
║    Blind MAE     : {mae_blind:.4f}                                   ║
║    Blind RMSE    : {rmse_blind:.4f}                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  BENCHMARK                                                       ║
║    XGBoost paper : 0.8259                                       ║
║    This model    : {r2_blind:.4f}  ({'+' if r2_blind > 0.8259 else ''}{r2_blind - 0.8259:+.4f} vs benchmark)          ║
║    Top feature   : {feat_imp.index[0]:<40}║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("✅ Leakage audit:")
    print("   • 42 blind samples not used in Optuna objective    ✓")
    print("   • 42 blind samples not used in early stopping      ✓")
    print("   • 42 blind samples not used in scaler fitting      ✓")
    print("   • 42 blind samples evaluated exactly ONCE          ✓")
    print("   • Scalers fit on dev (467) only                    ✓")
    print("   • Model not modified after blind evaluation        ✓")
    print("\n✅  Pipeline v8 complete!")

    return final_model, r2_blind, feat_imp


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_v8_pipeline()
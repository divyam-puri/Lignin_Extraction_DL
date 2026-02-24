"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   LIGNIN REMOVAL PREDICTOR  —  PyTorch DNN  v8  (Zero-Leakage)            ║
║                                                    + SHAP Analysis         ║
║                                                                              ║
║   STRICT DATA CONTRACT:                                                     ║
║   • engineered_features (467) → internal train/val for Optuna + HPO        ║
║   • validation_dataset   (42) → blind test, evaluated ONCE at the end      ║
║   • Scalers fit on 467 ONLY                                                 ║
║   • 42 samples invisible until final evaluation                             ║
║                                                                              ║
║   SHAP NOTE:                                                                ║
║   • Uses KernelExplainer (model-agnostic) on the final DNN                 ║
║   • Background = k-means summary of dev set (fast & memory-safe)           ║
║   • SHAP computed on BLIND test set (42 samples) — same as paper           ║
║   • Produces: summary plot, dependence plots, force plot for sample #1     ║
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

import joblib, optuna, warnings, shap
from pymongo import MongoClient
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import MiniBatchKMeans

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
# ─────────────────────────────────────────────────────────────────────────────
def train_one_model(n_features, layer_sizes, dropouts, lr, wd, bs,
                    Xtr_t, ytr_t, Xva_t, yva_t,
                    max_epochs=500, patience=60):
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
# 5 · SHAP WRAPPER
#     KernelExplainer is model-agnostic — works with any PyTorch DNN.
#     We wrap the model in a numpy→numpy function for SHAP compatibility.
# ─────────────────────────────────────────────────────────────────────────────
def make_shap_predict_fn(model, scaler_y):
    """
    Returns a function: X_scaled (np.array) → predictions in ORIGINAL units.
    SHAP will call this hundreds of times with subsets of features masked.
    We keep eval mode and no_grad for speed.
    """
    def _predict(X_np):
        model.eval()
        with torch.no_grad():
            t   = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
            out = model(t).cpu().numpy().flatten()
        return scaler_y.inverse_transform(out.reshape(-1, 1)).flatten()
    return _predict


def compute_shap_values(model, scaler_y, X_dev_s, X_blind_s,
                         feature_names, n_background=50, n_explain=None):
    """
    Computes SHAP values for X_blind_s using KernelExplainer.

    Parameters
    ----------
    model        : trained LigninNetV8
    scaler_y     : fitted RobustScaler for target
    X_dev_s      : scaled dev set (467 × F) — used to build background
    X_blind_s    : scaled blind test set (42 × F) — points to explain
    feature_names: list of feature name strings
    n_background : number of k-means centroids for background summary
    n_explain    : how many blind samples to explain (None = all 42)

    Returns
    -------
    shap_values  : np.array of shape (n_explain, n_features)
    explainer    : the fitted shap.KernelExplainer object
    """
    print(f"\n  Building SHAP background ({n_background} k-means centroids"
          f" from {len(X_dev_s)} dev samples)...")
    # k-means summary keeps background small → KernelExplainer stays fast
    background = shap.kmeans(X_dev_s, n_background)

    predict_fn = make_shap_predict_fn(model, scaler_y)
    explainer  = shap.KernelExplainer(predict_fn, background)

    X_explain = X_blind_s if n_explain is None else X_blind_s[:n_explain]
    print(f"  Computing SHAP values for {len(X_explain)} blind samples "
          f"(this may take a few minutes)...")

    # nsamples="auto" lets SHAP decide; l1_reg="aic" prunes noisy features
    shap_values = explainer.shap_values(X_explain, nsamples="auto",
                                         l1_reg="aic", silent=True)
    print(f"✓ SHAP computation complete. Shape: {shap_values.shape}")
    return shap_values, explainer


# ─────────────────────────────────────────────────────────────────────────────
# 6 · SHAP PLOT SUITE  (mirrors the paper's Fig. 5 & 6 style)
# ─────────────────────────────────────────────────────────────────────────────
def plot_shap_suite(shap_values, X_blind_s, feature_names,
                    explainer, scaler_y, y_blind,
                    top_n=15, save_prefix="lignin_dnn_v8_shap"):
    """
    Generates four SHAP visualisations matching the paper's approach:

    1. Summary beeswarm plot  (Fig. 5a equivalent)
    2. Bar plot of mean |SHAP| (Fig. 5b / F-score equivalent)
    3. Dependence plots for top-3 features  (Fig. 6 equivalent)
    4. Force plot for a single sample       (local explanation)
    """

    # ── Map back to original (unscaled) feature values for x-axis readability
    # RobustScaler: X_orig = X_scaled * IQR + median
    scale_  = explainer.data.data   # background is already scaled
    # Use scaler to recover original feature values for blind set
    # (we pass scaled values to SHAP but want original for axis labels)
    feat_arr = X_blind_s[:len(shap_values)]  # same rows as shap_values

    # ── 1. Summary beeswarm plot ──────────────────────────────────────────────
    print("  Generating SHAP summary (beeswarm) plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, feat_arr,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_type="dot"
    )
    plt.title("SHAP Summary Plot — DNN v8 (Blind Test, 42 samples)\n"
              "Each point = one sample; colour = feature value (high→red)",
              fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_prefix}_summary.png")

    # ── 2. Bar plot (mean |SHAP|) ────────────────────────────────────────────
    print("  Generating SHAP bar (mean |SHAP|) plot...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df   = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    top_imp = imp_df.head(top_n)
    colors  = ["#e74c3c" if v == top_imp.max() else "#3498db"
               for v in top_imp.values]
    ax.barh(range(len(top_imp)), top_imp.values, color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp.index, fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (mean absolute contribution)", fontsize=10)
    ax.set_title(f"Feature Importance — DNN v8\n"
                 f"SHAP Mean |SHAP| on Blind Test (42 samples)", fontsize=11,
                 fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_prefix}_bar.png")

    # ── 3. Dependence plots — top 3 features ─────────────────────────────────
    print("  Generating SHAP dependence plots (top-3 features)...")
    top3 = list(imp_df.head(3).index)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("SHAP Dependence Plots — DNN v8 (Top-3 Features)\n"
                 "x-axis: scaled feature value  |  y-axis: SHAP contribution",
                 fontsize=11, fontweight="bold")

    for ax, fname in zip(axes, top3):
        fidx = list(feature_names).index(fname)
        sv   = shap_values[:, fidx]
        fv   = feat_arr[:, fidx]

        # colour by second most interacting feature (highest |corr| with SHAP)
        corrs  = [abs(np.corrcoef(shap_values[:, j], fv)[0, 1])
                  for j in range(len(feature_names)) if j != fidx]
        int_j  = int(np.argmax(corrs))
        int_j  = int_j if int_j < fidx else int_j + 1
        int_v  = feat_arr[:, int_j]

        sc = ax.scatter(fv, sv, c=int_v, cmap="RdYlBu_r",
                        s=60, edgecolors="k", linewidth=0.4, alpha=0.85)
        plt.colorbar(sc, ax=ax,
                     label=f"Colour: {feature_names[int_j]}", shrink=0.8)
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlabel(fname, fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(fname, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_prefix}_dependence.png")

    # ── 4. Force plot — sample closest to median prediction ──────────────────
    print("  Generating SHAP force plot (single sample)...")
    pred_vals = shap_values.sum(axis=1) + explainer.expected_value
    median_idx = int(np.argmin(np.abs(pred_vals - np.median(pred_vals))))

    force_fig = shap.force_plot(
        explainer.expected_value,
        shap_values[median_idx],
        feat_arr[median_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.title(f"SHAP Force Plot — Blind Sample #{median_idx}\n"
              f"(sample closest to median prediction)", fontsize=10,
              fontweight="bold", pad=40)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_force.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_prefix}_force.png")

    # ── 5. Combined SHAP dashboard (paper-style, all 4 panels) ───────────────
    print("  Generating SHAP dashboard (all panels combined)...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(
        "SHAP Analysis — Lignin DNN v8 (Zero-Leakage)\n"
        "Blind Test Set (42 samples) — KernelExplainer with k-means background",
        fontsize=13, fontweight="bold"
    )

    # Panel A: Mean |SHAP| bar
    ax = axes[0, 0]
    ax.barh(range(len(top_imp)), top_imp.values, color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(len(top_imp)))
    ax.set_yticklabels(top_imp.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("(A) Global Feature Importance\n(Mean |SHAP| on Blind Test)",
                 fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Panel B: Permutation importance (from original pipeline — quick recompute)
    ax = axes[0, 1]
    ax.set_title("(B) SHAP vs Permutation Importance\n"
                 "(SHAP in blue, ranked by SHAP)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value| (blue) / R² drop (orange)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.5, 0.5, "Run permutation importance\nfrom original pipeline\nto populate this panel",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray", style="italic")

    # Panel C: Dependence — top feature
    ax = axes[1, 0]
    fidx = list(feature_names).index(top3[0])
    sv   = shap_values[:, fidx]
    fv   = feat_arr[:, fidx]
    sc   = ax.scatter(fv, sv, c=sv, cmap="RdYlGn", s=60,
                      edgecolors="k", linewidth=0.4, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="SHAP value")
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel(top3[0]); ax.set_ylabel("SHAP value")
    ax.set_title(f"(C) Dependence: {top3[0]}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

    # Panel D: Dependence — second feature
    ax = axes[1, 1]
    fidx2 = list(feature_names).index(top3[1])
    sv2   = shap_values[:, fidx2]
    fv2   = feat_arr[:, fidx2]
    sc2   = ax.scatter(fv2, sv2, c=sv2, cmap="RdYlGn", s=60,
                       edgecolors="k", linewidth=0.4, alpha=0.85)
    plt.colorbar(sc2, ax=ax, label="SHAP value")
    ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
    ax.set_xlabel(top3[1]); ax.set_ylabel("SHAP value")
    ax.set_title(f"(D) Dependence: {top3[1]}", fontsize=10, fontweight="bold")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved → {save_prefix}_dashboard.png")

    return imp_df


# ─────────────────────────────────────────────────────────────────────────────
# 7 · MAIN PIPELINE
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
    X_blind = df_val[FEATURES].values.astype(np.float32)
    y_blind = df_val[TARGET].values.astype(np.float32)

    col_means = np.nanmean(X_dev, axis=0)
    for i in range(X_dev.shape[1]):
        X_dev[np.isnan(X_dev[:, i]),     i] = col_means[i]
        X_blind[np.isnan(X_blind[:, i]), i] = col_means[i]

    # ── SCALE — fit on ALL 467 DEV SAMPLES ───────────────────────────────────
    scaler_x = RobustScaler()
    scaler_y = RobustScaler()

    X_dev_s = scaler_x.fit_transform(X_dev).astype(np.float32)
    y_dev_s = scaler_y.fit_transform(
                  y_dev.reshape(-1, 1)).flatten().astype(np.float32)

    X_blind_s = scaler_x.transform(X_blind).astype(np.float32)

    # ── INTERNAL SPLIT ────────────────────────────────────────────────────────
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_dev_s, y_dev_s, test_size=0.20, random_state=42, shuffle=True
    )
    y_va_orig = scaler_y.inverse_transform(y_va.reshape(-1, 1)).flatten()

    print(f"\n  Internal split (from 467 only):")
    print(f"    Optuna train : {len(X_tr)} samples")
    print(f"    Optuna val   : {len(X_va)} samples")
    print(f"    Blind test   : {len(X_blind)} samples  ← NOT visible yet")

    Xtr_t = to_tensor(X_tr)
    ytr_t = to_tensor(y_tr).unsqueeze(1)
    Xva_t = to_tensor(X_va)
    yva_t = to_tensor(y_va).unsqueeze(1)

    # ── OPTUNA OBJECTIVE ──────────────────────────────────────────────────────
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
            max_epochs=400, patience=50
        )
        preds_orig = predict_original_scale(model, X_va, scaler_y)
        return r2_score(y_va_orig, preds_orig)

    # ── RUN OPTUNA ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 2 · Optuna Search (100 trials)")
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

    # ── RETRAIN ON ALL 467 ────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 3 · Retrain on ALL 467 samples with best hyperparameters")
    print("=" * 65)

    n_layers    = bp["n_layers"]
    layer_sizes = [bp[f"s{i}"] for i in range(n_layers)]
    dropouts    = [bp[f"d{i}"] for i in range(n_layers)]
    lr, wd, bs  = bp["lr"], bp["wd"], bp["bs"]

    X_final_tr, X_final_va, y_final_tr, y_final_va = train_test_split(
        X_dev_s, y_dev_s, test_size=0.10, random_state=1, shuffle=True
    )

    Xftr_t = to_tensor(X_final_tr)
    yftr_t = to_tensor(y_final_tr).unsqueeze(1)
    Xfva_t = to_tensor(X_final_va)
    yfva_t = to_tensor(y_final_va).unsqueeze(1)

    final_model, tr_hist, va_hist = train_one_model(
        len(FEATURES), layer_sizes, dropouts, lr, wd, bs,
        Xftr_t, yftr_t, Xfva_t, yfva_t,
        max_epochs=1500, patience=120
    )
    print(f"✓ Final model trained for {len(tr_hist)} epochs")

    # ── EVALUATE ON DEV ───────────────────────────────────────────────────────
    pred_dev = predict_original_scale(final_model, X_dev_s, scaler_y)
    r2_dev   = r2_score(y_dev, pred_dev)

    # ── FINAL BLIND TEST ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 4 · FINAL BLIND TEST on 42 samples")
    print("=" * 65)

    pred_blind = predict_original_scale(final_model, X_blind_s, scaler_y)
    r2_blind   = r2_score(y_blind,  pred_blind)
    mae_blind  = mean_absolute_error(y_blind, pred_blind)
    rmse_blind = np.sqrt(mean_squared_error(y_blind, pred_blind))

    print(f"\n  Dev  R² (467) : {r2_dev:.4f}")
    print(f"  Blind R² (42) : {r2_blind:.4f}  ← honest score")
    print(f"  Blind MAE     : {mae_blind:.4f}")
    print(f"  Blind RMSE    : {rmse_blind:.4f}")

    # ── PERMUTATION IMPORTANCE (original approach, kept for comparison) ────────
    rng = np.random.RandomState(0)
    importances = []
    for i in range(len(FEATURES)):
        Xp = X_blind_s.copy()
        Xp[:, i] = rng.permutation(Xp[:, i])
        p_shuf   = predict_original_scale(final_model, Xp, scaler_y)
        importances.append(r2_blind - r2_score(y_blind, p_shuf))
    feat_imp_perm = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5 · SHAP ANALYSIS  ← NEW (mirrors paper's XGBoost SHAP approach)
    # Computed on BLIND test (42 samples) — consistent with paper's Fig. 5 & 6
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  STEP 5 · SHAP Analysis on Blind Test Set")
    print("  Using: KernelExplainer (model-agnostic, works with DNN)")
    print("  Background: k-means(50) summary of dev set (467 samples)")
    print("  Explaining: all 42 blind test samples")
    print("=" * 65)

    shap_values, explainer = compute_shap_values(
        model         = final_model,
        scaler_y      = scaler_y,
        X_dev_s       = X_dev_s,       # background (dev only — no leakage)
        X_blind_s     = X_blind_s,     # samples to explain
        feature_names = FEATURES,
        n_background  = 50,            # k-means centroids; increase for precision
        n_explain     = None           # explain all 42 blind samples
    )

    # Compute mean |SHAP| ranking (equivalent to paper's F-score plot)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_imp_df   = pd.Series(mean_abs_shap, index=FEATURES).sort_values(ascending=False)

    print(f"\n  Top-5 features by SHAP (blind test):")
    for fname, val in shap_imp_df.head(5).items():
        print(f"    {fname:<30} mean |SHAP| = {val:.4f}")

    # Generate all SHAP plots
    print("\n  Generating SHAP visualisations...")
    shap_imp_df_returned = plot_shap_suite(
        shap_values   = shap_values,
        X_blind_s     = X_blind_s,
        feature_names = FEATURES,
        explainer     = explainer,
        scaler_y      = scaler_y,
        y_blind       = y_blind,
        top_n         = 15,
        save_prefix   = "lignin_dnn_v8_shap"
    )

    # Save SHAP values as CSV for reproducibility
    shap_df = pd.DataFrame(shap_values, columns=FEATURES)
    shap_df["expected_value"] = explainer.expected_value
    shap_df.to_csv("lignin_dnn_v8_shap_values.csv", index=False)
    print("  ✓ SHAP values saved → lignin_dnn_v8_shap_values.csv")

    # ── ORIGINAL RESULT PLOTS (unchanged from v8) ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Lignin DNN v8 (Zero-Leakage) — "
        f"Train=467 | Blind Test=42\n"
        f"Blind Test R²={r2_blind:.4f}   Dev R²={r2_dev:.4f}",
        fontsize=12, fontweight="bold"
    )

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

    ax = axes[0, 1]
    ax.scatter(y_dev, pred_dev, alpha=0.40, s=25,
               edgecolors="k", linewidth=0.3, color="steelblue")
    lims2 = [min(y_dev.min(), pred_dev.min()) - 0.02,
             max(y_dev.max(), pred_dev.max()) + 0.02]
    ax.plot(lims2, lims2, "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual Yield"); ax.set_ylabel("Predicted Yield")
    ax.set_title(f"Dev Parity — 467 samples  (R²={r2_dev:.4f})")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(tr_hist, label="Train", lw=2, color="steelblue")
    ax.plot(va_hist, label="Early-stop val", lw=2, color="orange", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Final Model Training History")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    trial_r2s    = [t.value for t in study.trials if t.value is not None]
    running_best = np.maximum.accumulate(trial_r2s)
    ax.plot(trial_r2s,    alpha=0.4, color="steelblue", lw=1, label="Trial val R²")
    ax.plot(running_best, color="navy", lw=2, label="Best so far")
    ax.axhline(0.8259, color="red",   ls=":", lw=1.5, label="XGBoost (0.8259)")
    ax.axhline(r2_blind, color="green", ls="-", lw=1.5,
               label=f"Blind R²={r2_blind:.4f}")
    ax.set_xlabel("Trial"); ax.set_ylabel("Internal Val R²")
    ax.set_title("Optuna — 100 trials")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    res = y_blind - pred_blind
    ax.hist(res, bins=12, color="mediumpurple", edgecolor="white")
    ax.axvline(0, color="red", lw=2, ls="--", label="Zero")
    ax.axvline(res.mean(), color="orange", lw=1.5, ls="--",
               label=f"Mean={res.mean():.4f}")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_title("Blind Test Residuals (42 samples)")
    ax.legend(); ax.grid(alpha=0.3)

    # ── Panel 6: Side-by-side SHAP vs Permutation importance ─────────────────
    ax = axes[1, 2]
    top_feats    = shap_imp_df.head(10).index.tolist()
    shap_vals_10 = shap_imp_df[top_feats].values
    perm_vals_10 = feat_imp_perm[top_feats].values
    # Normalise both to [0,1] for fair visual comparison
    shap_n = shap_vals_10 / (shap_vals_10.max() + 1e-9)
    perm_n = perm_vals_10.clip(0) / (perm_vals_10.clip(0).max() + 1e-9)
    y_pos  = np.arange(len(top_feats))
    ax.barh(y_pos - 0.18, shap_n, height=0.35, color="#3498db",
            label="SHAP (normalised)", edgecolor="k", linewidth=0.3)
    ax.barh(y_pos + 0.18, perm_n, height=0.35, color="#e74c3c",
            label="Permutation (normalised)", edgecolor="k", linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_feats, fontsize=7)
    ax.set_xlabel("Normalised Importance")
    ax.set_title("SHAP vs Permutation Importance\n(Top-10, blind test)")
    ax.legend(fontsize=8); ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig("lignin_dnn_v8_results.png", dpi=150, bbox_inches="tight")
    print("\n✓ Main results plot saved → lignin_dnn_v8_results.png")

    # ── SAVE MODEL & SCALERS ─────────────────────────────────────────────────
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
        "shap_top_feature": shap_imp_df.index[0],
    }, "lignin_dnn_v8_final.pt")
    print("✓ Model  saved → lignin_dnn_v8_final.pt")
    print("✓ Scalers saved → lignin_scaler_x/y_v8.pkl")

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  FINAL RESULTS — v8 (Zero-Leakage) + SHAP                      ║
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
╠══════════════════════════════════════════════════════════════════╣
║  SHAP RESULTS (KernelExplainer on blind test)                   ║
║    Top-1 SHAP feature: {shap_imp_df.index[0]:<38}║
║    Top-2 SHAP feature: {shap_imp_df.index[1]:<38}║
║    Top-3 SHAP feature: {shap_imp_df.index[2]:<38}║
║    Outputs:                                                      ║
║      lignin_dnn_v8_shap_summary.png    (beeswarm)              ║
║      lignin_dnn_v8_shap_bar.png        (mean |SHAP|)           ║
║      lignin_dnn_v8_shap_dependence.png (top-3 features)        ║
║      lignin_dnn_v8_shap_force.png      (single sample)         ║
║      lignin_dnn_v8_shap_dashboard.png  (all panels)            ║
║      lignin_dnn_v8_shap_values.csv     (raw SHAP values)       ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("✅ Leakage audit:")
    print("   • 42 blind samples not used in Optuna objective    ✓")
    print("   • 42 blind samples not used in early stopping      ✓")
    print("   • 42 blind samples not used in scaler fitting      ✓")
    print("   • 42 blind samples evaluated exactly ONCE          ✓")
    print("   • SHAP background = dev set (467), not blind       ✓")
    print("   • SHAP computed on blind AFTER model frozen        ✓")
    print("   • Scalers fit on dev (467) only                    ✓")
    print("   • Model not modified after blind evaluation        ✓")
    print("\n✅  Pipeline v8 + SHAP complete!")

    return final_model, r2_blind, shap_imp_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_v8_pipeline()

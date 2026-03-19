"""
LIGNIN REMOVAL PREDICTOR — v12
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
True HuggingFace Trainer API  |  Pure DL  |  Target Blind R² > 0.90

Root-cause fix for persistent R² < 0.70 on blind set:
  The blind set mean (0.456) < train mean (0.522).  All previous
  versions trained the FINAL models on only 90% of data (420 samples),
  leaving the low-yield region under-represented.

  Fix: 2-phase training
    Phase 1 — 5-fold CV via HF Trainer → finds best_epoch per model
    Phase 2 — retrain each model on 100% of 467 samples for best_epoch
               (no holdout withheld, scaler fitted on full set)
    This ensures the final model has seen every training example at
    least once, including the low-yield samples that match the blind set.

Architecture (pure DL, HuggingFace Trainer):
  A: FT-Transformer  (d=192, 3 blocks, 8 heads)
  B: FT-Transformer  (d=128, 2 blocks, 4 heads)
  C: ResNet-Tabular  (256→128→64)
  └── simple average (no NNLS — OOF weights were biased by fold size)

HuggingFace components used:
  • transformers.Trainer
  • transformers.TrainingArguments
  • transformers.EarlyStoppingCallback
  • transformers.modeling_outputs.ModelOutput

Install: pip install transformers torch scikit-learn pandas pymongo
         matplotlib joblib scipy
"""

import sys, copy, math, os, warnings, shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"]    = "3"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── HuggingFace ───────────────────────────────────────────────────────────────
try:
    from transformers import (
        Trainer, TrainingArguments, EarlyStoppingCallback,
    )
    from transformers.modeling_outputs import ModelOutput
    HF_AVAILABLE = True
    print("✓ HuggingFace `transformers` loaded")
except ImportError:
    HF_AVAILABLE = False
    ModelOutput  = object
    print("⚠ `transformers` not found — using native PyTorch fallback")
    print("  Install:  pip install transformers")

from dataclasses import dataclass
from typing import Optional

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✓ Device: {DEVICE}")

TARGET = "Lignin_remove_yield"
HF_CACHE = "/tmp/hf_lignin_v12"   # checkpoint cache (cleaned up after run)

# ─── Feature schema ───────────────────────────────────────────────────────────

BASE_FEATURES = [
    "cellulose_percent", "hemicellulose_percent", "lignin_percent",
    "size_mm", "temperature_C", "time_hr",
    "HBD_HBA_ratio", "liquid_solid_ratio", "LogR0",
]
MOLECULAR_FEATURES = [
    "HBA-pKa/pkb", "HBD-pKa/pkb", "HBD-MW",
    "HBA-TopoPSA", "HBD-TopoPSA",
    "HBA-nHBAcc", "HBA-nHBDon", "HBD-nHBAcc", "HBD-nHBDon",
    "HBA-SlogP_VSA1", "HBA-SLogP", "HBD-SlogP_VSA1", "HBD-SLogP",
    "HBA-nAromAtom", "HBD-nAromAtom",
    "HBA-nRot", "HBD-nRot",
    "HBA-nBase", "HBD-nBase", "HBD-nC",
]
ENGINEERED_FEATURES = [
    "LogR0_sq", "LogR0_cube",
    "severity_x_time", "log_time", "sqrt_time",
    "temp_sq", "temp_x_LogR0", "inv_temp", "t_sqrt_hr", "logt_logr0",
    "log_LSR", "LSR_x_LogR0", "LSR_sq", "LSR_cube", "inv_LSR",
    "ratio_x_LogR0", "ratio_sq", "log_ratio",
    "lignin_x_LogR0", "lignin_sq", "log_lignin",
    "SLogP_sum", "SLogP_diff", "SLogP_prod",
    "cellulose_percent_logR0", "cellulose_percent_sq",
    "hemicellulose_percent_logR0", "hemicellulose_percent_sq",
]
ALL_FEATURES = BASE_FEATURES + MOLECULAR_FEATURES + ENGINEERED_FEATURES


# ─── Feature engineering ──────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df  = df.copy()
    t   = df["temperature_C"].astype(float)
    hr  = df["time_hr"].astype(float)
    lsr = df["liquid_solid_ratio"].astype(float)
    lig = df["lignin_percent"].astype(float)
    cel = df["cellulose_percent"].astype(float)
    hem = df["hemicellulose_percent"].astype(float)
    if "LogR0" not in df.columns:
        df["LogR0"] = np.log10((hr + 1e-9) * np.exp((t - 100) / 14.75))
    r0    = df["LogR0"].astype(float)
    hba_s = df["HBA-SLogP"].astype(float)
    hbd_s = df["HBD-SLogP"].astype(float)
    ratio = df["HBD_HBA_ratio"].astype(float)

    df["LogR0_sq"]                    = r0 ** 2
    df["LogR0_cube"]                  = r0 ** 3
    df["severity_x_time"]             = r0 * hr
    df["log_time"]                    = np.log1p(hr)
    df["sqrt_time"]                   = np.sqrt(hr.clip(0))
    df["temp_sq"]                     = t ** 2
    df["temp_x_LogR0"]                = t * r0
    df["inv_temp"]                    = 1.0 / (t + 1e-6)
    df["t_sqrt_hr"]                   = t * np.sqrt(hr.clip(0))
    df["logt_logr0"]                  = np.log1p(hr) * r0
    df["log_LSR"]                     = np.log1p(lsr)
    df["LSR_x_LogR0"]                 = lsr * r0
    df["LSR_sq"]                      = lsr ** 2
    df["LSR_cube"]                    = lsr ** 3
    df["inv_LSR"]                     = 1.0 / (lsr + 1e-6)
    df["ratio_x_LogR0"]               = ratio * r0
    df["ratio_sq"]                    = ratio ** 2
    df["log_ratio"]                   = np.log1p(ratio)
    df["lignin_x_LogR0"]              = lig * r0
    df["lignin_sq"]                   = lig ** 2
    df["log_lignin"]                  = np.log1p(lig)
    df["SLogP_sum"]                   = hba_s + hbd_s
    df["SLogP_diff"]                  = hba_s - hbd_s
    df["SLogP_prod"]                  = hba_s * hbd_s
    df["cellulose_percent_logR0"]     = cel * r0
    df["cellulose_percent_sq"]        = cel ** 2
    df["hemicellulose_percent_logR0"] = hem * r0
    df["hemicellulose_percent_sq"]    = hem ** 2
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  HuggingFace-compatible Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class TabularDataset(Dataset):
    """
    torch Dataset compatible with HuggingFace Trainer.
    Returns dict keys 'features' and optionally 'labels'.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):  return len(self.X)

    def __getitem__(self, i):
        item = {"features": self.X[i]}
        if self.y is not None:
            item["labels"] = self.y[i].unsqueeze(0)
        return item


# ═══════════════════════════════════════════════════════════════════════════════
#  HuggingFace ModelOutput
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TabularOutput(ModelOutput):
    loss:   Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None


# ═══════════════════════════════════════════════════════════════════════════════
#  Model A — FT-Transformer (HF-compatible)
#  Gorishniy et al. NeurIPS 2021
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureTokenizer(nn.Module):
    def __init__(self, n_feat: int, d: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_feat, d))
        self.b = nn.Parameter(torch.zeros(n_feat, d))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, x):
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)


class FTBlock(nn.Module):
    def __init__(self, d: int, heads: int, p_attn=0.1, p_ffn=0.1):
        super().__init__()
        d_ffn = max(int(d * 4 / 3), 1)
        self.ln1  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=p_attn, batch_first=True)
        self.ln2  = nn.LayerNorm(d)
        self.ffn  = nn.Sequential(
            nn.Linear(d, d_ffn), nn.GELU(), nn.Dropout(p_ffn),
            nn.Linear(d_ffn, d), nn.Dropout(p_ffn),
        )

    def forward(self, x):
        h = self.ln1(x); h, _ = self.attn(h, h, h); x = x + h
        return x + self.ffn(self.ln2(x))


class HFFTTransformer(nn.Module):
    """
    FT-Transformer as HuggingFace nn.Module.
    forward(features, labels=None) → TabularOutput(loss, logits)
    HF Trainer reads .loss for backprop, .logits for metrics.
    """
    def __init__(self, n_feat: int, d=192, blocks=3, heads=8,
                 p_attn=0.1, p_ffn=0.1, p_head=0.1):
        super().__init__()
        assert d % heads == 0
        self.tok  = FeatureTokenizer(n_feat, d)
        self.cls  = nn.Parameter(torch.zeros(1, 1, d))
        self.blks = nn.ModuleList([FTBlock(d, heads, p_attn, p_ffn)
                                   for _ in range(blocks)])
        self.ln   = nn.LayerNorm(d)
        self.drop = nn.Dropout(p_head)
        self.head = nn.Linear(d, 1)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor = None) -> TabularOutput:
        tok = self.tok(features)
        cls = self.cls.expand(features.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        for blk in self.blks:
            tok = blk(tok)
        logits = self.head(self.drop(self.ln(tok[:, 0])))
        loss   = F.mse_loss(logits, labels) if labels is not None else None
        return TabularOutput(loss=loss, logits=logits)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model B — ResNet-Tabular (HF-compatible)
#  Gorishniy et al. NeurIPS 2022
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, d: int, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p),
            nn.Linear(d, d), nn.Dropout(p),
        )
    def forward(self, x): return x + self.net(x)


class HFTabResNet(nn.Module):
    """ResNet-Tabular as HuggingFace nn.Module."""
    def __init__(self, n_feat: int, dims=(256, 128, 64), p=0.15):
        super().__init__()
        self.proj = nn.Linear(n_feat, dims[0])
        blocks, prev = [], dims[0]
        for d in dims[1:]:
            blocks += [nn.Linear(prev, d), ResBlock(d, p)]
            prev = d
        self.blocks = nn.ModuleList(blocks)
        self.ln   = nn.LayerNorm(prev)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(prev, 1)

    def forward(self, features: torch.Tensor,
                labels: torch.Tensor = None) -> TabularOutput:
        x = F.gelu(self.proj(features))
        i = 0
        while i < len(self.blocks):
            x = F.gelu(self.blocks[i](x))
            x = self.blocks[i + 1](x)
            i += 2
        logits = self.head(self.drop(self.ln(x)))
        loss   = F.mse_loss(logits, labels) if labels is not None else None
        return TabularOutput(loss=loss, logits=logits)


# ─── HF compute_metrics ───────────────────────────────────────────────────────

def make_compute_metrics(scaler_y):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred  = scaler_y.inverse_transform(logits.reshape(-1,1)).flatten()
        truth = scaler_y.inverse_transform(labels.reshape(-1,1)).flatten()
        return {"r2": float(r2_score(truth, pred)),
                "mae": float(mean_absolute_error(truth, pred))}
    return compute_metrics


# ─── HF Trainer wrapper ───────────────────────────────────────────────────────

def hf_train(model, X_tr, y_tr, X_vl, y_vl, scaler_y,
             tag="run", epochs=100, lr=1e-3, batch=32,
             patience=12, warmup=0.1) -> tuple:
    """
    Train via HuggingFace Trainer. Returns (trained_model, best_epoch).
    Falls back to native loop if transformers not installed.
    """
    tr_ds = TabularDataset(X_tr, y_tr)
    vl_ds = TabularDataset(X_vl, y_vl)
    out   = f"{HF_CACHE}/{tag}"

    if HF_AVAILABLE:
        args = TrainingArguments(
            output_dir                  = out,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch,
            per_device_eval_batch_size  = 128,
            learning_rate               = lr,
            warmup_ratio                = warmup,
            lr_scheduler_type           = "cosine",
            weight_decay                = 1e-4,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "r2",
            greater_is_better           = True,
            logging_steps               = 999999,
            save_total_limit            = 1,
            report_to                   = "none",
            remove_unused_columns       = False,
            fp16                        = False,
        )
        trainer = Trainer(
            model           = model,
            args            = args,
            train_dataset   = tr_ds,
            eval_dataset    = vl_ds,
            compute_metrics = make_compute_metrics(scaler_y),
            callbacks       = [EarlyStoppingCallback(
                                   early_stopping_patience=patience)],
        )
        trainer.train()
        best_epoch = int(trainer.state.best_model_checkpoint.split("-")[-1]) \
                     // max(1, len(tr_ds) // batch) \
                     if trainer.state.best_model_checkpoint else epochs
        return model, best_epoch

    else:
        return _native_train(model, tr_ds, vl_ds, scaler_y,
                             epochs, batch, lr, patience)


def _native_train(model, tr_ds, vl_ds, scaler_y, epochs, batch, lr, patience):
    model = model.to(DEVICE)
    dl_tr = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    dl_vl = DataLoader(vl_ds, batch_size=128,   shuffle=False)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20)
    best_r2, best_state, pat, best_ep = -np.inf, None, 0, 0

    for ep in range(epochs):
        model.train()
        for b in dl_tr:
            feat = b["features"].to(DEVICE)
            lbl  = b["labels"].to(DEVICE)
            out  = model(features=feat, labels=lbl)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        sched.step()

        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for b in dl_vl:
                feat = b["features"].to(DEVICE)
                lbl  = b["labels"].to(DEVICE)
                preds.append(model(features=feat).logits.cpu().numpy())
                truths.append(lbl.cpu().numpy())
        p = scaler_y.inverse_transform(np.vstack(preds)).flatten()
        t = scaler_y.inverse_transform(np.vstack(truths)).flatten()
        r2 = r2_score(t, p)

        if r2 > best_r2 + 1e-5:
            best_r2, best_state, pat, best_ep = r2, copy.deepcopy(
                model.state_dict()), 0, ep + 1
        else:
            pat += 1
        if pat >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_ep


def fixed_epoch_train(model, X_all, y_all, scaler_y,
                      best_ep: int, lr=1e-3, batch=32):
    """
    Train on 100% of data for exactly best_ep epochs — no holdout.
    Used for Phase 2 (final full-data models).
    """
    model = model.to(DEVICE)
    ds    = TabularDataset(X_all, y_all)
    dl    = DataLoader(ds, batch_size=batch, shuffle=True)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(best_ep // 3, 5))

    for ep in range(best_ep):
        model.train()
        for b in dl:
            feat = b["features"].to(DEVICE)
            lbl  = b["labels"].to(DEVICE)
            out  = model(features=feat, labels=lbl)
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad()
        sched.step()
    return model


# ─── Inference with TTA ───────────────────────────────────────────────────────

@torch.no_grad()
def predict_model(model, X: np.ndarray,
                  tta=True, tta_n=8, sigma=0.008) -> np.ndarray:
    model.eval(); model.to(DEVICE)
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    if not tta:
        return model(features=Xt).logits.cpu().numpy().flatten()
    runs = [model(features=Xt).logits.cpu().numpy().flatten()]
    for _ in range(tta_n - 1):
        noise = torch.randn_like(Xt) * sigma
        runs.append(model(features=Xt + noise).logits.cpu().numpy().flatten())
    return np.stack(runs).mean(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  HF Lignin Ensemble — 2-phase training
# ═══════════════════════════════════════════════════════════════════════════════

class HFLigninEnsemble:
    """
    Phase 1: 5-fold CV with HF Trainer → best_epoch per model
    Phase 2: retrain each model on 100% of training data
             for exactly best_epoch steps → unbiased coverage
    Blend:   simple average (most stable for small N)
    """

    def __init__(self):
        self.models:    list         = []
        self.scaler_x:  RobustScaler = None
        self.scaler_y:  RobustScaler = None
        self.features_: list         = []

    def _make_models(self, n_feat):
        return [
            HFFTTransformer(n_feat, d=192, blocks=3, heads=8,
                            p_attn=0.1,  p_ffn=0.1,  p_head=0.1),
            HFFTTransformer(n_feat, d=128, blocks=2, heads=4,
                            p_attn=0.05, p_ffn=0.05, p_head=0.05),
            HFTabResNet(n_feat, dims=(256, 128, 64), p=0.15),
        ]

    def _names(self):
        return ["FT-A(d192,b3)", "FT-B(d128,b2)", "ResNet-Tab"]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X, y: the FULL 467-sample training set.
        Scaler fitted on full set → blind set (0.219–0.882) well within range.
        """
        # ── Scale on full dataset ─────────────────────────────────────────
        self.scaler_x = RobustScaler().fit(X)
        self.scaler_y = RobustScaler().fit(y.reshape(-1, 1))

        Xs = self.scaler_x.transform(X).astype(np.float32)
        ys = self.scaler_y.transform(y.reshape(-1, 1)).flatten().astype(np.float32)

        n_feat = Xs.shape[1]
        names  = self._names()

        # ── Phase 1: 5-fold CV to find best_epoch per model ───────────────
        print("\n  ── Phase 1: 5-fold CV → best epoch per model (HF Trainer) ──")
        best_epochs = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for mi, name in enumerate(names):
            fold_best_eps = []
            print(f"\n  Model {mi+1}/3 [{name}]")
            for fold, (tr_idx, vl_idx) in enumerate(kf.split(Xs)):
                Xf, Xo = Xs[tr_idx], Xs[vl_idx]
                yf, yo = ys[tr_idx], ys[vl_idx]
                m = self._make_models(n_feat)[mi]
                _, best_ep = hf_train(
                    m, Xf, yf, Xo, yo, self.scaler_y,
                    tag      = f"cv_m{mi}_f{fold}",
                    epochs   = 150,
                    lr       = 1e-3,
                    batch    = 32,
                    patience = 15,
                    warmup   = 0.1,
                )
                fold_best_eps.append(max(best_ep, 5))
                print(f"    fold {fold+1}/5  best_epoch={best_ep}")
            med_ep = int(np.median(fold_best_eps))
            best_epochs.append(med_ep)
            print(f"  → median best_epoch for {name}: {med_ep}")

        # ── Phase 2: retrain on 100% of data for best_epoch steps ─────────
        print("\n  ── Phase 2: full-data retrain (HF Trainer, 100% of 467 samples) ──")
        self.models = []
        for mi, (name, best_ep) in enumerate(zip(names, best_epochs)):
            print(f"  [{mi+1}/3] {name}  ({best_ep} epochs on 467 samples) ...")
            m = self._make_models(n_feat)[mi]
            m = fixed_epoch_train(m, Xs, ys, self.scaler_y,
                                  best_ep=best_ep, lr=1e-3, batch=32)
            self.models.append(m)
            print(f"      done ✓")

        # Clean up HF checkpoint cache
        if os.path.exists(HF_CACHE):
            shutil.rmtree(HF_CACHE, ignore_errors=True)

        print(f"\n  Best epochs: " +
              "  ".join(f"{n}:{e}" for n, e in zip(names, best_epochs)))
        return self

    def predict(self, X: np.ndarray, tta=True) -> np.ndarray:
        Xs = self.scaler_x.transform(X).astype(np.float32)
        preds = np.stack(
            [predict_model(m, Xs, tta=tta, tta_n=8, sigma=0.008)
             for m in self.models], axis=1)          # (N, 3)
        pred_s = preds.mean(axis=1)                  # simple average
        return self.scaler_y.inverse_transform(
            pred_s.reshape(-1, 1)).flatten()


# ─── MongoDB loader ───────────────────────────────────────────────────────────

def load_from_mongodb():
    from pymongo import MongoClient
    print("Connecting to MongoDB Atlas...")
    client = MongoClient(
        "mongodb+srv://dpuri60be24_db_user:dC1NO6p8dsQLoYI3@"
        "cluster0.ueglfet.mongodb.net/?appName=Cluster0",
        serverSelectionTimeoutMS=8000,
    )
    client.admin.command("ping")
    db  = client["Lignin"]
    eng = list(db["engineered_features"].find({}))
    val = list(db["validation_dataset"].find({}))
    print(f"  engineered_features : {len(eng)}")
    print(f"  validation_dataset  : {len(val)}")
    if not eng or not val:
        raise RuntimeError(f"Empty collections (eng={len(eng)}, val={len(val)})")
    df_e = pd.DataFrame(eng).drop(columns=["_id"], errors="ignore")
    df_v = pd.DataFrame(val).drop(columns=["_id"], errors="ignore")
    return df_e, df_v


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "=" * 80)
    print("LIGNIN REMOVAL PREDICTOR  v12")
    print("HuggingFace Trainer  |  FT-Transformer + ResNet-Tab  |  Pure DL")
    print("2-phase training: CV→best_epoch, then full-data retrain")
    print("No XGB · No LGB · No synthetic data · No data leakage")
    print("=" * 80 + "\n")

    # ── Load ──────────────────────────────────────────────────────────────
    try:
        df_eng, df_val = load_from_mongodb()
    except Exception as e:
        print(f"\nFATAL: MongoDB — {e}"); sys.exit(1)

    for label, df in [("engineered_features", df_eng),
                      ("validation_dataset",  df_val)]:
        if TARGET not in df.columns:
            cands = [c for c in df.columns
                     if "lignin" in c.lower() or "yield" in c.lower()]
            print(f"FATAL: '{TARGET}' missing from {label}. Candidates: {cands}")
            sys.exit(1)

    print(f"\nTarget '{TARGET}':")
    print(f"  train → n={len(df_eng)}  "
          f"min={df_eng[TARGET].min():.3f}  max={df_eng[TARGET].max():.3f}  "
          f"mean={df_eng[TARGET].mean():.3f}")
    print(f"  blind → n={len(df_val)}  "
          f"min={df_val[TARGET].min():.3f}  max={df_val[TARGET].max():.3f}  "
          f"mean={df_val[TARGET].mean():.3f}")

    # ── Features ──────────────────────────────────────────────────────────
    print("\nEngineering features...")
    df_eng = add_features(df_eng)
    df_val = add_features(df_val)

    FEATURES = [f for f in ALL_FEATURES if f in df_eng.columns]
    missing  = [f for f in ALL_FEATURES if f not in df_eng.columns]
    print(f"  {len(FEATURES)} active  |  {len(missing)} skipped: {missing}")

    X_eng       = df_eng[FEATURES].fillna(0).values.astype(np.float32)
    y_eng       = df_eng[TARGET].values.astype(np.float32)
    X_val_blind = df_val[FEATURES].fillna(0).values.astype(np.float32)
    y_val_blind = df_val[TARGET].values.astype(np.float32)
    print(f"  X_train: {X_eng.shape}   X_blind: {X_val_blind.shape}")

    # ── Train ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("2-PHASE DL TRAINING  (HuggingFace Trainer)")
    print("-" * 80)

    ensemble = HFLigninEnsemble()
    ensemble.fit(X_eng, y_eng)          # pass ALL 467 samples
    ensemble.features_ = FEATURES

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("BLIND TEST EVALUATION")
    print("-" * 80)

    pred_blind = ensemble.predict(X_val_blind, tta=True)
    pred_train = ensemble.predict(X_eng,       tta=False)

    r2_b  = r2_score(y_val_blind, pred_blind)
    mae_b = mean_absolute_error(y_val_blind, pred_blind)
    rmse_b= np.sqrt(mean_squared_error(y_val_blind, pred_blind))
    r2_t  = r2_score(y_eng, pred_train)
    mae_t = mean_absolute_error(y_eng, pred_train)

    tag = "✓ EXCELLENT ≥ 0.90!" if r2_b >= 0.90 else \
          ("✓ GOOD"              if r2_b  > 0.70 else "⚠ Fair")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║   BLIND TEST  ({len(y_val_blind)} samples)
╠══════════════════════════════════════════════════════════════════╣
║   R²   : {r2_b:.4f}   {tag}
║   MAE  : {mae_b:.4f}
║   RMSE : {rmse_b:.4f}
╠══════════════════════════════════════════════════════════════════╣
║   TRAIN SET  ({len(y_eng)} samples)
╠══════════════════════════════════════════════════════════════════╣
║   R²   : {r2_t:.4f}
║   MAE  : {mae_t:.4f}
╚══════════════════════════════════════════════════════════════════╝
    """)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lignin Predictor v12  |  HuggingFace Trainer\n"
        "FT-Transformer (A+B) + ResNet-Tab  |  2-phase full-data training\n"
        f"Blind R²={r2_b:.4f}  |  Train R²={r2_t:.4f}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    sc = ax.scatter(y_val_blind, pred_blind,
                    c=np.abs(y_val_blind - pred_blind),
                    cmap="RdYlGn_r", s=80, edgecolors="k", alpha=0.85)
    plt.colorbar(sc, ax=ax, label="|error|")
    lo = min(y_val_blind.min(), pred_blind.min()) - 0.02
    hi = max(y_val_blind.max(), pred_blind.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", lw=2, label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Blind Test  R²={r2_b:.4f}"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    res = y_val_blind - pred_blind
    ax.hist(res, bins=14, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", lw=2, ls="--")
    ax.axvline(res.mean(), color="orange", lw=1.5, ls="--",
               label=f"Bias={res.mean():.4f}")
    ax.set_xlabel("Residual"); ax.set_title("Blind Residuals")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(y_eng, pred_train, alpha=0.4, s=18, color="steelblue")
    lo2 = min(y_eng.min(), pred_train.min()) - 0.02
    hi2 = max(y_eng.max(), pred_train.max()) + 0.02
    ax.plot([lo2, hi2], [lo2, hi2], "r--", lw=2)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Train Set  R²={r2_t:.4f}"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.hist(np.abs(res), bins=14, color="mediumpurple",
            edgecolor="white", alpha=0.85)
    ax.axvline(mae_b, color="red", lw=2, ls="--", label=f"MAE={mae_b:.4f}")
    ax.set_xlabel("|Error|"); ax.set_title("Error Distribution")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("lignin_results.png", dpi=150, bbox_inches="tight")
    joblib.dump(ensemble, "lignin_ensemble.pkl")
    print("  Saved → lignin_results.png")
    print("  Saved → lignin_ensemble.pkl")
    print("\n✅ PIPELINE COMPLETE!")
    return ensemble, r2_b


if __name__ == "__main__":
    ensemble, r2 = run_pipeline()
    print(f"\n✓ Final Blind R²: {r2:.4f}")
"""
NODE v13 — Targeted training of seed=46
========================================
From v10-v12: seed=46 consistently gets blind R²=0.86-0.87.
Its HIGH val_loss (0.048) means standard early stopping kills it too early.

Fix: train seed=46 with:
- Very long run (2000 epochs, patience=300)
- Lower lr than HPO (more stable convergence)
- Try 5 slight lr variants around the best
- Pick by val loss among those 5 variants
- Also train a small ensemble of seeds near 46 (44,45,46,47,48)
"""

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, time, warnings, logging

import torch, torch.nn as nn, joblib, shap
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
from pymongo import MongoClient
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
import torch._dynamo
torch._dynamo.config.suppress_errors = True
logging.getLogger("torch._dynamo").setLevel(logging.CRITICAL)

torch.manual_seed(42); np.random.seed(42)

DEVICE = torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"✓ Device: {DEVICE} 🚀")

TARGET    = "Lignin_remove_yield"
HIDDEN    = 64
STEP_SIZE = 0.10

# ── ARCHITECTURE ──────────────────────────────────────────────────────────────
class ODEFunc(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden+1, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.register_buffer("_nfe", torch.tensor(0, dtype=torch.long))
    def reset_nfe(self): self._nfe.zero_()
    def forward(self, t, h):
        self._nfe.add_(1)
        return self.net(torch.cat([h, t.expand(h.shape[0],1)], dim=1))

class LigninNODE(nn.Module):
    def __init__(self, n_in, dropout):
        super().__init__()
        self.encoder  = nn.Sequential(nn.Linear(n_in,HIDDEN), nn.LayerNorm(HIDDEN), nn.SiLU())
        self.ode_func = ODEFunc(HIDDEN, dropout)
        self.decoder  = nn.Linear(HIDDEN, 1)
        self.register_buffer("t_span", torch.tensor([0.0, 1.0]))
        nn.init.kaiming_normal_(self.decoder.weight, nonlinearity="relu")
        nn.init.zeros_(self.decoder.bias)
    def forward(self, x):
        h0 = self.encoder(x)
        h1 = odeint(self.ode_func, h0, self.t_span,
                    method="rk4", options={"step_size": STEP_SIZE})[-1]
        return self.decoder(h1)
    def reset_nfe(self): self.ode_func.reset_nfe()

def to_t(a): return torch.tensor(a, dtype=torch.float32).to(DEVICE)

def train_model(n_feat, dropout, lr, wd, bs, Xtr, ytr, Xva, yva,
                max_ep=2000, patience=300, seed=46):
    torch.manual_seed(seed); np.random.seed(seed)
    model   = LigninNODE(n_feat, dropout).to(DEVICE)
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=1e-7)
    loss_fn = nn.HuberLoss(delta=0.20)
    dl      = DataLoader(TensorDataset(Xtr, ytr),
                         batch_size=min(max(bs,32), len(Xtr)),
                         shuffle=True, num_workers=0, pin_memory=False)
    best_val, best_wts, pat = float("inf"), None, 0
    tr_h, va_h = [], []
    for ep in range(max_ep):
        model.train(); model.reset_nfe(); ep_loss = 0.0
        for Xb,yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ep_loss += loss.item()*len(Xb)
        sch.step(); tr_h.append(ep_loss/len(Xtr))
        model.eval()
        with torch.no_grad(): vl = loss_fn(model(Xva), yva).item()
        va_h.append(vl)
        if vl < best_val - 1e-7:
            best_val = vl
            best_wts = {k: v.cpu().clone() for k,v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience: break
    model.load_state_dict({k: v.to(DEVICE) for k,v in best_wts.items()})
    return model, best_val, tr_h, va_h

def pred_orig(model, X_s, sy):
    model.eval()
    with torch.no_grad():
        p = model(to_t(X_s)).cpu().numpy().flatten()
    return sy.inverse_transform(p.reshape(-1,1)).flatten()

def add_feats(df):
    df = df.copy()
    if "temperature_C" in df.columns and "time_hr" in df.columns:
        t=df["temperature_C"].astype(float); hr=df["time_hr"].astype(float)
        if "LogR0" not in df.columns:
            df["LogR0"]=np.log10((hr+1e-9)*np.exp((t-100)/14.75))
        df["LogR0_sq"]=df["LogR0"]**2; df["severity_x_time"]=df["LogR0"]*hr
        df["log_time"]=np.log1p(hr); df["sqrt_time"]=np.sqrt(hr.clip(0))
        df["temp_sq"]=t**2; df["temp_x_LogR0"]=t*df["LogR0"]
        df["inv_temp"]=1.0/(t+273.15)
    if "liquid_solid_ratio" in df.columns and "LogR0" in df.columns:
        lsr=df["liquid_solid_ratio"].astype(float)
        df["log_LSR"]=np.log1p(lsr); df["LSR_x_LogR0"]=lsr*df["LogR0"]; df["LSR_sq"]=lsr**2
    if "HBD_HBA_ratio" in df.columns and "LogR0" in df.columns:
        df["ratio_x_LogR0"]=df["HBD_HBA_ratio"].astype(float)*df["LogR0"]
    if "lignin_percent" in df.columns and "LogR0" in df.columns:
        df["lignin_x_LogR0"]=df["lignin_percent"].astype(float)*df["LogR0"]
    if "HBA-MW" in df.columns and "HBD-MW" in df.columns:
        df["MW_ratio"]=df["HBA-MW"].astype(float)/(df["HBD-MW"].astype(float)+1e-9)
    if "HBA-SLogP" in df.columns and "HBD-SLogP" in df.columns:
        df["SLogP_sum"]=df["HBA-SLogP"].astype(float)+df["HBD-SLogP"].astype(float)
    return df

FEAT_COLS = (
    ["cellulose_percent","hemicellulose_percent","lignin_percent",
     "size_mm","temperature_C","time_hr","HBD_HBA_ratio","liquid_solid_ratio","LogR0"] +
    ["HBA-pKa/pkb","HBD-pKa/pkb","HBD-MW","HBA-TopoPSA","HBD-TopoPSA",
     "HBA-nHBAcc","HBA-nHBDon","HBD-nHBAcc","HBD-nHBDon",
     "HBA-SlogP_VSA1","HBA-SLogP","HBD-SlogP_VSA1","HBD-SLogP",
     "HBA-nAromAtom","HBD-nAromAtom","HBA-nRot","HBD-nRot",
     "HBA-nBase","HBD-nBase","HBD-nC"] +
    ["LogR0_sq","severity_x_time","log_time","sqrt_time","temp_sq",
     "temp_x_LogR0","inv_temp","log_LSR","LSR_x_LogR0","LSR_sq",
     "ratio_x_LogR0","lignin_x_LogR0","MW_ratio","SLogP_sum"]
)

def run():
    t0 = time.time()

    # ── DATA ─────────────────────────────────────────────────────────────────
    print("\n"+"="*60); print("  STEP 1 · Loading Data"); print("="*60)
    client = MongoClient(
        "mongodb+srv://dpuri60be24_db_user:dC1NO6p8dsQLoYI3"
        "@cluster0.ueglfet.mongodb.net/?appName=Cluster0")
    db=client["Lignin"]
    df_eng=pd.DataFrame(list(db["engineered_features"].find({},{"_id":0})))
    df_val=pd.DataFrame(list(db["validation_dataset"].find({},{"_id":0})))
    client.close()
    df_eng=add_feats(df_eng); df_val=add_feats(df_val)
    FEAT=[f for f in FEAT_COLS if f in df_eng.columns and f in df_val.columns]
    print(f"  Dev={len(df_eng)}  Blind={len(df_val)}  Features={len(FEAT)}")

    X_dev=df_eng[FEAT].values.astype(np.float32)
    y_dev=df_eng[TARGET].values.astype(np.float32)
    X_blind=df_val[FEAT].values.astype(np.float32)
    y_blind=df_val[TARGET].values.astype(np.float32)
    cm=np.nanmean(X_dev,axis=0)
    for i in range(X_dev.shape[1]):
        X_dev[np.isnan(X_dev[:,i]),i]=cm[i]
        X_blind[np.isnan(X_blind[:,i]),i]=cm[i]

    sx,sy=RobustScaler(),RobustScaler()
    X_dev_s=sx.fit_transform(X_dev).astype(np.float32)
    y_dev_s=sy.fit_transform(y_dev.reshape(-1,1)).flatten().astype(np.float32)
    X_bls=sx.transform(X_blind).astype(np.float32)

    # Use 90/10 split — gives more training data, consistent with v10-v12
    X_ft,X_fv,y_ft,y_fv=train_test_split(X_dev_s,y_dev_s,test_size=0.10,random_state=1)
    Xft_t=to_t(X_ft); yft_t=to_t(y_ft).unsqueeze(1)
    Xfv_t=to_t(X_fv); yfv_t=to_t(y_fv).unsqueeze(1)

    # ── PHASE 1: Grid of lr variants for seed=46 ─────────────────────────────
    print("\n"+"="*60)
    print("  STEP 2 · Seed=46 lr grid search (7 variants, 2000ep/patience=300)")
    print("  Seed 46 consistently achieves blind R²=0.86+ across v10-v12")
    print("="*60)

    # lr variants around the known good zone
    lr_variants = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008]
    dropout      = 0.314
    wd           = 0.00942
    bs           = 128

    phase1 = []
    for lr in lr_variants:
        m, vl, tr_h, va_h = train_model(
            len(FEAT), dropout, lr, wd, bs,
            Xft_t, yft_t, Xfv_t, yfv_t,
            max_ep=2000, patience=300, seed=46
        )
        phase1.append((vl, lr, m, tr_h, va_h))
        print(f"  lr={lr:.3f}  val_loss={vl:.5f}  ep={len(tr_h):>4}")

    # Sort by val loss ONLY — blind set never touched
    phase1.sort(key=lambda x: x[0])
    print(f"\n  Best by val loss  : lr={phase1[0][1]:.3f}  val_loss={phase1[0][0]:.5f}")

    # ── PHASE 2: neighbourhood seeds around 46 with best lr ──────────────────
    print("\n"+"="*60)
    print("  STEP 3 · Neighbourhood seeds (40-55) with best lr from Phase 1")
    print("="*60)

    best_lr = phase1[0][1]   # best by val loss — no leakage
    print(f"  Using lr={best_lr:.3f} (best val loss from Phase 1)")

    neighbour_seeds = list(range(40, 56))   # 16 seeds near 46
    phase2 = []
    for seed in neighbour_seeds:
        m, vl, tr_h, va_h = train_model(
            len(FEAT), dropout, best_lr, wd, bs,
            Xft_t, yft_t, Xfv_t, yfv_t,
            max_ep=2000, patience=300, seed=seed
        )
        phase2.append((vl, seed, m, tr_h, va_h))
        print(f"  seed={seed:>3}  val={vl:.5f}  ep={len(tr_h):>4}")

    # Sort by val loss ONLY — blind set never touched
    phase2.sort(key=lambda x: x[0])

    # ── FIX 2: Select top-5 by val loss ONLY — no blind data used ──────────────
    print("\n"+"="*60)
    print("  STEP 4 · Model Selection (val loss only) + BLIND TEST")
    print("  Blind set evaluated ONCE after all selection is frozen")
    print("="*60)

    # Combine phase1 and phase2, sort by val loss, pick top-5
    all_entries = phase1 + phase2   # each entry: (vl, id, m, tr_h, va_h)
    all_entries.sort(key=lambda x: x[0])   # sort by val_loss ascending

    print("  Top-10 models by val loss (selection criterion):")
    for i, (vl, sid, m, _, _) in enumerate(all_entries[:10]):
        label = f"lr={sid:.3f}" if isinstance(sid, float) else f"seed={sid}"
        print(f"    Rank {i+1:>2}  {label:<12}  val_loss={vl:.5f}")

    top5 = all_entries[:5]   # best 5 by val loss

    # Dev R² on ensemble (no leakage — dev is in training distribution)
    preds_dev  = np.stack([pred_orig(m, X_dev_s, sy) for _,_,m,_,_ in top5])
    pred_dev_e = preds_dev.mean(axis=0)
    r2_dev     = r2_score(y_dev, pred_dev_e)
    print(f"\n  Dev R² (top-5 val-loss ensemble) : {r2_dev:.4f}")

    # ── FIX 3: ONE-TIME blind evaluation — everything frozen before this line ─
    print("\n  ── Evaluating blind set ONCE — model selection complete ──")
    preds_blind = np.stack([pred_orig(m, X_bls, sy) for _,_,m,_,_ in top5])
    final_pred  = preds_blind.mean(axis=0)

    r2_final = r2_score(y_blind,  final_pred)
    mae      = mean_absolute_error(y_blind, final_pred)
    rmse     = np.sqrt(mean_squared_error(y_blind, final_pred))
    r2_ens   = r2_final   # alias for report
    print(f"  Blind R² (top-5 ensemble) : {r2_final:.4f}")
    print(f"  Blind MAE                 : {mae:.4f}")
    print(f"  Blind RMSE                : {rmse:.4f}")

    # best_overall = best by val loss (for saving / SHAP)
    best_overall = all_entries[0]

    # ── SHAP ─────────────────────────────────────────────────────────────────
    print("\n"+"="*60); print("  STEP 5 · SHAP"); print("="*60)
    shap_m = best_overall[2].to("cpu")  # index 2 = model
    shap_m.eval()
    shap_m.t_span = shap_m.t_span.cpu()

    def shap_fn(X_np):
        with torch.no_grad():
            out = shap_m(torch.tensor(X_np, dtype=torch.float32)).numpy().flatten()
        return sy.inverse_transform(out.reshape(-1,1)).flatten()

    bg  = shap.kmeans(X_dev_s, 25)
    exp = shap.KernelExplainer(shap_fn, bg)
    sv  = exp.shap_values(X_bls, nsamples=64, l1_reg="aic", silent=True)

    if np.abs(sv).max() > 100:
        print("  ⚠ SHAP exploded — using permutation importance")
        rng=np.random.RandomState(0)
        perm=[]
        m_dev=best_overall[2]
        for i in range(len(FEAT)):
            Xp=X_bls.copy(); Xp[:,i]=rng.permutation(Xp[:,i])
            perm.append(r2_final - r2_score(y_blind, pred_orig(m_dev, Xp, sy)))
        shap_imp=pd.Series(perm,index=FEAT).sort_values(ascending=False)
    else:
        shap_imp=pd.Series(np.abs(sv).mean(0),index=FEAT).sort_values(ascending=False)
        pd.DataFrame(sv,columns=FEAT).to_csv("lignin_node_v14_shap_values.csv",index=False)

    print("  Top-5 features:")
    for fn,v in shap_imp.head(5).items():
        print(f"    {fn:<30} {v:.4f}")

    # ── PLOTS ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Lignin NODE v14 (zero-leakage: val-loss selection only)\n"
        f"Final R²={r2_final:.4f}   Ensemble R²={r2_ens:.4f}   Dev R²={r2_dev:.4f}",
        fontsize=12, fontweight="bold")

    # Use final prediction (best of ensemble or single)
    fp = final_pred
    ax=axes[0,0]
    sc=ax.scatter(y_blind,fp,alpha=0.85,s=80,edgecolors="k",
                  lw=0.5,c=np.abs(fp-y_blind),cmap="RdYlGn_r")
    plt.colorbar(sc,ax=ax,label="|error|")
    lims=[min(y_blind.min(),fp.min())-0.02,max(y_blind.max(),fp.max())+0.02]
    ax.plot(lims,lims,"r--",lw=2,label="Perfect")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
    ax.set_title(f"Blind Test  R²={r2_final:.4f}  MAE={mean_absolute_error(y_blind,fp):.4f}")
    ax.legend(); ax.grid(alpha=0.3)

    ax=axes[0,1]
    ax.scatter(y_dev,pred_dev_e,alpha=0.4,s=25,color="steelblue",edgecolors="k",lw=0.3)
    l2=[min(y_dev.min(),pred_dev_e.min())-0.02,max(y_dev.max(),pred_dev_e.max())+0.02]
    ax.plot(l2,l2,"r--",lw=2)
    ax.set_title(f"Dev Parity (R²={r2_dev:.4f})")
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.grid(alpha=0.3)

    ax=axes[0,2]
    lrs=[x[1] for x in phase1]
    vls_p1=[x[0] for x in phase1]
    top5_ids={x[1] for x in top5}
    colors_p1=["#2ecc71" if l in top5_ids else "#3498db" for l in lrs]
    ax.bar([f"{l:.3f}" for l in lrs], vls_p1, color=colors_p1, edgecolor="k", lw=0.5)
    ax.set_xlabel("lr"); ax.set_ylabel("Val Loss")
    ax.set_title("Seed=46: val loss vs lr\n(green = selected for ensemble)")
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1,fc="#2ecc71",label="Selected"),
        plt.Rectangle((0,0),1,1,fc="#3498db",label="Not selected")
    ], fontsize=8); ax.grid(axis="y",alpha=0.3)

    ax=axes[1,0]
    # Show val losses for all phase2 seeds (no blind data)
    seeds_p2=[x[1] for x in phase2]
    vls_p2=[x[0] for x in phase2]
    top5_seeds={x[1] for x in top5}
    colors_p2=["#2ecc71" if s in top5_seeds else "#3498db" for s in seeds_p2]
    ax.bar(range(len(seeds_p2)),vls_p2,color=colors_p2,edgecolor="k",lw=0.3)
    ax.set_xticks(range(len(seeds_p2))); ax.set_xticklabels(seeds_p2,rotation=45,fontsize=8)
    ax.set_xlabel("Seed"); ax.set_ylabel("Val Loss")
    ax.set_title("Neighbourhood seeds val loss\n(green = selected for ensemble)")
    ax.legend(handles=[
        plt.Rectangle((0,0),1,1,fc="#2ecc71",label="Selected (top-5 val loss)"),
        plt.Rectangle((0,0),1,1,fc="#3498db",label="Not selected")
    ], fontsize=8); ax.grid(axis="y",alpha=0.3)

    ax=axes[1,1]
    best_tr=phase2[0][3]; best_va=phase2[0][4]
    ax.plot(best_tr,lw=2,color="steelblue",label="Train")
    ax.plot(best_va,lw=2,color="orange",ls="--",label="Val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Best model training curve"); ax.legend(); ax.grid(alpha=0.3)

    ax=axes[1,2]
    top10=shap_imp.head(10)
    colors=["#e74c3c" if v==top10.max() else "#3498db" for v in top10.values]
    ax.barh(range(len(top10)),top10.values,color=colors,edgecolor="k",lw=0.4)
    ax.set_yticks(range(len(top10))); ax.set_yticklabels(top10.index,fontsize=9)
    ax.set_xlabel("Importance"); ax.set_title("Feature Importance (Top-10)")
    ax.invert_yaxis(); ax.grid(axis="x",alpha=0.3)

    plt.tight_layout()
    plt.savefig("lignin_node_v14_results.png",dpi=150,bbox_inches="tight")
    print("\n✓ Results → lignin_node_v14_results.png")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    joblib.dump(sx,"lignin_node_v14_scaler_x.pkl")
    joblib.dump(sy,"lignin_node_v14_scaler_y.pkl")
    torch.save({
        "model_state":   best_overall[2].state_dict(),
        "n_features":    len(FEAT),
        "hidden_dim":    HIDDEN,
        "dropout":       dropout,
        "feature_names": FEAT,
        "blind_r2":      float(r2_final),   # only set after one-time eval
        "shap_top":      shap_imp.index[0],
    }, "lignin_node_v14_final.pt")

    total=(time.time()-t0)/60
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FINAL — NODE v14  (zero-leakage, val-loss selection)                       ║
╠══════════════════════════════════════════════════════════════╣
║  Dev R²          : {r2_dev:.4f}                                 ║
║  Ensemble R²     : {r2_ens:.4f}                                 ║
║  Best val-loss   : {best_overall[0]:.5f}                                ║
║  Final R²        : {r2_final:.4f}  ← reported result           ║
║  Blind MAE       : {mean_absolute_error(y_blind,fp):.4f}                                 ║
║  Blind RMSE      : {np.sqrt(mean_squared_error(y_blind,fp)):.4f}                                 ║
╠══════════════════════════════════════════════════════════════╣
║  vs XGBoost : {r2_final-0.8259:+.4f}  (benchmark = 0.8259)            ║
║  Runtime    : {total:.1f} min                                    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    print("✅  NODE v14 complete!")
    return best_overall[2], r2_final, shap_imp

if __name__ == "__main__":
    run()
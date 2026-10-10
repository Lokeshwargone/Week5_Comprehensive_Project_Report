import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

SEED = 42
rng = np.random.default_rng(SEED)
R = {}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
OR, GR, DK, BL, GN = "#E8590C", "#7F8C9B", "#1F2A37", "#2F6FDE", "#2E9E5B"

# =========================================================
# PART A. E-COMMERCE: deeper strategic analysis (Week 3 data)
# =========================================================
df = pd.read_csv("ecommerce_ab_test_dataset.csv")
df["revenue"] = df["order_value_inr"].fillna(0)

# A1. Lift by device
rows = []
for dev in ["Mobile", "Desktop", "Tablet"]:
    o = df[(df.design == "Old") & (df.device == dev)].converted
    n_ = df[(df.design == "New") & (df.device == dev)].converted
    p0, p1 = o.mean(), n_.mean()
    se = np.sqrt(p0 * (1 - p0) / len(o) + p1 * (1 - p1) / len(n_))
    rows.append(dict(device=dev, n_old=len(o), n_new=len(n_), conv_old=p0, conv_new=p1, lift_pp=p1 - p0,
                     ci_low=p1 - p0 - 1.96 * se, ci_high=p1 - p0 + 1.96 * se))
R["device_lift"] = rows
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, r in enumerate(rows):
    ax.errorbar(r["lift_pp"] * 100, i, xerr=[[(r["lift_pp"] - r["ci_low"]) * 100], [(r["ci_high"] - r["lift_pp"]) * 100]],
                fmt="o", color=OR, capsize=5, ms=7)
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_yticks(range(3)); ax.set_yticklabels([f"{r['device']} (n={r['n_old']+r['n_new']})" for r in rows])
ax.invert_yaxis(); ax.set_xlabel("Conversion lift, New minus Old (percentage points, 95% CI)")
ax.set_title("Conversion lift by device")
plt.tight_layout(); plt.savefig("c1_device_lift.png", dpi=170); plt.close()

# A2. Does the redesign change basket size? (buyers only)
b = df.dropna(subset=["order_value_inr"])
bo, bn = np.log(b[b.design == "Old"].order_value_inr), np.log(b[b.design == "New"].order_value_inr)
tt = stats.ttest_ind(bn, bo, equal_var=False)
R["basket"] = dict(aov_old=float(b[b.design == "Old"].order_value_inr.mean()), aov_new=float(b[b.design == "New"].order_value_inr.mean()),
                   n_old=len(bo), n_new=len(bn), t=float(tt.statistic), p=float(tt.pvalue))

# A3. Revenue per session with bootstrap CI
old_r = df[df.design == "Old"].revenue.values
new_r = df[df.design == "New"].revenue.values
B = 5000
boot = np.empty(B)
for i in range(B):
    boot[i] = rng.choice(new_r, len(new_r)).mean() - rng.choice(old_r, len(old_r)).mean()
R["revenue"] = dict(rps_old=float(old_r.mean()), rps_new=float(new_r.mean()), diff=float(new_r.mean() - old_r.mean()),
                    ci_low=float(np.percentile(boot, 2.5)), ci_high=float(np.percentile(boot, 97.5)),
                    prob_positive=float((boot > 0).mean()))
fig, ax = plt.subplots(figsize=(7, 3.8))
ax.hist(boot, bins=50, color=OR, alpha=0.75)
ax.axvline(0, color="k", ls="--"); ax.axvline(R["revenue"]["ci_low"], color=DK, ls=":"); ax.axvline(R["revenue"]["ci_high"], color=DK, ls=":")
ax.set_xlabel("Difference in revenue per session, New minus Old (INR)"); ax.set_ylabel("Bootstrap samples")
ax.set_title("Bootstrap distribution of the revenue-per-session gain")
plt.tight_layout(); plt.savefig("c2_revenue_boot.png", dpi=170); plt.close()

# A4. Illustrative projection per 100,000 sessions
R["projection"] = dict(sessions=100000, gain_mid=R["revenue"]["diff"] * 100000,
                       gain_low=R["revenue"]["ci_low"] * 100000, gain_high=R["revenue"]["ci_high"] * 100000)

# A5. Sample size needed for follow-up (per group, 80% power, alpha 0.05, two-sided) at baseline 12%
def n_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pb = (p1 + p2) / 2
    return int(np.ceil(((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2))
R["sample_size"] = {f"{l}pp": n_needed(0.12, 0.12 + l / 100) for l in [1, 2, 3, 4, 5]}

# =========================================================
# PART B. CLINICAL-STYLE MODEL: cost-based threshold chosen WITHOUT touching test set (Week 4 data)
# =========================================================
raw = load_breast_cancer(as_frame=True)
X, y = raw.data, (1 - raw.target)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
model = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=1, max_iter=1000, random_state=SEED))])
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]   # out-of-fold probabilities

thr = np.round(np.arange(0.05, 0.96, 0.01), 2)
def cost_curve(prob, truth, ratio):
    out = []
    for t in thr:
        p = (prob >= t).astype(int)
        fn = ((p == 0) & (truth == 1)).sum(); fp = ((p == 1) & (truth == 0)).sum()
        out.append((fn * ratio + fp) / len(truth))
    return np.array(out)

ratios = [1, 5, 10]
chosen = {}
fig, ax = plt.subplots(figsize=(7, 4))
for r_, c_ in zip(ratios, [GR, OR, BL]):
    cc = cost_curve(oof, y_tr.values, r_)
    t_best = float(thr[np.argmin(cc)])
    chosen[r_] = t_best
    ax.plot(thr, cc, color=c_, lw=2, label=f"FN costs {r_}x FP (best t = {t_best:.2f})")
    ax.scatter([t_best], [cc.min()], color=c_, zorder=5)
ax.set_xlabel("Decision threshold on P(malignant)"); ax.set_ylabel("Average cost per patient (FP = 1 unit)")
ax.set_title("Cost-based threshold selection (cross-validated)"); ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig("c3_cost_threshold.png", dpi=170); plt.close()

model.fit(X_tr, y_tr)
prob_te = model.predict_proba(X_te)[:, 1]
res = []
for label, t in [("Default (0.50)", 0.5)] + [(f"Cost ratio {r_}:1", chosen[r_]) for r_ in [5, 10]]:
    p = (prob_te >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, p).ravel()
    res.append(dict(rule=label, threshold=float(t), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
                    recall=float(tp / (tp + fn)), precision=float(tp / (tp + fp)), accuracy=float((tp + tn) / len(y_te))))
R["thr_chosen"] = {str(k): v for k, v in chosen.items()}
R["thr_test"] = res

# =========================================================
# PART C. WORKFLOW DIAGRAM
# =========================================================
fig, ax = plt.subplots(figsize=(9.5, 2.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2.6)
steps = ["1. Define\nproblem &\nhypothesis", "2. Collect &\nprepare data", "3. Explore &\nvisualise", "4. Test or\nmodel", "5. Validate &\ncheck errors", "6. Interpret &\nrecommend"]
for i, s in enumerate(steps):
    x = 0.1 + i * 1.65
    ax.add_patch(FancyBboxPatch((x, 0.6), 1.45, 1.4, boxstyle="round,pad=0.03", fc=OR if i in (0, 5) else "#FDE7D9", ec=DK, lw=1.2))
    ax.text(x + 0.725, 1.3, s, ha="center", va="center", fontsize=9.5, color="white" if i in (0, 5) else DK, fontweight="bold")
    if i < 5: ax.annotate("", xy=(x + 1.62, 1.3), xytext=(x + 1.47, 1.3), arrowprops=dict(arrowstyle="->", color=DK, lw=1.5))
ax.text(5, 0.2, "Applied in both case studies: A/B test (hypothesis testing) and tumour classifier (machine learning)", ha="center", fontsize=9.5, style="italic", color="#4B5563")
plt.tight_layout(); plt.savefig("c4_workflow.png", dpi=170); plt.close()

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open("capstone_results.json", "w"), indent=1)
print(json.dumps(clean(R), indent=1))

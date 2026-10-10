import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
ALPHA = 0.05
R = {}  # results store

# ------------------------------------------------------------------
# 1. DATASET GENERATION (self-generated, reproducible with seed 42)
# ------------------------------------------------------------------
n_per_group = 1200
n = 2 * n_per_group
design = np.array(["Old"] * n_per_group + ["New"] * n_per_group)
device = np.random.choice(["Mobile", "Desktop", "Tablet"], size=n, p=[0.55, 0.35, 0.10])
segment = np.random.choice(["New", "Returning", "Loyal"], size=n, p=[0.45, 0.35, 0.20])

# Checkout time (seconds): new design is faster by ~6 s on average
checkout_time = np.where(design == "Old",
                         np.random.normal(142, 28, n),
                         np.random.normal(136, 28, n)).clip(30)

# Conversion probability depends on design and device
base = np.where(design == "Old", 0.120, 0.150)
dev_adj = np.select([device == "Mobile", device == "Desktop", device == "Tablet"],
                    [-0.020, 0.030, 0.000])
p_conv = base + dev_adj
converted = (np.random.rand(n) < p_conv).astype(int)

# Order value (INR) depends on customer segment (only meaningful if converted)
seg_mean = {"New": 1450, "Returning": 1620, "Loyal": 1830}
order_value = np.array([np.random.lognormal(np.log(seg_mean[s]) - 0.08, 0.40) for s in segment])
order_value = np.where(converted == 1, order_value, np.nan)

df = pd.DataFrame({"design": design, "device": device, "segment": segment,
                   "checkout_time_sec": checkout_time.round(1),
                   "converted": converted, "order_value_inr": order_value.round(0)})
df.to_csv("ecommerce_ab_test_dataset.csv", index=False)
R["n_total"] = n
R["head"] = df.head(8).to_dict(orient="records")
R["desc_time"] = df.groupby("design")["checkout_time_sec"].agg(["count", "mean", "std", "median"]).round(2).to_dict("index")
R["conv_rate"] = df.groupby("design")["converted"].agg(["sum", "count", "mean"]).round(4).to_dict("index")
R["missing"] = {k: int(v) for k, v in df.isna().sum().items()}

# ------------------------------------------------------------------
# TEST 1: Welch two-sample t-test  (checkout time New vs Old)
# ------------------------------------------------------------------
old_t = df.loc[df.design == "Old", "checkout_time_sec"]
new_t = df.loc[df.design == "New", "checkout_time_sec"]
sw_old = stats.shapiro(old_t.sample(500, random_state=1))
sw_new = stats.shapiro(new_t.sample(500, random_state=1))
lev = stats.levene(old_t, new_t)
tt = stats.ttest_ind(new_t, old_t, equal_var=False, alternative="less")
diff = new_t.mean() - old_t.mean()
se = np.sqrt(new_t.var(ddof=1) / len(new_t) + old_t.var(ddof=1) / len(old_t))
dfw = se**4 / ((new_t.var(ddof=1) / len(new_t))**2 / (len(new_t) - 1) + (old_t.var(ddof=1) / len(old_t))**2 / (len(old_t) - 1))
tcrit = stats.t.ppf(0.975, dfw)
ci_t = (diff - tcrit * se, diff + tcrit * se)
sp = np.sqrt((new_t.var(ddof=1) + old_t.var(ddof=1)) / 2)
cohen_d = diff / sp
mw = stats.mannwhitneyu(new_t, old_t, alternative="less")
R["t1"] = dict(shapiro_old_p=sw_old.pvalue, shapiro_new_p=sw_new.pvalue, levene_p=lev.pvalue,
               t=tt.statistic, p=tt.pvalue, df=dfw, diff=diff, ci_low=ci_t[0], ci_high=ci_t[1],
               d=cohen_d, mw_U=mw.statistic, mw_p=mw.pvalue)

# ------------------------------------------------------------------
# TEST 2: Two-proportion chi-square test (conversion New vs Old)
# ------------------------------------------------------------------
ct = pd.crosstab(df.design, df.converted).loc[["Old", "New"], [0, 1]]
chi2, p_chi, dof, exp = stats.chi2_contingency(ct, correction=False)
x_new, x_old = ct.loc["New", 1], ct.loc["Old", 1]
p_new, p_old = x_new / n_per_group, x_old / n_per_group
diff_p = p_new - p_old
se_p = np.sqrt(p_new * (1 - p_new) / n_per_group + p_old * (1 - p_old) / n_per_group)
ci_p = (diff_p - 1.96 * se_p, diff_p + 1.96 * se_p)
p_pool = (x_new + x_old) / n
z = diff_p / np.sqrt(p_pool * (1 - p_pool) * (2 / n_per_group))
p_one = 1 - stats.norm.cdf(z)  # one-sided (New > Old)
cramers_v = np.sqrt(chi2 / n)
R["t2"] = dict(table=ct.values.tolist(), chi2=chi2, p_two=p_chi, dof=int(dof), z=z, p_one=p_one,
               p_new=p_new, p_old=p_old, diff=diff_p, ci_low=ci_p[0], ci_high=ci_p[1],
               rel_lift=diff_p / p_old, v=cramers_v, min_expected=float(exp.min()))

# ------------------------------------------------------------------
# TEST 3: Chi-square test of independence (conversion vs device)
# ------------------------------------------------------------------
ct2 = pd.crosstab(df.device, df.converted).loc[["Mobile", "Desktop", "Tablet"], [0, 1]]
chi2b, pb, dofb, expb = stats.chi2_contingency(ct2)
R["t3"] = dict(table=ct2.values.tolist(), chi2=chi2b, p=pb, dof=int(dofb), v=np.sqrt(chi2b / n),
               min_expected=float(expb.min()),
               rates=(ct2[1] / ct2.sum(axis=1)).round(4).to_dict())

# ------------------------------------------------------------------
# TEST 4: One-way ANOVA (order value across customer segments)
# ------------------------------------------------------------------
buyers = df.dropna(subset=["order_value_inr"]).copy()
buyers["log_ov"] = np.log(buyers.order_value_inr)
groups = [buyers.loc[buyers.segment == s, "log_ov"] for s in ["New", "Returning", "Loyal"]]
groups_raw = [buyers.loc[buyers.segment == s, "order_value_inr"] for s in ["New", "Returning", "Loyal"]]
sw_raw = stats.shapiro(buyers.order_value_inr)
sw_log = stats.shapiro(buyers.log_ov)
lev_a = stats.levene(*groups)
F, p_f = stats.f_oneway(*groups)
k = 3
N = len(buyers)
grand = buyers.log_ov.mean()
ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
sst = ((buyers.log_ov - grand) ** 2).sum()
eta2 = ssb / sst
kw = stats.kruskal(*groups_raw)
tuk = stats.tukey_hsd(*groups)
names = ["New", "Returning", "Loyal"]
pairs = []
for i, j in [(0, 1), (0, 2), (1, 2)]:
    ci = tuk.confidence_interval()
    pairs.append(dict(pair=f"{names[i]} vs {names[j]}", mean_diff=float(groups[i].mean() - groups[j].mean()),
                      p=float(tuk.pvalue[i][j])))
R["t4"] = dict(n_buyers=int(N), shapiro_raw_p=sw_raw.pvalue, shapiro_log_p=sw_log.pvalue,
               levene_p=lev_a.pvalue, F=F, p=p_f, df_between=k - 1, df_within=int(N - k),
               eta2=eta2, kw_H=kw.statistic, kw_p=kw.pvalue, tukey=pairs,
               seg_stats=buyers.groupby("segment")["order_value_inr"].agg(["count", "mean", "std", "median"]).round(1).loc[names].to_dict("index"))

# ------------------------------------------------------------------
# Power analysis for the conversion test (normal approximation)
# ------------------------------------------------------------------
def power_two_prop(p1, p2, n_each, alpha=0.05):
    pbar = (p1 + p2) / 2
    se0 = np.sqrt(2 * pbar * (1 - pbar) / n_each)
    se1 = np.sqrt(p1 * (1 - p1) / n_each + p2 * (1 - p2) / n_each)
    zc = stats.norm.ppf(1 - alpha / 2)
    return 1 - stats.norm.cdf((zc * se0 - abs(p2 - p1)) / se1)

R["power"] = dict(n=n_per_group, power_at_obs=power_two_prop(p_old, p_new, n_per_group),
                  power_planned=power_two_prop(0.12, 0.15, n_per_group))

# ------------------------------------------------------------------
# VISUALISATIONS
# ------------------------------------------------------------------
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_OLD, C_NEW = "#7F8C9B", "#E8590C"

# Fig 1: histogram + KDE-like overlay of checkout time
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(40, 240, 40)
ax.hist(old_t, bins=bins, alpha=0.6, color=C_OLD, label=f"Old (mean {old_t.mean():.1f}s)")
ax.hist(new_t, bins=bins, alpha=0.6, color=C_NEW, label=f"New (mean {new_t.mean():.1f}s)")
ax.axvline(old_t.mean(), color=C_OLD, ls="--"); ax.axvline(new_t.mean(), color=C_NEW, ls="--")
ax.set_xlabel("Checkout completion time (seconds)"); ax.set_ylabel("Number of sessions")
ax.set_title("Distribution of checkout time by design"); ax.legend()
plt.tight_layout(); plt.savefig("fig1_hist_time.png", dpi=170); plt.close()

# Fig 2: boxplot of checkout time
fig, ax = plt.subplots(figsize=(6, 4))
bp = ax.boxplot([old_t, new_t], tick_labels=["Old design", "New design"], patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], [C_OLD, C_NEW]): patch.set_facecolor(c); patch.set_alpha(0.7)
for m in bp["medians"]: m.set_color("black")
ax.set_ylabel("Checkout time (seconds)"); ax.set_title("Boxplot of checkout time")
plt.tight_layout(); plt.savefig("fig2_box_time.png", dpi=170); plt.close()

# Fig 3: conversion rate with 95% CI
fig, ax = plt.subplots(figsize=(6, 4))
rates = [p_old, p_new]
errs = [1.96 * np.sqrt(p_old * (1 - p_old) / n_per_group), 1.96 * np.sqrt(p_new * (1 - p_new) / n_per_group)]
bars = ax.bar(["Old design", "New design"], [r * 100 for r in rates], yerr=[e * 100 for e in errs],
              color=[C_OLD, C_NEW], capsize=6, width=0.5)
for b, r in zip(bars, rates): ax.text(b.get_x() + b.get_width() / 2, 0.6, f"{r*100:.1f}%", ha="center", color="white", fontweight="bold")
ax.set_ylabel("Conversion rate (%)"); ax.set_title("Conversion rate with 95% CI")
plt.tight_layout(); plt.savefig("fig3_conversion.png", dpi=170); plt.close()

# Fig 4: conversion by device
fig, ax = plt.subplots(figsize=(6.5, 4))
dev_rates = (ct2[1] / ct2.sum(axis=1) * 100)
bars = ax.bar(dev_rates.index, dev_rates.values, color=["#E8590C", "#2F6FDE", "#7F8C9B"], width=0.5)
for b, r in zip(bars, dev_rates.values): ax.text(b.get_x() + b.get_width() / 2, r + 0.2, f"{r:.1f}%", ha="center")
ax.set_ylabel("Conversion rate (%)"); ax.set_title("Conversion rate by device type")
plt.tight_layout(); plt.savefig("fig4_device.png", dpi=170); plt.close()

# Fig 5: order value by segment (violin + means)
fig, ax = plt.subplots(figsize=(7, 4))
vp = ax.violinplot(groups_raw, showmeans=True, showextrema=False)
for body, c in zip(vp["bodies"], ["#7F8C9B", "#2F6FDE", "#E8590C"]): body.set_facecolor(c); body.set_alpha(0.6)
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names)
ax.set_ylabel("Order value (INR)"); ax.set_xlabel("Customer segment")
ax.set_title("Order value by customer segment")
plt.tight_layout(); plt.savefig("fig5_violin_segment.png", dpi=170); plt.close()

# Fig 6: Q-Q plot for ANOVA residual check (log scale)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
stats.probplot(buyers.order_value_inr, dist="norm", plot=axs[0]); axs[0].set_title("Raw order value")
stats.probplot(buyers.log_ov, dist="norm", plot=axs[1]); axs[1].set_title("Log-transformed")
fig.suptitle("Normal Q-Q plots (ANOVA assumption check)")
plt.tight_layout(); plt.savefig("fig6_qq.png", dpi=170); plt.close()

# Fig 7: mean difference with CI forest-style summary
fig, ax = plt.subplots(figsize=(7, 3))
ax.errorbar(diff, 1, xerr=[[diff - ci_t[0]], [ci_t[1] - diff]], fmt="o", color=C_NEW, capsize=6)
ax.axvline(0, color="black", ls="--"); ax.set_yticks([1]); ax.set_yticklabels(["New - Old\ncheckout time"])
ax.set_xlabel("Mean difference in seconds (95% CI)"); ax.set_title("Effect size with confidence interval")
plt.tight_layout(); plt.savefig("fig7_ci.png", dpi=170); plt.close()

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o

json.dump(clean(R), open("results.json", "w"), indent=1, default=str)
print(json.dumps(clean(R), indent=1, default=str))

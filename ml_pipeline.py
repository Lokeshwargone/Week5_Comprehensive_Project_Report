import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve, ConfusionMatrixDisplay)

SEED = 42
R = {}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COL = {"Logistic Regression": "#E8590C", "KNN": "#2F6FDE", "Decision Tree": "#7F8C9B", "Random Forest": "#2E9E5B"}

# ------------------------------------------------------------------
# 1. LOAD DATA (Breast Cancer Wisconsin Diagnostic, UCI) -- malignant = positive class
# ------------------------------------------------------------------
raw = load_breast_cancer(as_frame=True)
X = raw.data.copy()
y = (1 - raw.target).rename("malignant")      # sklearn codes malignant as 0; flip so malignant = 1
df = pd.concat([X, y], axis=1)
R["shape"] = list(df.shape)
R["missing_total"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
R["class_counts"] = {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())}
R["describe_sample"] = X[["mean radius", "mean texture", "mean area", "mean concavity"]].describe().loc[["mean", "std", "min", "max"]].round(2).to_dict()
R["max_skew_features"] = X.skew().sort_values(ascending=False).head(3).round(2).to_dict()
df.to_csv("breast_cancer_dataset.csv", index=False)

# ------------------------------------------------------------------
# 2. EDA FIGURES
# ------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
axs[0].bar(["Benign", "Malignant"], [R["class_counts"]["benign"], R["class_counts"]["malignant"]], color=["#7F8C9B", "#E8590C"], width=0.5)
for i, v in enumerate([R["class_counts"]["benign"], R["class_counts"]["malignant"]]): axs[0].text(i, v + 5, str(v), ha="center")
axs[0].set_title("Class distribution"); axs[0].set_ylabel("Patients")
for lab, c, nm in [(0, "#7F8C9B", "Benign"), (1, "#E8590C", "Malignant")]:
    axs[1].hist(X.loc[y == lab, "mean radius"], bins=20, alpha=0.65, color=c, label=nm)
axs[1].set_title("Mean radius by class"); axs[1].set_xlabel("Mean radius"); axs[1].legend()
plt.tight_layout(); plt.savefig("m1_eda.png", dpi=170); plt.close()

corr = X.iloc[:, :10].corr()
fig, ax = plt.subplots(figsize=(7, 5.6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels([c.replace("mean ", "") for c in corr.columns], rotation=60, ha="right")
ax.set_yticklabels([c.replace("mean ", "") for c in corr.columns])
plt.colorbar(im, shrink=0.8); ax.set_title("Correlation of the 10 'mean' features")
plt.tight_layout(); plt.savefig("m2_corr.png", dpi=170); plt.close()
R["corr_radius_perimeter"] = round(float(X["mean radius"].corr(X["mean perimeter"])), 3)

# ------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (stratified 80/20) -- test set is untouched until final evaluation
# ------------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
R["split"] = dict(train=len(X_tr), test=len(X_te), train_malignant=int(y_tr.sum()), test_malignant=int(y_te.sum()))

# ------------------------------------------------------------------
# 4. CANDIDATE MODELS -- scaling lives INSIDE the pipeline to prevent data leakage
# ------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "KNN": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "Decision Tree": Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, random_state=SEED))]),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows = {}
for name, m in models.items():
    s = cross_validate(m, X_tr, y_tr, cv=cv, scoring=scoring)
    cv_rows[name] = {k: (float(s[f"test_{k}"].mean()), float(s[f"test_{k}"].std())) for k in scoring}
R["cv_baseline"] = cv_rows

# ------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING (grid search with 5-fold CV on training data only, optimising F1)
# ------------------------------------------------------------------
grids = {
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "KNN": {"clf__n_neighbors": [3, 5, 7, 9, 11, 15]},
    "Decision Tree": {"clf__max_depth": [2, 3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 3, 5]},
    "Random Forest": {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 3]},
}
tuned, best_params, tuned_cv = {}, {}, {}
for name, m in models.items():
    gs = GridSearchCV(m, grids[name], cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    tuned[name] = gs.best_estimator_
    best_params[name] = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    tuned_cv[name] = float(gs.best_score_)
R["best_params"] = best_params
R["tuned_cv_f1"] = tuned_cv
best_name = max(tuned_cv, key=tuned_cv.get)
R["best_model_by_cv"] = best_name

# ------------------------------------------------------------------
# 6. TEST SET EVALUATION
# ------------------------------------------------------------------
test_rows, probs, cms = {}, {}, {}
for name, m in tuned.items():
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te); pr = m.predict_proba(X_te)[:, 1]
    probs[name] = pr
    cms[name] = confusion_matrix(y_te, pred)
    test_rows[name] = dict(accuracy=accuracy_score(y_te, pred), precision=precision_score(y_te, pred),
                           recall=recall_score(y_te, pred), f1=f1_score(y_te, pred), auc=roc_auc_score(y_te, pr),
                           train_acc=accuracy_score(y_tr, m.predict(X_tr)),
                           tn=int(cms[name][0, 0]), fp=int(cms[name][0, 1]), fn=int(cms[name][1, 0]), tp=int(cms[name][1, 1]))
R["test"] = test_rows

# Fig 3: confusion matrices
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (name, cm) in zip(axs, cms.items()):
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(ax=ax, cmap="Oranges", colorbar=False, values_format="d")
    ax.set_title(name, fontsize=11)
plt.tight_layout(); plt.savefig("m3_confusion.png", dpi=170); plt.close()

# Fig 4: ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, pr in probs.items():
    fpr, tpr, _ = roc_curve(y_te, pr)
    ax.plot(fpr, tpr, lw=2, color=COL[name], label=f"{name} (AUC = {test_rows[name]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)"); ax.set_title("ROC curves on the test set")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("m4_roc.png", dpi=170); plt.close()

# Fig 5: metric comparison bars
metrics = ["accuracy", "precision", "recall", "f1", "auc"]
fig, ax = plt.subplots(figsize=(8.5, 4))
w = 0.2
for i, (name, row) in enumerate(test_rows.items()):
    ax.bar(np.arange(len(metrics)) + i * w, [row[m] for m in metrics], w, color=COL[name], label=name)
ax.set_xticks(np.arange(len(metrics)) + 1.5 * w); ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "ROC AUC"])
ax.set_ylim(0.8, 1.02); ax.set_title("Test-set performance of the four models"); ax.legend(ncol=2, fontsize=9, loc="lower left")
plt.tight_layout(); plt.savefig("m5_metrics.png", dpi=170); plt.close()

# ------------------------------------------------------------------
# 7. OVERFITTING / UNDERFITTING: decision-tree depth curve and learning curve of final model
# ------------------------------------------------------------------
depths = list(range(1, 13))
tr_acc, cv_acc = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=SEED)
    tr_acc.append(t.fit(X_tr, y_tr).score(X_tr, y_tr))
    cv_acc.append(cross_validate(t, X_tr, y_tr, cv=cv)["test_score"].mean())
R["depth_curve"] = dict(depths=depths, train=[round(a, 4) for a in tr_acc], cv=[round(a, 4) for a in cv_acc])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(depths, tr_acc, "o-", color="#E8590C", label="Training accuracy")
ax.plot(depths, cv_acc, "s-", color="#2F6FDE", label="Cross-validated accuracy")
ax.set_xlabel("Decision tree max_depth"); ax.set_ylabel("Accuracy"); ax.set_title("Underfitting vs overfitting (decision tree)")
ax.legend(); plt.tight_layout(); plt.savefig("m6_depth.png", dpi=170); plt.close()

final = tuned[best_name]
sizes, ltr, lcv = learning_curve(final, X_tr, y_tr, cv=cv, train_sizes=np.linspace(0.1, 1.0, 8), scoring="accuracy", random_state=SEED)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(sizes, ltr.mean(1), "o-", color="#E8590C", label="Training accuracy")
ax.plot(sizes, lcv.mean(1), "s-", color="#2F6FDE", label="Cross-validated accuracy")
ax.fill_between(sizes, lcv.mean(1) - lcv.std(1), lcv.mean(1) + lcv.std(1), color="#2F6FDE", alpha=0.15)
ax.set_xlabel("Training set size"); ax.set_ylabel("Accuracy"); ax.set_title(f"Learning curve: {best_name}")
ax.legend(); plt.tight_layout(); plt.savefig("m7_learning.png", dpi=170); plt.close()
R["learning_curve_last"] = dict(train=float(ltr.mean(1)[-1]), cv=float(lcv.mean(1)[-1]))

# ------------------------------------------------------------------
# 8. INTERPRETATION: feature importance of final model, error analysis, threshold analysis
# ------------------------------------------------------------------
rf = tuned["Random Forest"].named_steps["clf"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
R["rf_top"] = imp.round(3).to_dict()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(imp.index[::-1], imp.values[::-1], color="#2E9E5B")
ax.set_xlabel("Importance"); ax.set_title("Top 10 features (Random Forest)")
plt.tight_layout(); plt.savefig("m8_importance.png", dpi=170); plt.close()

pr_final = probs[best_name]
pred_final = (pr_final >= 0.5).astype(int)
err = X_te.copy(); err["true"] = y_te.values; err["prob_malignant"] = pr_final.round(3); err["pred"] = pred_final
err = err[err.true != err.pred][["mean radius", "mean concavity", "true", "prob_malignant", "pred"]]
R["errors"] = err.round(3).reset_index(drop=True).to_dict("records")

prec, rec, thr = precision_recall_curve(y_te, pr_final)
th_rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    p_ = (pr_final >= t).astype(int)
    th_rows.append(dict(threshold=t, precision=precision_score(y_te, p_), recall=recall_score(y_te, p_),
                        fp=int(((p_ == 1) & (y_te.values == 0)).sum()), fn=int(((p_ == 0) & (y_te.values == 1)).sum())))
R["threshold"] = th_rows
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(rec, prec, color="#E8590C", lw=2)
for r_ in th_rows:
    ax.scatter(r_["recall"], r_["precision"], color="#1F2A37", zorder=5)
    ax.annotate(f"t={r_['threshold']}", (r_["recall"], r_["precision"]), textcoords="offset points", xytext=(-8, -14), fontsize=8)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Precision–recall curve: {best_name}")
plt.tight_layout(); plt.savefig("m9_pr.png", dpi=170); plt.close()

# Random-seed stability of the final model: 10 different splits
accs = []
for s in range(10):
    a, b, c_, d_ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=s)
    accs.append(final.fit(a, c_).score(b, d_))
R["stability"] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), min=float(np.min(accs)), max=float(np.max(accs)))
final.fit(X_tr, y_tr)

if best_name == "Logistic Regression":
    coefs = pd.Series(final.named_steps["clf"].coef_[0], index=X.columns).sort_values(key=abs, ascending=False).head(6)
    R["lr_top_coefs"] = coefs.round(3).to_dict()

def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open("ml_results.json", "w"), indent=1, default=str)
print(json.dumps(clean(R), indent=1, default=str))

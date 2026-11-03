# -*- coding: utf-8 -*-
"""
INT234 Predictive Analytics - CA-2
Project : Student Academic Performance Prediction Using Machine Learning
Author  : <Kashi vishweshwar>   |   Reg. No: <12316160>

HOW TO RUN (Spyder)
-------------------
1. Put this file and student-mat.csv in the SAME folder.
2. Open this file in Spyder and press F5 (Run file).
3. All graphs and result files are saved in a folder called  outputs/
   (created next to this script).

Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
    pip install pandas numpy matplotlib seaborn scikit-learn
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
PASS_MARK = 10          # PASS if G3 >= 10, else FAIL
SHOW_PLOTS = False      # set True to also display plots inside Spyder

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:       # when code is run cell-by-cell in Spyder
    BASE_DIR = os.getcwd()

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {"FAIL": "#C0392B", "PASS": "#1E8449"}


def save_fig(name):
    """Save the current figure into outputs/ and close it."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# 1. LOAD DATA  (separator is ';')
# ----------------------------------------------------------------------------
candidates = [os.path.join(BASE_DIR, "student-mat.csv"),
              os.path.join(os.getcwd(), "student-mat.csv"),
              os.path.join(BASE_DIR, "data", "student-mat.csv")]
csv_path = next((p for p in candidates if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("student-mat.csv not found. Place it in the same "
                            "folder as this script and run again.")

df = pd.read_csv(csv_path, sep=";")
N_ORIGINAL_COLS = df.shape[1]   # 33 original columns (before we add the PASS/FAIL column)

print("=" * 70)
print("1. DATASET INSPECTION")
print("=" * 70)
print("File loaded      :", csv_path)
print("Shape            :", df.shape, "(rows, columns)")
print("\nColumns          :\n", list(df.columns))
print("\nData types:\n", df.dtypes.value_counts())
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values   :", int(df.isnull().sum().sum()))
print("Duplicate rows   :", int(df.duplicated().sum()))

expected_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu',
                 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime',
                 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
                 'activities', 'nursery', 'higher', 'internet', 'romantic',
                 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health',
                 'absences', 'G1', 'G2', 'G3']
assert list(df.columns) == expected_cols, "Columns differ from student.txt!"
print("\nColumn check     : OK - all 33 columns match student.txt")

# Make sure grades are numeric (some CSV copies quote G1/G2 as text)
for g in ["G1", "G2", "G3"]:
    df[g] = pd.to_numeric(df[g])

desc = df.describe().T
desc.to_csv(os.path.join(OUT_DIR, "descriptive_statistics.csv"))

# ----------------------------------------------------------------------------
# 2. CREATE TARGET + REMOVE G3 (avoid data leakage)
# ----------------------------------------------------------------------------
df["result"] = np.where(df["G3"] >= PASS_MARK, "PASS", "FAIL")
y = (df["G3"] >= PASS_MARK).astype(int)         # 1 = PASS, 0 = FAIL

# G3 defines the target, so keeping it as an input would let the model simply
# read the answer (data leakage). It is dropped from the features.
X = df.drop(columns=["G3", "result"])

print("\n" + "=" * 70)
print("2. TARGET CREATION")
print("=" * 70)
print("Rule: PASS = G3 >= 10 ; FAIL = G3 < 10")
print(df["result"].value_counts())
print((df["result"].value_counts(normalize=True) * 100).round(2).astype(str) + " %")
print("\nG3 removed from features -> 'G3' in X.columns ?", "G3" in X.columns)
print("Number of input features:", X.shape[1])

# ----------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. EXPLORATORY DATA ANALYSIS  (graphs saved in outputs/)")
print("=" * 70)

# 3.1 Target distribution
fig, ax = plt.subplots(figsize=(5.5, 4))
counts = df["result"].value_counts().reindex(["FAIL", "PASS"])
bars = ax.bar(counts.index, counts.values,
              color=[PALETTE[c] for c in counts.index])
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 3,
            f"{v} ({v / len(df) * 100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Target Distribution (PASS vs FAIL)")
ax.set_ylabel("Number of students")
ax.set_ylim(0, counts.max() * 1.15)
save_fig("eda_01_target_distribution.png")

# 3.2 G3 histogram with pass mark
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df["G3"], bins=21, kde=False, color="#2E86C1", ax=ax)
ax.axvline(PASS_MARK, color="red", linestyle="--", label=f"Pass mark = {PASS_MARK}")
ax.set_title("Distribution of Final Grade (G3)")
ax.set_xlabel("G3 (0-20)")
ax.legend()
save_fig("eda_02_g3_distribution.png")

# 3.3 Correlation heatmap (numeric columns incl. G3 for EDA only)
num_cols = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(12, 9))
corr = df[num_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 6}, linewidths=0.3)
plt.title("Correlation Heatmap of Numeric Variables")
save_fig("eda_03_correlation_heatmap.png")

corr_g3 = corr["G3"].drop("G3").sort_values(key=np.abs, ascending=False)
corr_g3.to_csv(os.path.join(OUT_DIR, "correlation_with_G3.csv"),
               header=["correlation_with_G3"])
print("Top correlations with G3:\n", corr_g3.head(8).round(3))

# 3.4 G1 / G2 vs G3
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, col in zip(axes, ["G1", "G2"]):
    sns.scatterplot(data=df, x=col, y="G3", hue="result", palette=PALETTE,
                    alpha=0.7, ax=ax)
    ax.set_title(f"{col} vs Final Grade G3")
save_fig("eda_04_g1_g2_vs_g3.png")

# 3.5 Pass rate by key numeric/ordinal features
pass_rate_cols = ["studytime", "failures", "Medu", "goout"]
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.ravel(), pass_rate_cols):
    rate = df.groupby(col)["G3"].apply(lambda s: (s >= PASS_MARK).mean() * 100)
    sns.barplot(x=rate.index.astype(str), y=rate.values, color="#2E86C1", ax=ax)
    ax.set_title(f"Pass rate (%) by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 105)
save_fig("eda_05_pass_rate_by_feature.png")

# 3.6 Pass rate by categorical features
cat_plot_cols = ["higher", "school", "sex", "internet", "schoolsup", "address"]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for ax, col in zip(axes.ravel(), cat_plot_cols):
    rate = df.groupby(col)["G3"].apply(lambda s: (s >= PASS_MARK).mean() * 100)
    sns.barplot(x=rate.index, y=rate.values, color="#117A65", ax=ax)
    ax.set_title(f"Pass rate (%) by {col}")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 105)
save_fig("eda_06_pass_rate_categorical.png")

# 3.7 Absences & age boxplots
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="result", y="absences", order=["FAIL", "PASS"],
            palette=PALETTE, ax=axes[0])
axes[0].set_title("Absences by Result")
sns.boxplot(data=df, x="result", y="age", order=["FAIL", "PASS"],
            palette=PALETTE, ax=axes[1])
axes[1].set_title("Age by Result")
save_fig("eda_07_absences_age_boxplot.png")

# Group means (used in report)
group_means = df.groupby("result")[["G1", "G2", "G3", "absences", "failures",
                                    "studytime", "age"]].mean().round(2)
group_means.to_csv(os.path.join(OUT_DIR, "group_means_by_result.csv"))
print("\nMean values by result:\n", group_means)

# ----------------------------------------------------------------------------
# 4. PREPROCESSING + STRATIFIED TRAIN-TEST SPLIT
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

print("\n" + "=" * 70)
print("4. STRATIFIED TRAIN-TEST SPLIT (80/20)")
print("=" * 70)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])
print("Train PASS %:", round(y_train.mean() * 100, 2),
      "| Test PASS %:", round(y_test.mean() * 100, 2))

cat_features = X.select_dtypes(include="object").columns.tolist()
num_features = X.select_dtypes(exclude="object").columns.tolist()
print("Categorical features:", len(cat_features), "| Numeric features:", len(num_features))


def make_preprocessor(cat_cols, num_cols):
    """Scale numeric columns, one-hot encode categorical columns.
    The preprocessor is fitted ONLY on training data inside each pipeline."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


def make_models():
    return {
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    }


# ----------------------------------------------------------------------------
# 5. TRAIN, TEST AND EVALUATE ALL FIVE MODELS
# ----------------------------------------------------------------------------
def run_experiment(X_tr, X_te, y_tr, y_te, X_all, y_all, tag):
    """Train all 5 models, return (results_df, fitted_pipes, cms, reports, roc)."""
    cats = X_tr.select_dtypes(include="object").columns.tolist()
    nums = X_tr.select_dtypes(exclude="object").columns.tolist()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, pipes, cms, reports, rocs = [], {}, {}, {}, {}
    for name, model in make_models().items():
        pipe = Pipeline([("prep", make_preprocessor(cats, nums)), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]

        cv_f1 = cross_val_score(pipe, X_all, y_all, cv=skf, scoring="f1")
        cv_acc = cross_val_score(pipe, X_all, y_all, cv=skf, scoring="accuracy")

        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Precision": precision_score(y_te, pred),
            "Recall": recall_score(y_te, pred),
            "F1-Score": f1_score(y_te, pred),
            "ROC-AUC": roc_auc_score(y_te, proba),
            "Train Accuracy": accuracy_score(y_tr, pipe.predict(X_tr)),
            "CV Accuracy (mean)": cv_acc.mean(),
            "CV Accuracy (std)": cv_acc.std(),
            "CV F1 (mean)": cv_f1.mean(),
        })
        pipes[name] = pipe
        cms[name] = confusion_matrix(y_te, pred)
        reports[name] = classification_report(
            y_te, pred, target_names=["FAIL", "PASS"], digits=3)
        fpr, tpr, _ = roc_curve(y_te, proba)
        rocs[name] = (fpr, tpr, roc_auc_score(y_te, proba))
    res = pd.DataFrame(rows)
    return res, pipes, cms, reports, rocs


print("\n" + "=" * 70)
print("5. MODEL TRAINING AND EVALUATION (positive class = PASS)")
print("=" * 70)
results, pipes, cms, reports, rocs = run_experiment(
    X_train, X_test, y_train, y_test, X, y, "main")

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(results.round(4).to_string(index=False))

results.round(4).to_csv(os.path.join(OUT_DIR, "model_comparison_results.csv"), index=False)

with open(os.path.join(OUT_DIR, "classification_reports.txt"), "w") as f:
    for name in reports:
        f.write(f"===== {name} =====\n")
        f.write(f"Confusion matrix [rows=actual FAIL,PASS ; cols=predicted FAIL,PASS]:\n"
                f"{cms[name]}\n\n")
        f.write(reports[name] + "\n\n")
for name in reports:
    print(f"\n--- {name} ---")
    print("Confusion matrix (rows=actual FAIL/PASS, cols=predicted FAIL/PASS):")
    print(cms[name])
    print(reports[name])

# ----------------------------------------------------------------------------
# 6. RESULT GRAPHS
# ----------------------------------------------------------------------------
model_names = results["Model"].tolist()
short = {"Gaussian Naive Bayes": "GNB", "Logistic Regression": "LogReg",
         "Decision Tree": "DTree", "Random Forest": "RForest",
         "K-Nearest Neighbors": "KNN"}

# 6.1 Individual confusion matrices
for name in model_names:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    ConfusionMatrixDisplay(cms[name], display_labels=["FAIL", "PASS"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {name}")
    ax.grid(False)
    save_fig("cm_" + short[name].lower() + ".png")

# 6.2 All confusion matrices in one figure
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, name in zip(axes.ravel(), model_names):
    ConfusionMatrixDisplay(cms[name], display_labels=["FAIL", "PASS"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(name)
    ax.grid(False)
axes.ravel()[-1].axis("off")
plt.suptitle("Confusion Matrices of All Five Models (Test Set)", fontsize=14)
save_fig("cm_all_models.png")

# 6.3 Metric comparison bar chart
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
plot_df = results.melt(id_vars="Model", value_vars=metrics,
                       var_name="Metric", value_name="Score")
plt.figure(figsize=(11, 5))
ax = sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", palette="viridis")
ax.set_ylim(0, 1.25)
ax.set_title("Model Comparison - Accuracy, Precision, Recall, F1-Score")
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=7, padding=2)
plt.xticks(rotation=12)
ax.legend(loc="upper center", ncol=4)
save_fig("model_comparison_bar.png")

# 6.4 ROC curves
plt.figure(figsize=(6.5, 5.5))
for name in model_names:
    fpr, tpr, auc = rocs[name]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Test Set")
plt.legend(loc="lower right", fontsize=8)
save_fig("roc_curves.png")

# 6.5 Cross-validation accuracy
plt.figure(figsize=(8, 4.5))
ax = sns.barplot(x=results["Model"], y=results["CV Accuracy (mean)"],
                 yerr=results["CV Accuracy (std)"], color="#5DADE2")
ax.set_ylim(0, 1.1)
ax.set_title("5-Fold Stratified Cross-Validation Accuracy (mean +/- std)")
for i, v in enumerate(results["CV Accuracy (mean)"]):
    ax.text(i, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
plt.xticks(rotation=12)
save_fig("cross_validation_accuracy.png")

# 6.6 Decision-tree and Random-Forest feature importance
prep_fitted = pipes["Random Forest"].named_steps["prep"]
feature_names = [n.replace("num__", "").replace("cat__", "")
                 for n in prep_fitted.get_feature_names_out()]

rf_imp = pd.Series(pipes["Random Forest"].named_steps["model"].feature_importances_,
                   index=feature_names).sort_values(ascending=False)
rf_imp.to_csv(os.path.join(OUT_DIR, "rf_feature_importance.csv"), header=["importance"])
plt.figure(figsize=(8, 5.5))
sns.barplot(x=rf_imp.head(12).values, y=rf_imp.head(12).index, color="#1E8449")
plt.title("Random Forest - Top 12 Feature Importances")
plt.xlabel("Importance")
save_fig("rf_feature_importance.png")

dt_imp = pd.Series(pipes["Decision Tree"].named_steps["model"].feature_importances_,
                   index=feature_names).sort_values(ascending=False)
dt_imp.to_csv(os.path.join(OUT_DIR, "dt_feature_importance.csv"), header=["importance"])

# 6.7 Logistic regression coefficients
lr_coef = pd.Series(pipes["Logistic Regression"].named_steps["model"].coef_[0],
                    index=feature_names).sort_values(key=np.abs, ascending=False)
lr_coef.to_csv(os.path.join(OUT_DIR, "logreg_coefficients.csv"), header=["coefficient"])

print("\nTop 8 Random Forest features:\n", rf_imp.head(8).round(4))

# 6.8 Decision tree picture
from sklearn.tree import plot_tree
plt.figure(figsize=(16, 7))
plot_tree(pipes["Decision Tree"].named_steps["model"], feature_names=feature_names,
          class_names=["FAIL", "PASS"], filled=True, rounded=True, max_depth=3,
          fontsize=8)
plt.title("Decision Tree (first 3 levels)")
save_fig("decision_tree_plot.png")

# ----------------------------------------------------------------------------
# 7. EXTRA EXPERIMENT: WITHOUT G1 AND G2 (harder, more realistic early-warning case)
# ----------------------------------------------------------------------------
print("\n" + "=" * 70)
print("7. EXTRA EXPERIMENT - features WITHOUT G1 and G2")
print("=" * 70)
X2 = X.drop(columns=["G1", "G2"])
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
results_nog, _, cms_nog, _, _ = run_experiment(
    X2_train, X2_test, y2_train, y2_test, X2, y, "no_g1_g2")
print(results_nog.round(4).to_string(index=False))
results_nog.round(4).to_csv(os.path.join(OUT_DIR, "model_comparison_without_G1_G2.csv"),
                            index=False)

cmp_df = pd.DataFrame({
    "Model": results["Model"],
    "With G1 & G2": results["F1-Score"],
    "Without G1 & G2": results_nog["F1-Score"]}).melt(
        id_vars="Model", var_name="Feature set", value_name="F1-Score")
plt.figure(figsize=(10, 4.8))
ax = sns.barplot(data=cmp_df, x="Model", y="F1-Score", hue="Feature set",
                 palette=["#1E8449", "#E67E22"])
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
ax.set_ylim(0, 1.12)
ax.set_title("F1-Score: With vs Without Previous Grades (G1, G2)")
plt.xticks(rotation=12)
save_fig("with_vs_without_g1g2.png")

# ----------------------------------------------------------------------------
# 8. SUMMARY + SAVE EVERYTHING FOR THE REPORT / PPT GENERATORS
# ----------------------------------------------------------------------------
best = results.sort_values(["F1-Score", "Accuracy"], ascending=False).iloc[0]
print("\n" + "=" * 70)
print("8. SUMMARY")
print("=" * 70)
print(f"Best model on test set (by F1-score): {best['Model']}")
print(f"  Accuracy={best['Accuracy']:.4f}  Precision={best['Precision']:.4f}  "
      f"Recall={best['Recall']:.4f}  F1={best['F1-Score']:.4f}")

summary = {
    "dataset": {
        "rows": int(df.shape[0]), "columns": int(N_ORIGINAL_COLS),
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "n_pass": int((y == 1).sum()), "n_fail": int((y == 0).sum()),
        "n_features": int(X.shape[1]),
        "n_categorical": len(cat_features), "n_numeric": len(num_features),
        "n_features_after_encoding": len(feature_names),
        "g3_mean": float(df["G3"].mean()), "g3_median": float(df["G3"].median()),
        "g3_std": float(df["G3"].std()),
        "age_min": int(df["age"].min()), "age_max": int(df["age"].max()),
    },
    "split": {"train": int(len(X_train)), "test": int(len(X_test)),
              "train_pass": int(y_train.sum()), "test_pass": int(y_test.sum()),
              "test_fail": int(len(y_test) - y_test.sum())},
    "results": results.round(4).to_dict(orient="records"),
    "results_without_g1g2": results_nog.round(4).to_dict(orient="records"),
    "confusion_matrices": {k: v.tolist() for k, v in cms.items()},
    "confusion_matrices_without_g1g2": {k: v.tolist() for k, v in cms_nog.items()},
    "best_model": best["Model"],
    "top_rf_features": rf_imp.head(10).round(4).to_dict(),
    "top_dt_features": dt_imp.head(5).round(4).to_dict(),
    "top_corr_g3": corr_g3.head(8).round(4).to_dict(),
    "group_means": group_means.to_dict(orient="index"),
    "logreg_top_coef": lr_coef.head(8).round(4).to_dict(),
}
with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll graphs and result files were saved to:", OUT_DIR)
print("Done.")

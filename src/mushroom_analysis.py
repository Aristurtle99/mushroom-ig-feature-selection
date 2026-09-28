"""
Does Information-Gain Feature Selection Preserve Classifier Accuracy?
A Comparative Study on Mushroom Edibility Classification

This script implements the full methodology from the project proposal:
  1. Load and clean the UCI Mushroom dataset
  2. One-hot encode the categorical features
  3. Split into train/test sets
  4. Compute the entropy of the target and the information gain (IG) of
     every original feature, then rank features by IG
  5. Train logistic regression + decision tree on the FULL feature set
  6. Train the same two models on only the TOP-K features (by IG)
  7. Compare accuracy / precision / recall / confusion matrices
  8. Cross-check: does the decision tree's own first splits agree with
     the IG ranking computed independently in step 4?

Run with:  python mushroom_analysis.py
Requires:  pandas, numpy, scikit-learn, matplotlib  (see requirements.txt)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Patch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ----------------------------------------------------------------------
# CONFIG — change these if you want to experiment
# ----------------------------------------------------------------------
DATA_PATH = "data/raw/agaricus-lepiota.data"  # the raw UCI file, no header row
TEST_SIZE = 0.25                     # 25% held out for testing (within the 20-30% range in the proposal)
RANDOM_STATE = 42                    # fixes the random shuffling so results are reproducible
TOP_K = 5                            # how many top information-gain features to keep for the "reduced" model
MAX_TREE_DEPTH = 5                   # cap tree depth so it doesn't just memorize the training data (overfitting)
RESULTS_DIR = Path("results")

# The dataset file has no header row, so we supply the column names ourselves,
# taken directly from agaricus-lepiota.names (attribute list, in order).
# "class" is the target (e = edible, p = poisonous); the other 22 are features.
COLUMN_NAMES = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat",
]


# ----------------------------------------------------------------------
# STEP 1: LOAD AND CLEAN THE DATA
# ----------------------------------------------------------------------
def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Loads the raw CSV (no header) and assigns readable column names.

    Cleaning notes:
    - Every value in this dataset is a single-letter categorical code
      (e.g. 'x' = convex cap-shape). There are no numeric columns.
    - The only missing values are in 'stalk-root', encoded as '?'
      (2480 of the 8124 rows). We treat '?' as its own category
      ("missing is missing-ness is informative here") rather than
      dropping those rows, since dropping ~30% of the data would be
      wasteful and rows aren't missing at random (it's tied to species).
    """
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)

    missing_count = (df["stalk-root"] == "?").sum()
    print(f"Loaded {len(df)} rows, {df.shape[1] - 1} features.")
    print(f"'?' (missing) values in stalk-root: {missing_count} "
          f"({missing_count / len(df):.1%}) — kept as their own category.\n")

    return df


# ----------------------------------------------------------------------
# STEP 2: ENTROPY AND INFORMATION GAIN
# ----------------------------------------------------------------------
def entropy(labels: pd.Series) -> float:
    """
    Shannon entropy of a categorical variable, in bits:
        H = -sum_i p_i * log2(p_i)

    For our binary target this is 0 when a set is pure (all edible or
    all poisonous) and 1 (its maximum) when it's a perfect 50/50 split.
    """
    counts = labels.value_counts(normalize=True)  # normalize=True -> proportions p_i, not raw counts
    return -np.sum(counts * np.log2(counts))


def information_gain(df: pd.DataFrame, feature: str, target: str) -> float:
    """
    Information gain of `feature` with respect to `target`:
        IG(feature) = H(target) - H(target | feature)

    H(target | feature) is the *weighted average* entropy of the target
    after splitting the data into groups by each value the feature can
    take (e.g. odor = almond / anise / creosote / ...): each group's
    entropy is weighted by what fraction of the rows fall into it.

    This is exactly the quantity ID3/C4.5-style decision trees compute
    internally to decide which feature to split on at each node, and it
    is mathematically identical to the mutual information between the
    feature and the target in the discrete case.
    """
    total_entropy = entropy(df[target])

    weighted_conditional_entropy = 0.0
    for value, group in df.groupby(feature, observed=True):
        weight = len(group) / len(df)
        weighted_conditional_entropy += weight * entropy(group[target])

    return total_entropy - weighted_conditional_entropy


def rank_features_by_information_gain(df: pd.DataFrame, target: str) -> pd.Series:
    """Computes IG for every feature column and returns them sorted, highest first."""
    feature_cols = [c for c in df.columns if c != target]
    ig_scores = {f: information_gain(df, f, target) for f in feature_cols}
    return pd.Series(ig_scores).sort_values(ascending=False)


# ----------------------------------------------------------------------
# STEP 3: ONE-HOT ENCODING (needed because logistic regression and
# sklearn's DecisionTreeClassifier both require numeric input, but
# every column here is a category like 'odor' with no natural order)
# ----------------------------------------------------------------------
def one_hot_encode(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Turns each categorical column into a set of binary (0/1) columns,
    one per category — e.g. 'odor' becomes 'odor_almond', 'odor_foul',
    etc. This avoids implying a false numeric ordering between
    categories (which a plain 1,2,3... encoding would do).
    """
    return pd.get_dummies(df[feature_cols], columns=feature_cols)


# ----------------------------------------------------------------------
# STEP 4: TRAIN + EVALUATE ONE MODEL
# ----------------------------------------------------------------------
def train_and_evaluate(model, X_train, X_test, y_train, y_test, label: str) -> dict:
    """
    Fits `model` and returns its accuracy, precision, recall and
    confusion matrix on the held-out test set. Precision/recall are
    computed treating 'p' (poisonous) as the positive class, since a
    false negative (calling a poisonous mushroom edible) is the
    dangerous kind of mistake.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results = {
        "label": label,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="p"),
        "recall": recall_score(y_test, y_pred, pos_label="p"),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=["e", "p"]),
        "report": classification_report(y_test, y_pred, target_names=["edible", "poisonous"]),
        "model": model,
    }

    print(f"--- {label} ---")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision (poisonous): {results['precision']:.4f}")
    print(f"Recall (poisonous):    {results['recall']:.4f}")
    print("Confusion matrix [rows=actual, cols=predicted], order = [edible, poisonous]:")
    print(results["confusion_matrix"])
    print()

    return results


# ----------------------------------------------------------------------
# STEP 5: PLOTS (saved as PNG files so they can be dropped into the report)
# ----------------------------------------------------------------------
def plot_information_gain(ig_ranked: pd.Series, top_k: int, out_path: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#d95f02" if i < top_k else "#7570b3" for i in range(len(ig_ranked))]
    ax.barh(ig_ranked.index[::-1], ig_ranked.values[::-1], color=colors[::-1])
    ax.set_xlabel("Information gain (bits)")
    ax.set_title(f"Feature ranking by information gain (top {top_k} highlighted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved information-gain chart -> {out_path}")


def plot_confusion_matrices(results_list: list[dict], out_path: str):
    fig, axes = plt.subplots(1, len(results_list), figsize=(5 * len(results_list), 4))
    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm = res["confusion_matrix"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(res["label"], fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["edible", "poisonous"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["edible", "poisonous"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion-matrix comparison -> {out_path}")


def plot_decision_tree(tree_model, feature_names, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        tree_model, feature_names=feature_names, class_names=["edible", "poisonous"],
        filled=True, rounded=True, fontsize=8, max_depth=3, ax=ax,  # max_depth=3 here just limits what's DRAWN, not the trained tree
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved decision tree diagram -> {out_path}")


def plot_logistic_regression_coefficients(model, feature_names, out_path: str,
                                           title: str, top_n: int = 20):
    """
    Bar chart of the top_n one-hot columns with the largest-magnitude
    logistic regression coefficients (the weights the model actually learned).

    scikit-learn sorts the two class labels alphabetically for a binary
    target, so classes_ = ['e', 'p'] here — the coefficients describe the
    push toward 'p' (poisonous). A positive coefficient increases the
    predicted log-odds of poisonous; a negative one increases the odds of
    edible. This is the "sign and magnitude" interpretation from the
    background research section, made visual.
    """
    coefs = pd.Series(model.coef_[0], index=feature_names)
    top_coefs = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:top_n])
    top_coefs = top_coefs.sort_values()  # ascending, so the largest bars land at the top of the chart

    colors = ["#d62728" if c > 0 else "#1f77b4" for c in top_coefs.values]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_coefs.index, top_coefs.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-odds weight)")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(facecolor="#d62728", label="Pushes toward poisonous"),
        Patch(facecolor="#1f77b4", label="Pushes toward edible"),
    ], loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved logistic regression coefficient chart -> {out_path}")


def plot_sigmoid_function(out_path: str, title: str = "Logistic Regression"):
    """
    Draws the sigmoid (logistic) function itself:
        sigmoid(X) = 1 / (1 + e^-X)

    This is NOT computed from the mushroom data — it's the general-purpose
    curve that logistic regression squashes its raw linear output (the
    "log-odds", called X here) through, to turn it into a 0-1 probability.
    Useful as a conceptual figure for the background-research section,
    separate from the data-driven plots.

    Note: your actual model's decision-function values on the mushroom
    test set range from about -11.6 to +11.8, and cluster heavily near
    those two extremes rather than spreading evenly through the middle
    (because the classes are almost perfectly separable here — see the
    quasi-separation caveat in the coefficient-plot discussion). So a
    version of this plot made from your real predictions would look like
    two dense clumps near y=0 and y=1 with a sparse middle, not this even
    curve. This function draws the clean textbook version on purpose.
    """
    X = np.linspace(-5, 6, 100)
    y = 1 / (1 + np.exp(-X))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color="orange", edgecolor="blue", linewidth=0.8, s=40, zorder=3)
    ax.set_xlabel("X")
    ax.set_ylabel("sigmoid(X)")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved sigmoid function illustration -> {out_path}")


def plot_sigmoid_real_data(model, X_test, y_test, out_path: str,
                            title: str = "Logistic Regression — actual test-set predictions"):
    """
    The same sigmoid curve as plot_sigmoid_function(), but plotted from the
    ACTUAL trained model and the real held-out test set, instead of a
    made-up linspace.

    X here is model.decision_function(X_test) — the raw log-odds score the
    model computes for each mushroom (a weighted sum of its one-hot
    feature values, before squashing). Y is model.predict_proba(X_test),
    the probability of "poisonous" after the sigmoid is applied. Every
    point falls exactly on the sigmoid curve by definition (predict_proba
    IS sigmoid(decision_function) for logistic regression) — the only
    difference from the idealized plot is WHERE along the curve the real
    points land.

    Points are colored by their true label, so you can see how cleanly
    the two classes separate along the curve. The dashed lines mark the
    decision boundary: X=0 (log-odds of exactly 50/50) and Y=0.5.
    """
    decision_scores = model.decision_function(X_test)
    probabilities = model.predict_proba(X_test)[:, list(model.classes_).index("p")]

    colors = np.where(y_test.values == "p", "#d62728", "#1f77b4")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(decision_scores, probabilities, c=colors, edgecolor="black",
               linewidth=0.3, s=35, alpha=0.7, zorder=3)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.set_xlabel("Decision function score (log-odds of poisonous)")
    ax.set_ylabel("Predicted probability of poisonous")
    ax.set_title(title)
    ax.legend(handles=[
        Patch(facecolor="#d62728", label="Actually poisonous"),
        Patch(facecolor="#1f77b4", label="Actually edible"),
    ], loc="center right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved real-data sigmoid plot -> {out_path}")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    target = "class"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: load & clean -------------------------------------------------
    df = load_and_clean_data(DATA_PATH)

    # --- Step 2: train/test split (done on the ORIGINAL categorical columns
    #     first, so information gain and one-hot encoding are both computed
    #     only from the training set — this avoids "leaking" test-set
    #     information into feature selection) --------------------------------
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[target]
    )
    print(f"Train rows: {len(train_df)}   Test rows: {len(test_df)}\n")

    # --- Step 3: entropy of the target + information gain per feature --------
    target_entropy = entropy(train_df[target])
    print(f"Entropy of target on training set: {target_entropy:.4f} bits "
          f"(1.0 = perfectly balanced classes)\n")

    ig_ranked = rank_features_by_information_gain(train_df, target)
    print("Features ranked by information gain (highest first):")
    print(ig_ranked.to_string(float_format=lambda x: f"{x:.4f}"))
    print()

    top_k_features = list(ig_ranked.index[:TOP_K])
    print(f"Top-{TOP_K} features selected for the reduced model: {top_k_features}\n")

    plot_information_gain(ig_ranked, TOP_K, str(RESULTS_DIR / "information_gain_ranking.png"))
    plot_sigmoid_function(str(RESULTS_DIR / "sigmoid_function.png"))

    # --- Step 4: one-hot encode, full feature set -----------------------------
    all_features = [c for c in df.columns if c != target]
    X_train_full = one_hot_encode(train_df, all_features)
    X_test_full = one_hot_encode(test_df, all_features)
    # Test set might not contain every category the train set has (or vice
    # versa) purely by chance in the split; reindex so both have identical
    # columns, filling anything missing with 0.
    X_test_full = X_test_full.reindex(columns=X_train_full.columns, fill_value=0)

    y_train = train_df[target]
    y_test = test_df[target]

    # --- Step 4: one-hot encode, reduced (top-k) feature set ------------------
    X_train_topk = one_hot_encode(train_df, top_k_features)
    X_test_topk = one_hot_encode(test_df, top_k_features)
    X_test_topk = X_test_topk.reindex(columns=X_train_topk.columns, fill_value=0)

    print(f"Full one-hot feature matrix:   {X_train_full.shape[1]} columns")
    print(f"Reduced (top-{TOP_K}) one-hot feature matrix: {X_train_topk.shape[1]} columns\n")

    # --- Step 5: train + evaluate all four models ------------------------------
    results = []

    results.append(train_and_evaluate(
        LogisticRegression(max_iter=1000),
        X_train_full, X_test_full, y_train, y_test,
        "Logistic Regression — full features",
    ))
    results.append(train_and_evaluate(
        LogisticRegression(max_iter=1000),
        X_train_topk, X_test_topk, y_train, y_test,
        f"Logistic Regression — top-{TOP_K} features",
    ))

    dt_full = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH, random_state=RANDOM_STATE)
    results.append(train_and_evaluate(
        dt_full, X_train_full, X_test_full, y_train, y_test,
        "Decision Tree — full features",
    ))

    dt_topk = DecisionTreeClassifier(max_depth=MAX_TREE_DEPTH, random_state=RANDOM_STATE)
    results.append(train_and_evaluate(
        dt_topk, X_train_topk, X_test_topk, y_train, y_test,
        f"Decision Tree — top-{TOP_K} features",
    ))

    plot_confusion_matrices(results, str(RESULTS_DIR / "confusion_matrices.png"))
    plot_decision_tree(dt_full, X_train_full.columns, str(RESULTS_DIR / "decision_tree_full.png"),
                        "Decision tree trained on the full feature set (top 3 levels shown)")
    plot_logistic_regression_coefficients(
        results[0]["model"], X_train_full.columns, str(RESULTS_DIR / "logistic_regression_coefficients.png"),
        "Logistic regression coefficients — full feature model (top 20 by magnitude)",
    )
    plot_sigmoid_real_data(results[0]["model"], X_test_full, y_test, str(RESULTS_DIR / "sigmoid_function_real_data.png"))

    # --- Step 6: summary table ---------------------------------------------------
    summary = pd.DataFrame([
        {"Model": r["label"], "Accuracy": r["accuracy"],
         "Precision": r["precision"], "Recall": r["recall"]}
        for r in results
    ])
    print("=" * 70)
    print("SUMMARY: full-feature vs. top-k accuracy/precision/recall")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    # --- Step 7: cross-check — do the tree's own splits agree with IG? --------
    # feature_importances_ on a one-hot-encoded tree gives one score per
    # ONE-HOT COLUMN (e.g. 'odor_foul'), so we group those back up to the
    # original feature (e.g. 'odor') by summing, to compare fairly against
    # the information-gain ranking (which was computed per original feature).
    importances = pd.Series(dt_full.feature_importances_, index=X_train_full.columns)
    importances_by_original_feature = (
        importances.groupby(importances.index.str.rsplit("_", n=1).str[0]).sum()
        .sort_values(ascending=False)
    )

    print("=" * 70)
    print("CROSS-CHECK: information-gain ranking vs. what the full-feature")
    print("decision tree actually relies on (its feature_importances_,")
    print("summed back up per original feature)")
    print("=" * 70)
    comparison = pd.DataFrame({
        "IG rank": ig_ranked.rank(ascending=False).astype(int),
        "Information gain": ig_ranked,
        "Tree importance rank": importances_by_original_feature.rank(ascending=False).astype(int),
        "Tree importance": importances_by_original_feature,
    }).sort_values("IG rank")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

    print(f"\nThe tree's actual root-node split was on: "
          f"'{X_train_full.columns[dt_full.tree_.feature[0]].rsplit('_', 1)[0]}' "
          f"— compare this to the #1 information-gain feature above.")

    summary.to_csv(RESULTS_DIR / "results_summary.csv", index=False)
    comparison.to_csv(RESULTS_DIR / "ig_vs_tree_importance.csv")
    print("\nSaved results_summary.csv and ig_vs_tree_importance.csv")


if __name__ == "__main__":
    main()

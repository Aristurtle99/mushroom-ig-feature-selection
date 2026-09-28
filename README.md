# Does Information-Gain Feature Selection Preserve Classifier Accuracy?

### A Comparative Study on Mushroom Edibility Classification

A reproducible machine-learning mini-project investigating whether **information gain** can be used to reduce a categorical dataset from 22 original features to its top 5 features without materially reducing classifier performance.

The project uses the **UCI Mushroom dataset** (8,124 observations, 22 categorical features) and compares two classifiers:

- Logistic Regression
- Decision Tree

Each classifier is trained twice:

1. using all 22 original features
2. using only the top 5 features ranked by information gain

The project also compares information-gain rankings with the decision tree's learned feature importances and first split. The methodology uses a stratified 75/25 train/test split with `random_state=42`; feature selection is performed only on the training split to avoid test-set leakage. fileciteturn0file0L68-L97

## Key results

| Model | Features | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | Full (22) | 99.95% | 100.00% | 99.90% |
| Logistic Regression | Top 5 | 99.85% | 100.00% | 99.69% |
| Decision Tree | Full (22) | 99.90% | 100.00% | 99.80% |
| Decision Tree | Top 5 | 99.85% | 100.00% | 99.69% |

These are the held-out test-set results reported in the project report (`n = 2,031`). fileciteturn0file0L134-L143

The top information-gain features were:

1. `odor` — 0.9028 bits
2. `spore-print-color` — 0.4849 bits
3. `gill-color` — 0.4150 bits
4. `ring-type` — 0.3169 bits
5. `stalk-surface-above-ring` — 0.2885 bits

fileciteturn0file0L98-L112

The decision tree independently selected `odor` as its root split, matching the highest-information-gain feature. fileciteturn0file0L153-L165

## Repository structure

```text
mushroom-ig-feature-selection/
├── README.md
├── requirements.txt
├── src/
│   └── mushroom_analysis.py
├── data/
│   └── raw/
│       └── agaricus-lepiota.data
├── results/
│   ├── confusion_matrices.png
│   ├── decision_tree_full.png
│   ├── information_gain_ranking.png
│   ├── logistic_regression_coefficients.png
│   ├── sigmoid_function.png
│   ├── sigmoid_function_real_data.png
│   ├── results_summary.csv
│   └── ig_vs_tree_importance.csv
└── docs/
    └── Mushroom_Project_Report.pdf
```

## Methodology

### 1. Data preparation

The raw UCI file contains a target class (`e` = edible, `p` = poisonous) and 22 categorical features. The only missing values occur in `stalk-root`, where `?` appears in 2,480 of 8,124 observations. These values are retained as their own category rather than dropping the affected rows. fileciteturn0file0L68-L78

### 2. Information gain

For a categorical target, entropy is calculated as

\[
H(Y) = -\sum_i p_i\log_2 p_i
\]

and information gain is

\[
IG(Y,X) = H(Y) - H(Y\mid X).
\]

The 22 original features are ranked using information gain calculated on the training data. The top 5 form the reduced feature set. fileciteturn0file0L35-L42

### 3. Encoding

All categorical predictors are one-hot encoded. This avoids imposing a false numerical ordering on categories. The full feature representation contains 117 one-hot columns, while the top-5 representation contains 39. fileciteturn0file0L82-L93

### 4. Models

Two classifiers are evaluated:

- `LogisticRegression(max_iter=1000)`
- `DecisionTreeClassifier(max_depth=5, random_state=42)`

Precision and recall use **poisonous** as the positive class because incorrectly calling a poisonous mushroom edible is the critical error considered in the project. fileciteturn0file0L87-L93

### 5. Cross-check

The decision tree's one-hot feature importances are grouped back into the original categorical features and compared against the information-gain ranking. The analysis finds that both approaches put `odor` first, while lower-ranked features can diverge because tree importance depends on which features have already been used in earlier splits. fileciteturn0file0L196-L214

## Reproducing the results

### Requirements

Python 3.9+ is recommended. Install the required packages:

```bash
pip install -r requirements.txt
```

The project uses pandas, NumPy, scikit-learn, and Matplotlib. fileciteturn0file3L1-L5

### Run

From the repository root:

```bash
python src/mushroom_analysis.py
```

The script reads `data/raw/agaricus-lepiota.data` and writes all generated plots and CSV summaries into `results/`.

The pipeline implemented by the script is: data loading/cleaning → train/test split → entropy and information gain → one-hot encoding → full and top-k models → evaluation → visualizations → information-gain/tree cross-check. fileciteturn0file1L5-L17

## Configurable parameters

The main configuration is near the top of `src/mushroom_analysis.py`:

```python
TEST_SIZE = 0.25
RANDOM_STATE = 42
TOP_K = 5
MAX_TREE_DEPTH = 5
```

These can be changed to experiment with different train/test splits, numbers of selected features, or tree depths. fileciteturn0file1L38-L45

## Results and visualizations

The `results/` directory contains the figures used in the project report:

- **Information-gain ranking** — all 22 features ranked by information gain.
- **Confusion matrices** — the four model/feature-set combinations.
- **Decision tree** — the top three displayed levels of the full-feature tree.
- **Logistic-regression coefficients** — the 20 largest-magnitude one-hot coefficients.
- **Sigmoid function** — the conceptual logistic function.
- **Real test-set sigmoid output** — actual decision scores and predicted poisonous probabilities.
- **CSV summaries** — model metrics and information-gain vs. tree-importance comparison.

These outputs correspond to the figures and tables documented in the report. fileciteturn0file2L25-L34

## Interpretation

The central result is that reducing the model from 22 original features to the top 5 by information gain changed accuracy by at most 0.10 percentage point in this experiment. The report therefore treats information gain as an effective and inexpensive feature-selection method **for this dataset**, while explicitly noting that the conclusion should not automatically be generalized to other datasets. fileciteturn0file0L153-L165 fileciteturn0file0L231-L251

A particularly important methodological caveat is that scikit-learn's decision tree uses Gini impurity by default, whereas the information-gain discussion is based on entropy. The two criteria agree on `odor` here, but they are not mathematically identical. fileciteturn0file0L206-L214

## Project report

The complete course mini-project report is included in [`docs/Mushroom_Project_Report.pdf`](docs/Mushroom_Project_Report.pdf). The report identifies the project as a BITS Pilani course mini-project by Mandaar Kumar Prabhudeva Rangapura and Muhammad Wildan Tadas. fileciteturn0file0L2-L9

## Dataset

The dataset is the UCI Mushroom dataset. The original project report lists the **UCI Machine Learning Repository — Mushroom Data Set** as its first reference. fileciteturn0file0L289-L292

## Limitations

The project report identifies several limitations:

- the mushroom dataset is almost perfectly separable using a small number of categorical features;
- the tree uses Gini impurity rather than entropy;
- `TOP_K = 5` was selected manually rather than by cross-validation or a formal cutoff;
- results are based on a single stratified train/test split;
- `stalk-root` missing values are treated as a separate category; and
- the conclusion is specific to this dataset and should not be assumed to generalize to feature-selection problems in general. fileciteturn0file0L231-L251

## Authors

**Muhammad Wildan Tadas**  
**Mandaar Kumar Prabhudeva Rangapura**

BITS Pilani — September 2026

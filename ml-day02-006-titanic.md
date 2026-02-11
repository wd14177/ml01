## Plan: Titanic Survival Model Pipeline

TL;DR: Build a **logistic regression** baseline with a clean train/test split and leakage-safe preprocessing, extend features with a few common Titanic signals (title, family size, deck/has-cabin), then tune via cross-validated grid search optimizing accuracy. Evaluate on the held-out test split and report metrics. All work will be contained in the Titanic notebook, following patterns used in the other training notebooks.

**Steps**
1. Expand EDA in [ml-day2-006-titinic.ipynb](ml-day2-006-titinic.ipynb) to check missingness, target balance, and basic feature distributions for key columns in [data/titanic.csv](data/titanic.csv).
2. Refactor preprocessing in [ml-day2-006-titinic.ipynb](ml-day2-006-titinic.ipynb) to add common features: `Title` from `Name`, `FamilySize = SibSp + Parch + 1`, `IsAlone`, and a `HasCabin` or `CabinDeck` indicator; keep `PassengerId` as index, keep `Survived` as target, and drop unused columns after feature extraction.
3. Create an 80/20 split with `random_state=3333` and `stratify=y` in [ml-day2-006-titinic.ipynb](ml-day2-006-titinic.ipynb); fit imputers/encoders on train only and apply to test to avoid leakage, following patterns from [wt_day1-data_preprocessing.ipynb](wt_day1-data_preprocessing.ipynb).
4. Train a logistic regression baseline (with scaling if needed) in [ml-day2-006-titinic.ipynb](ml-day2-006-titinic.ipynb) and evaluate accuracy and a classification report on the held-out test set, similar to [ml-003.ipynb](ml-003.ipynb).
5. Run `GridSearchCV` for logistic regression hyperparameters (e.g., `C`, penalty, solver) using cross-validation and accuracy scoring, following the tuning style in [ml-day2-005-apply.ipynb](ml-day2-005-apply.ipynb), then evaluate the best model on the test split.
6. Summarize results and keep the best model for future prediction use in [ml-day2-006-titinic.ipynb](ml-day2-006-titinic.ipynb).

**Verification**
- Manual checks in the notebook: confirm split sizes, no target leakage, and that accuracy is reported for both baseline and tuned models.
- Optional: compare tuned vs baseline accuracy to ensure tuning provides a tangible improvement.

**Decisions**
- Model: Logistic regression baseline.
- Metric: Accuracy for tuning and evaluation.
- Output: Evaluation only (no submission CSV).
- Preprocessing: Add a few standard Titanic features beyond the current ones.
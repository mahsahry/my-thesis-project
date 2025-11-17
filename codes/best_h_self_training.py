"""
Select h for Semi-Supervised Self-Training (h-sweep)
----------------------------------------------------

This script is ONLY for choosing a good value of h for a self-training
(Semi-Supervised) procedure.

Given a fully labeled binary dataset, it:

1. Simulates a semi-supervised setting by:
   - keeping a small fraction of the training data as labeled,
   - treating the rest as unlabeled.

2. For each candidate h in a grid (e.g., 0.05, 0.10, ..., 0.50):
   - runs a self-training loop that repeatedly:
       * trains a classifier on the current labeled set,
       * selects the top h fraction of unlabeled samples with highest
         predicted probability for the positive class,
       * pseudo-labels them and adds them to the labeled set,
       * removes them from the unlabeled pool,
     until the unlabeled pool is empty,
   - evaluates the final model on a held-out test set,
   - records Recall, G-Mean, and AUC.

3. Averages metrics over multiple random splits and prints/saves one row
   per h. You can then choose the h that maximizes the metric you care
   about (e.g. G-Mean or AUC) and plug that h into your main SSSTR code.

To adapt this script:
  - set FILE_PATH to your CSV file,
  - set TARGET_COLUMN to your label column,
  - set FEATURE_COLUMNS to the numeric feature columns you want to use.
"""


import numpy as np
import pandas as pd
import warnings
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import ClusterCentroids

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIG – ADAPT THESE TO YOUR DATASET
# ---------------------------------------------------------

FILE_PATH = "labeled_data.csv"   # your labeled dataset
TARGET_COLUMN = "label"          # name of the target column (0/1)
FEATURE_COLUMNS = [              # numeric features to use
    "feature1",
    "feature2",
    "feature3",
]

# optional: metadata / ID columns to drop if present
DROP_COLUMNS = []                # e.g. ["id", "timestamp"]

TEST_SIZE = 0.2                  # 20% held-out test
LABELED_FRACTION = 0.1           # 10% of train used as initial labeled set

H_VALUES = np.arange(0.05, 0.55, 0.05)  # candidate h values
N_REPEATS = 50                             # repetitions per h

# classifier used for h selection (must support predict_proba)
BASE_MODEL = DecisionTreeClassifier(random_state=42)


# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Dataset file not found: {FILE_PATH}")

df = pd.read_csv(FILE_PATH)

# Drop any metadata columns if specified
for col in DROP_COLUMNS:
    if col in df.columns:
        df.drop(col, axis=1, inplace=True)

# Basic validation
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in dataset.")

for f in FEATURE_COLUMNS:
    if f not in df.columns:
        raise ValueError(f"Feature '{f}' not found in dataset.")

X_raw = df[FEATURE_COLUMNS]
y = df[TARGET_COLUMN]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

# Impute missing
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X_scaled)


# ---------------------------------------------------------
# h-sweep: choose best h for self-training
# ---------------------------------------------------------

results = []

for h in H_VALUES:
    recalls, gmeans, aucs = [], [], []

    print(f"\nEvaluating h = {h:.2f}")

    for rep in range(N_REPEATS):
        # 1) Split into train/test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=rep
        )

        # 2) From train_full: create small labeled subset and big unlabeled pool
        X_labeled, X_unlabeled, y_labeled, _ = train_test_split(
            X_train_full,
            y_train_full,
            test_size=1.0 - LABELED_FRACTION,
            stratify=y_train_full,
            random_state=rep,
        )

        # 3) Balance labeled subset with under-sampling (optional but useful)
        under_sampler = ClusterCentroids(random_state=rep)
        X_labeled_bal, y_labeled_bal = under_sampler.fit_resample(X_labeled, y_labeled)

        # Initialize labeled and unlabeled pools for self-training
        L_X = X_labeled_bal.copy()
        L_y = y_labeled_bal.to_numpy()
        U = X_unlabeled.copy()

        # 4) Self-training loop for this h
        model = BASE_MODEL

        while len(U) > 0:
            model.fit(L_X, L_y)

            # predicted probability of positive class (assumed label=1)
            proba = model.predict_proba(U)[:, 1]

            # number of unlabeled samples to add in this iteration
            k = max(1, int(h * len(U)))

            # take top-k most confident samples
            top_idx = np.argsort(proba)[-k:]
            X_new = U[top_idx]
            y_new = model.predict(X_new)  # pseudo-labels

            # expand labeled set
            L_X = np.vstack((L_X, X_new))
            L_y = np.concatenate((L_y, y_new))

            # remove them from the unlabeled pool
            U = np.delete(U, top_idx, axis=0)

        # 5) Final evaluation on test set
        model.fit(L_X, L_y)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Recall
        recall = recall_score(y_test, y_pred)

        # Specificity and G-Mean
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = float(np.sqrt(recall * specificity))

        # AUC
        auc = roc_auc_score(y_test, y_proba)

        recalls.append(recall)
        gmeans.append(gmean)
        aucs.append(auc)

    # Average metrics for this h
    results.append(
        {
            "h": h,
            "Recall": np.mean(recalls),
            "G-Mean": np.mean(gmeans),
            "AUC": np.mean(aucs),
        }
    )

# ---------------------------------------------------------
# Output: table + plot to choose h
# ---------------------------------------------------------

results_df = pd.DataFrame(results)
print("\nAverage metrics per h:\n", results_df)

results_df.to_csv("h_sweep_results.csv", index=False)

# Optional: visualization to help pick h
plt.figure(figsize=(8, 5))
plt.plot(results_df["h"], results_df["G-Mean"], marker="o", label="G-Mean")
plt.plot(results_df["h"], results_df["Recall"], marker="s", label="Recall")
plt.plot(results_df["h"], results_df["AUC"], marker="^", label="AUC")
plt.xlabel("h (fraction of unlabeled samples added per iteration)")
plt.ylabel("Average metric value")
plt.title("h-sweep: Recall / G-Mean / AUC vs h")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

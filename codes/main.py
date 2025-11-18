"""
SSSTR: Top-h Self-Training with Unlabeled Data
----------------------------------------------

SSSTR (Self-training with Selective Sample Transfer based on Ranking)
is a semi-supervised learning strategy that exploits an unlabeled pool
to improve a classifier trained on a small labeled set.

Notation
--------
- L : initial labeled set
- U : unlabeled pool
- T : held-out test set
- C : classifier (with predict_proba)
- h : proportion (0 < h <= 1) of the most confident unlabeled samples
      to transfer from U to L at each iteration

Basic Workflow
--------------
1. Split the labeled dataset into:
     - L-Train  (for training)
     - L-Test   (test set T, kept fixed for evaluation)

2. (Optional) Apply a sampling method on L-Train to handle class
   imbalance, producing a balanced training set L-New.

3. Initialize:
     - L_time = L-New
     - U_time = U

4. Repeat until U_time is empty or no new samples are added:
     a. Train classifier C on L_time.
     b. Use C to predict class probabilities on U_time.
     c. Rank U_time by the probability of the positive class (e.g., fake = 1).
     d. Select the top h proportion of the most confident samples.
     e. Add these samples to L_time as pseudo-labeled positives and
        remove them from U_time.

5. After self-training converges, retrain C on the final L_time and
   evaluate on L-Test (T) using:
     - Recall
     - AUC
     - G-Mean = sqrt(recall * specificity)

This module provides a reusable function `ssstr_self_train` that
implements Step 4 (the core SSSTR loop). It can be plugged into any
experiment script where:
  - you already prepared L-Train, U, and L-Test,
  - and you want to augment L-Train with pseudo-labeled data from U.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix


def ssstr_self_train(
    base_estimator: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_unlabeled: np.ndarray,
    h: float,
    max_iterations: int = 10,
    positive_class: int = 1,
    return_history: bool = False,
):

    if not hasattr(base_estimator, "predict_proba"):
        raise ValueError("SSSTR requires an estimator with predict_proba().")

    if not (0 < h <= 1):
        raise ValueError("h must be in (0, 1].")

    # Clone estimator to avoid modifying external instance
    model = clone(base_estimator)

    # Copy arrays so the original data is not modified
    X_L = np.array(X_train, copy=True)
    y_L = np.array(y_train, copy=True)
    U = np.array(X_unlabeled, copy=True)

    history = {"n_L": [], "n_added": []} if return_history else None

    for _ in range(max_iterations):
        if U.shape[0] == 0:
            break

        # Train on current labeled set
        model.fit(X_L, y_L)

        # Predict probabilities on U
        proba = model.predict_proba(U)
        conf = proba[:, positive_class]

        # Number of samples to transfer this iteration
        top_h = int(h * len(U))
        if top_h <= 0:
            break

        # Indices of the top-h most confident positives
        top_indices = np.argsort(conf)[-top_h:]
        X_new = U[top_indices]
        y_new = np.full(len(top_indices), positive_class, dtype=y_L.dtype)

        # If nothing new is added, stop
        if X_new.shape[0] == 0:
            break

        # Expand labeled set
        X_L = np.vstack((X_L, X_new))
        y_L = np.concatenate((y_L, y_new))

        # Remove from unlabeled pool
        U = np.delete(U, top_indices, axis=0)

        if history is not None:
            history["n_L"].append(len(y_L))
            history["n_added"].append(len(y_new))

    # Final fit on expanded labeled set
    model.fit(X_L, y_L)

    if return_history:
        return model, X_L, y_L, history
    else:
        return model, X_L, y_L, None


def evaluate_ssstr_model(
    estimator: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    positive_class: int = 1,
) -> Dict[str, float]:

    y_pred = estimator.predict(X_test)

    # Recall
    recall = recall_score(y_test, y_pred)

    # Specificity for G-Mean
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = float(np.sqrt(recall * specificity))

    # AUC
    proba = estimator.predict_proba(X_test)[:, positive_class]
    auc = roc_auc_score(y_test, proba)

    return {
        "recall": recall,
        "auc": auc,
        "gmean": gmean,
    }



if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier

    FILE_PATH = "your_dataset.csv"
    TARGET_COLUMN = "label"
    FEATURE_COLUMNS = [...]

    df = pd.read_csv(FILE_PATH)
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_L, X_U, y_L, _ = train_test_split(
        X_train_full, y_train_full, test_size=0.90, random_state=42
    )

    H_LIST = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    base_classifier = DecisionTreeClassifier(random_state=42)

    results = []

    for h in H_LIST:
        model, X_final, y_final, _ = ssstr_self_train(
            base_estimator=base_classifier,
            X_train=X_L,
            y_train=y_L,
            X_unlabeled=X_U,
            h=h,
            max_iterations=10,
            return_history=False,
        )

        metrics = evaluate_ssstr_model(model, X_test, y_test)
        metrics["h"] = h
        results.append(metrics)

    for r in results:
        print(f"{r['h']},{r['recall']:.4f},{r['gmean']:.4f},{r['auc']:.4f}")

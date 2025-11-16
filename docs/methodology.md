
# Algorithms

This document provides the key algorithms used in the TikTok fake-account detection
pipeline. These descriptions are implementation-agnostic and suitable for
reproducibility without revealing code or dataset details.

---

## 1. Recursive Feature Elimination with SVM (RFE-SVM)

**Input:**  
- Training data `X`  
- Labels `y`  
- Target number of features `k`

**Output:**  
- Selected subset `S` of `k` features

**Procedure:**

1. Initialize the feature set `F` with all feature indices in `X`.
2. While the size of `F` is greater than `k`:
   1. Train a linear SVM classifier on `X[:, F]` with labels `y`.
   2. Obtain the weight vector `w` from the trained SVM.
   3. Rank features in `F` by the absolute values of their weights `|wᵢ|`.
   4. Remove from `F` the feature with the smallest `|wᵢ|`.
3. Set `S = F`.
4. Return `S` as the selected feature subset.

RFE-SVM provides a simple yet effective way to identify compact feature subsets by
iteratively pruning the least informative features.

---

## 2. Self-Training Semi-Supervised Learning (SSSTR)

**Input:**  
- Labeled data `L = {(xᵢ, yᵢ)}`  
- Unlabeled data `U = {xⱼ}`  
- Base classifier `C`  
- Confidence threshold `τ` (optional)

**Output:**  
- Expanded labeled dataset

**Procedure:**

1. Initialize the labeled set `L` and unlabeled pool `U`.
2. Train the classifier `C` on `L`.
3. Predict pseudo-labels for all instances in `U`.
4. Select the instances whose predicted probabilities exceed the threshold `τ`
   (e.g., highest-confidence predictions).
5. Move these selected instances from `U` to `L`, using the pseudo-labels.
6. Optionally apply a resampling method (e.g., SMOTE or CBUTE) to rebalance `L`.
7. Repeat Steps 2–6 until `U` becomes empty or no new confident predictions remain.
8. Return the expanded labeled set.

This process leverages the large pool of unlabeled TikTok accounts to gradually
augment the labeled dataset and improve classifier performance under limited supervision.

---

## Notes

- Both algorithms are provided in pseudocode to preserve anonymity and avoid binding
  the project to a particular programming language.
- These algorithms are referenced in the main `README.md` and used throughout the
  experimental pipeline.

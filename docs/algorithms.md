# Algorithms

## Algorithm 1 — Recursive Feature Elimination with SVM (RFE-SVM)

```text
Input:
    X — training data
    y — labels
    k — target number of features

Output:
    S — selected feature subset

Procedure:
    1. Initialize F = all feature indices in X
    2. While |F| > k:
           a. Train a linear SVM classifier on X[:, F] with labels y
           b. Extract the weight vector w from the trained SVM
           c. Rank features in F by ascending |w_i|
           d. Remove the feature with the smallest |w_i|
    3. Set S = F
    4. Return S

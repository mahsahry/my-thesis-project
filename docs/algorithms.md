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
```
## Algorithm 2 — Self-Training Semi-Supervised Learning with Resampling (SSSTR)

```text
Input:
    L  — labeled dataset
    U  — unlabeled dataset
    C  — classifier
    h  — number of high-confidence samples to add per iteration

Output:
    Predicted labels for L_Test

Procedure:

1. Normalize features in both L and U using Min–Max and Z-Score scaling.

2. Split labeled set L into:
       L_Train  — training portion
       L_Test   — test portion

3. Let:
       L_Train = {l1, l2, ..., ln}
       L_maj   = majority-class samples
       L_min   = minority-class samples
       where |L_maj| > |L_min| and |L_maj| + |L_min| = n

4. Oversampling (minority class, SMOTE):
       • For each li in L_min:
             - Compute distances to other minority samples
             - Find K nearest neighbors
       • Set sampling ratio N based on class imbalance
       • Generate synthetic samples using Formula (1)
       • Let L_over be the oversampled set

5. Undersampling (majority class, CBUTE):
       • Randomly choose |L_min| majority samples as centroids μ = {μ1...μq}
       • Assign each sample in L_maj to nearest centroid
       • Recompute centroids until convergence
       • Let L_under be the resulting centroid set

6. Let L_new = (choose either L_over or L_under)

7. Initialize:
       time = 0
       L_time = L_new
       U_time = U

8. While U_time is not empty:
       a. Train classifier C on L_time
       b. Predict labels + confidences for all samples in U_time
       c. Sort predictions by descending confidence
       d. Select the top h samples → S_time
       e. Add S_time to L_time
       f. Remove S_time from U_time
       g. time = time + 1

9. Train classifier C on the final expanded labeled set L_time

10. Predict labels for L_Test using the trained classifier

11. Return predicted labels for L_Test
```

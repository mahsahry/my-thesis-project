<p align="center">
  <img src="figures/logo.png" alt="Detecting Fake Accounts on TikTok" width="200">
</p>


## Dataset

| Dataset | Accounts | Description |
|---|---|---|
| Labeled | 6,873 | 4,020 fake + 2,853 real TikTok accounts, sourced from [GitHub](https://github.com/raffaele-aurucci/Fake-TikTok-Account-Detection) and enriched via TikTok API |
| Unlabeled | 23,877 | Followers of the top 10 Dutch TikTok influencers, collected via [HypeAuditor](https://hypeauditor.com) |

The labeled dataset experienced **83% attrition** from the original 10,043-account source due to account deletions and privacy changes — a known challenge with social media datasets and a key motivation for using semi-supervised learning.

---

## Features

Nine TikTok-specific features are used, selected from profile metadata, engagement signals, and username consistency measures:

| Feature | Type | 3-feat | 5-feat | 7-feat | Best-5 |
|---|---|:---:|:---:|:---:|:---:|
| Likes Count | Numeric | | | | |
| Video Count | Numeric | ✓ | ✓ | ✓ | |
| Follower Count | Numeric | | | | |
| Following Count | Numeric | | ✓ | ✓ | ✓ |
| Verified | Binary | | | ✓ | |
| Bio | Binary | ✓ | ✓ | ✓ | ✓ |
| Has Contact Info | Binary | | | ✓ | ✓ |
| Jaro Similarity | Numeric (0–1) | | ✓ | ✓ | ✓ |
| Nickname Complexity | Numeric | ✓ | ✓ | ✓ | ✓ |

Feature subsets were identified using **RFE-SVM** (3, 5, 7 features) and **Bit-Flip local search** (Best-5).

---

## Method

The SSSTR pipeline for each experimental run:

```
Labeled set L (6,873)  +  Unlabeled set U (23,877)
         │
    Stratified 80/20 split
         │
   ┌─────┴──────┐
L_train        L_test
   │
Feature normalization (Min-Max or Z-score, fitted on L_train only)
   │
Resampling on L_train → L_new
   ├── NORS  (no resampling)
   ├── SMOTE (minority oversampling)
   └── CBUTE (majority cluster undersampling)
   │
Self-training loop:
   ├── Train classifier C on L_t
   ├── Predict confidence on U_t
   ├── Move top-h fraction to L_t with pseudo-labels
   └── Repeat until U_t is empty
   │
Evaluate final C on L_test → Recall, G-Mean, AUC
```

**Classifiers:** CART, Random Forest, Gradient Boosting, AdaBoost, KNN, Naïve Bayes

**Self-training parameter h** tuned over {0.1, 0.2, 0.3, 0.4, 0.5} per classifier, selecting the value maximizing (G-Mean + AUC) / 2.

**Evaluation:** 5-fold stratified cross-validation; mean and standard deviation reported.

---

## Results

### Peak Performance per Feature Subset

| Feature Subset | Normalization | Classifier | Resampling | Recall | G-Mean | AUC |
|---|---|---|---|---|---|---|
| 3-feat | Min-Max | GB | SMOTE | 83.12% | 89.50% | **95.04%** ± 2.61 |
| 5-feat | Min-Max | AB | CBUTE | 90.62% | 92.22% | **95.34%** ± 0.28 |
| 7-feat | Min-Max | AB | CBUTE | 90.00% | 91.03% | **94.42%** ± 0.66 |
| Best-5 | Min-Max | AB | CBUTE | 82.50% | 89.00% | **94.92%** ± 0.59 |

### Key Findings

- **5-feature subset** offers the best trade-off: peak AUC of 95.34% with low variance (±0.28)
- **3-feature subset** (Nickname Complexity, Video Count, Bio) is only 0.30pp behind at 95.04% — attractive for large-scale deployment due to lower feature collection cost
- **AdaBoost and Gradient Boosting** are the strongest classifiers (mean AUC 93.57% and 93.30%)
- **Naïve Bayes** degrades sharply beyond 5 features (AUC drops from 87.7% to ~57%), revealing sensitivity to redundant features
- **Min-Max vs Z-score** normalization: virtually identical overall (88.76% vs 88.82% mean AUC)
- **CBUTE** delivers the single best result; all three resampling strategies are competitive within a narrow band

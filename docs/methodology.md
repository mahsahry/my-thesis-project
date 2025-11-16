## Methodology

The detection pipeline consists of five main components: preprocessing, handling
class imbalance, feature selection, semi-supervised learning (SSSTR), and classifier
evaluation. Figures 1 and 2 (see below) provide a visual overview of the entire process.

---

### 1. Data Preprocessing
The dataset contains TikTok account metadata and engineered behavioral features.
Preprocessing included:
- cleaning text fields and converting boolean attributes to numeric form,
- normalizing all continuous features using **Min–Max** and **Z-Score** scaling,
- removing unreliable time-based features (Create Time Hour, Weekday, Nickname Time Gap)
  due to time-zone inconsistencies.

Two normalized versions of the dataset were created to examine model sensitivity
to scaling differences.

---

### 2. Handling Class Imbalance
Fake accounts form the minority class. Three strategies were explored:

- **No resampling** — baseline  
- **SMOTE oversampling** — generates synthetic minority instances  
- **CBUTE undersampling** — removes redundant majority samples using a clustering-based strategy  

Resampling was applied only to the **labeled portion** of the data during the
semi-supervised training phase.  
This corresponds to the *“Resample”* block in **Figure 1** and **Figure 2**.

---

### 3. Feature Selection
Two methods were used to obtain compact and informative feature subsets:

- **RFE-SVM (Recursive Feature Elimination)**  
  Iteratively removes the least informative features using linear SVM weights.

- **Bit-Flip Local Search**  
  Starts from an RFE subset and explores neighboring subsets by flipping one
  feature at a time, keeping any subset that improves the metric.

Subsets of 3, 5, 7, and a refined 4-feature combination were evaluated.

---

### 4. Semi-Supervised Self-Training (SSSTR)
The core of the pipeline is a semi-supervised self-training loop that augments
the limited labeled dataset using predictions on a larger unlabeled pool.

**Figure 1** illustrates the general SSSTR pipeline,  
while **Figure 2** shows the version used in this study with an initial
train/test split.

#### SSSTR workflow:
1. Split initial labeled data into **train** and **test** sets (Figure 2).
2. Treat 90% of the training data as **unlabeled (U)** and 10% as **labeled (L)**.
3. Apply SMOTE or CBUTE to rebalance L → produce **L-New**.
4. Train classifier **C** on L-New.
5. Predict labels for all instances in U.
6. Sort predictions by confidence.
7. Select the top high-confidence samples and move them from U to L-New.
8. Repeat until U is empty or no confident instances remain.

The final classifier is then trained on the expanded labeled set and evaluated on
the held-out test set.

---

### 5. Classifiers and Metrics

Six classifiers were evaluated:
- CART  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- K-Nearest Neighbors  
- Naïve Bayes  

The following metrics were used:
- **Recall (fake class)** — detects fake accounts  
- **G-Mean** — ensures balanced class performance  
- **AUC** — measures overall separability  

These metrics capture both minority-class sensitivity and cross-class stability.

---

## 6. SSSTR Pipeline Diagrams

The following diagrams illustrate the entire semi-supervised workflow:

### **Figure 1 – SSSTR Pipeline (Version 1)**
A high-level structure showing labeled/unlabeled split, resampling, and the
self-training loop.

![SSSTR Pipeline Version 1](figures/ssstr_pipeline_v1.png)

### **Figure 2 – SSSTR Pipeline (Version 2)**
The version used in this study, which includes an initial train/test split before
entering the self-training loop.

![SSSTR Pipeline Version 2](figures/ssstr_pipeline_v2.png)

These figures summarize how data flows through the system from preprocessing to
final evaluation.

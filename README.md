![ Detecting Fake Accounts on TikTok](figures/download.png)

This repository accompanies a study on semi-supervised fake-account detection in TikTok.
It documents the dataset schema, preprocessing steps, feature selection strategies,
class imbalance solutions, experimental settings, and summarized results.  
The repository is anonymized for double-blind review and does **not** contain the
original dataset or any identifying information.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset and Feature Schema](#dataset-and-feature-schema)  
3. [Methodology](#methodology)  
4. [Experimental Results (Overview)](#experimental-results-overview)  
5. [Repository Structure](#repository-structure)  
---

## Project Overview

This project investigates fake-account detection on TikTok under limited supervision.
A semi-supervised self-training framework (SSSTR) is used to leverage a small labeled
dataset together with a larger pool of unlabeled TikTok accounts.

### Key Contributions
- **Self-Training Semi-Supervised Learning (SSSTR)** for fake account detection.
- **Resampling techniques** to handle class imbalance:
  - SMOTE (oversampling)
  - CBUTE (cluster-based undersampling)
- **Feature Selection** using:
  - Recursive Feature Elimination with SVM (RFE-SVM)
  - Bit-Flip local search to refine compact feature subsets
- **Evaluation across six classifiers** using:
  - Recall  
  - G-Mean  
  - AUC  

### Data Sources
The study combines:
- A publicly available labeled dataset from GitHub  
- Additional metadata collected via the TikTok API and the Apify platform  

To maintain anonymity and privacy, this repository distributes **only synthetic sample
data and schema descriptions**, not the real dataset.

---

## Dataset and Feature Schema

The real dataset includes labeled TikTok accounts (fake/real) and metadata extracted
through the TikTok API. Typical features include:

- Username  
- Likes Count  
- Video Count  
- Follower / Following Count  
- Is Verified  
- Bio Length  
- Has Contact Info  
- Jaro Similarity  
- Nickname Complexity  
- Followers Ratio  
- Face Detected (binary)  
- Avatar Status  
- Time-based features (weekday, hour, time gap)

The repository provides:
- `data/synthetic_sample.csv` – a **fully artificial sample** that mirrors the real dataset structure.
- `data/README.md` – documentation of the schema and instructions for preparing your own dataset.

---

## Methodology

### 1. Preprocessing  
- Cleaning of text and boolean fields  
- Conversion to numeric format  
- Normalization using **Min–Max** and **Z-Score** scalers  

### 2. Handling Class Imbalance  
Three modes were tested:
- No resampling  
- SMOTE oversampling  
- CBUTE undersampling  

### 3. Feature Selection  
Two complementary strategies were applied:

1. **RFE-SVM (Recursive Feature Elimination with SVM)**  
   Iteratively removes the least informative feature using SVM weight magnitudes.

2. **Bit-Flip Local Search**  
   Starts from an RFE result and explores neighbor subsets to refine performance.

### 4. Semi-Supervised Self-Training (SSSTR)  
Process:
1. Start with few labeled data + many unlabeled accounts.  
2. Train a classifier on the labeled set.  
3. Predict unlabeled data.  
4. Add high-confidence predictions to the labeled set.  
5. Repeat until the unlabeled pool is empty.

### 5. Classifiers  
Six models were evaluated:
- CART  
- Random Forest  
- Gradient Boosting  
- AdaBoost  
- K-Nearest Neighbors  
- Naïve Bayes  

### 6. Metrics  
- **Recall** – sensitivity to fake accounts  
- **G-Mean** – class balance  
- **AUC** – ability to separate fake vs real  

---

## Experimental Results (Overview)

Experiments explored:
- 3-, 4-, 5-, and 7-feature subsets  
- Two normalization methods  
- Three resampling strategies  
- Six classifiers  

### Key Observations
- Very small feature sets (3 features) achieve **high recall** but poor G-Mean/AUC.  
- Medium-sized subsets (5–7 features) consistently improve balanced performance.  
- Z-Score + SMOTE often gives the most stable performance across classifiers.  
- A compact **4-feature subset** achieves strong results while remaining interpretable.  
- Semi-supervised self-training improves recall but may amplify class imbalance if the
  initial feature set is too weak.

These findings highlight the importance of carefully chosen feature subsets,
appropriate normalization, and resampling strategies in TikTok fake-account detection.

---

### 🔒 Why the dataset is not shared

The experiments in the associated work were conducted on **real-world social media data**, which may include sensitive user information.  
Due to **ethical, privacy, and platform-policy constraints**, the original dataset:

- **cannot be made publicly available**, and  
- **cannot be uploaded to this repository**.

Sharing it would violate privacy and responsible data-use guidelines.

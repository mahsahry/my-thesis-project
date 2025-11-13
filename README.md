## Detecting fake accounts on TikTok
Project Overview:
This repository provides the implementation and experiment pipelines for a study on semi-supervised fake-account detection in online social networks. It includes data preprocessing scripts, model training/evaluation code, and instructions to reproduce the reported results. The repository is anonymized for double-blind review.

Key Contributions:
- A Self-Training Semi-Supervised Learning Algorithm (SSSTR)
- Resampling techniques to address class imbalance: SMOTE and CBUTE
- Recursive Feature Elimination (RFE) with SVM and Bit-Flip local search for feature selection
- Evaluation using six classifiers and multiple metrics: Recall, G-Mean, and AUC

This research's dataset combines:
A publicly available labeled dataset from GitHub:
- [Fake TikTok Account Detection](https://github.com/raffaele-aurucci/Fake-TikTok-Account-Detection)
- Metadata collected through the TikTok API and the Apify platform

This work investigated model performance and the challenges of applying semi-supervised learning to TikTok fake account detection. 
The results show that factors such as the amount of labeled data, classifier sensitivity to settings, and the quality of feature selection significantly influence outcomes. 
In particular, models trained with limited labeled data or minimal feature sets often showed imbalanced performance, even when resampling techniques were applied. These insights are valuable for developing more reliable fake account detection systems and can guide future research toward improving model robustness under limited supervision.

### Detailed RFE and Bit-Flip Results

Below are the selected feature subsets obtained through RFE-SVM and the local
search optimization, along with their corresponding accuracies:

- **RFE with 3 Features**  
  - Accuracy: **70.59%**  
  - Selected Features: **Bio, Jaro Similarity, Nickname Complexity**

- **RFE with 5 Features**  
  - Accuracy: **79.02%**  
  - Selected Features: **Face Detected, Bio, Jaro Similarity, Nickname Complexity, Video Count**

- **RFE with 7 Features**  
  - Accuracy: **93.33%**  
  - Selected Features: **Face Detected, Follower Count, Bio, Jaro Similarity, Nickname Complexity, Create Time Weekday, Video Count**

- **Best Subset After Local Search (Bit-Flip)**  
  - Accuracy: **80.59%**  
  - Selected Features: **Face Detected, Bio, Jaro Similarity, Video Count**

These results show that feature engineering and selection play a critical role in
detection accuracy, with the 7-feature subset achieving the highest performance.

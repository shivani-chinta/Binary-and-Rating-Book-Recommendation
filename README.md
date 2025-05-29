# BookMatch: Binary and Rating-Based Book Recommendation System

This project builds a two-part book recommendation engine that predicts both **whether a user will read a book** and **what rating they might give**. It demonstrates the application of baseline models and collaborative filtering techniques using real-world interaction data.

---

## 📚 Dataset

The dataset consists of user-book interactions (from a Book-Crossing-like dataset), structured as:
- `train_Interactions.csv`: Training data with userID, bookID, and ratings
- `pairs_Read.csv`: Pairs for read/not-read prediction
- `pairs_Rating.csv`: Pairs for rating prediction

Each row includes a user-book interaction, where a rating of `0` means *not read* and `1` means *read*, or provides an explicit rating from 1–10.

---

## 🧠 Tasks and Methods

### Task 1: Read Prediction (Binary Classification)
- **Goal:** Predict whether a user will read a book.
- **Baseline:** Most popular books (threshold-based selection).
- **Model:**  
  - Bayesian Personalized Ranking (BPR) using `cornac`
  - Cosine similarity and Jaccard similarity for evaluation
- **Evaluation Metric:** Classification accuracy on held-out validation pairs.

### Task 2: Rating Prediction (Regression)
- **Goal:** Predict what rating a user would give to a book.
- **Model:** Matrix Factorization using `SVD` from `surprise`
- **Hyperparameter Tuning:**  
  - `n_factors`: [1, 5, 10]  
  - `reg_all`: [0.1, 0.2, 0.3]  
- **Evaluation Metric:** Mean Squared Error (MSE)

---

## 🛠 Technologies Used

- Python 3
- `pandas`, `numpy`, `scipy`
- `cornac` for BPR modeling
- `scikit-learn` and `surprise` for SVD and grid search
- `tqdm` for progress tracking

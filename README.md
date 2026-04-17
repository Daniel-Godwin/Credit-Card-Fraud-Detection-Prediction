
---

# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques on a highly imbalanced dataset. Fraud cases represent only a very small fraction of all transactions, making accurate detection challenging.

The goal is to build a robust classification system that maximizes fraud detection (recall) while minimizing false positives, with additional emphasis on model interpretability using SHAP.

---

## 🎯 Objectives

* Detect fraudulent transactions accurately in an imbalanced dataset
* Handle extreme class imbalance effectively
* Minimize false positives while maximizing recall
* Evaluate models using appropriate metrics for imbalance problems
* Provide interpretability using SHAP explanations

---

## 📊 Dataset

* Source: Kaggle Credit Card Fraud Dataset
* Features: Anonymized variables (`V1`–`V28`) due to confidentiality
* Additional fields:

  * `Time`
  * `Amount`
  * `Class` (Target)

    * `0` → Legitimate transaction
    * `1` → Fraudulent transaction
* Key Challenge: Severe class imbalance (fraud cases are extremely rare)

---

## 🧪 Methodology

### 1. Exploratory Data Analysis (EDA)

* Analyzed class imbalance distribution
* Studied feature correlations
* Identified patterns related to fraudulent behavior

---

### 2. Data Preprocessing

* Feature scaling using **StandardScaler / RobustScaler**
* Train-test split with stratification
* Separation of features and target variable
* Handling missing or inconsistent data (if any)

---

### 3. Handling Class Imbalance

To address dataset imbalance:

* **SMOTE (Synthetic Minority Oversampling Technique)**
* Optional comparison with:

  * Class weighting
  * No resampling baseline

---

### 4. Models Implemented

* Logistic Regression
* Random Forest
* XGBoost
* Isolation Forest (Anomaly Detection approach)

---

### 5. Model Evaluation

Models were evaluated using metrics suitable for imbalanced classification:

* Precision
* Recall (critical metric)
* F1-score
* ROC-AUC
* Precision-Recall AUC (PR-AUC)
* Confusion Matrix

---

### 6. Model Interpretability

* Applied **SHAP (SHapley Additive exPlanations)**
* Explained:

  * Global feature importance
  * Individual prediction reasoning

---

## 📈 Results Summary

* SMOTE significantly improved minority class learning
* Tree-based models performed best overall
* XGBoost and Random Forest showed strong balance between precision and recall
* Logistic Regression struggled under extreme imbalance without tuning
* Isolation Forest performed moderately as an unsupervised baseline

---

## 🏆 Model Performance Comparison

### 🔹 Ranked by F1-Score

| Rank | Model               | Technique         | Precision | Recall   | F1-score     | PR-AUC   |
| ---- | ------------------- | ----------------- | --------- | -------- | ------------ | -------- |
| 1    | Random Forest       | None              | 0.941860  | 0.826531 | **0.880435** | 0.873043 |
| 2    | XGBoost             | Scale_Pos_Weight  | 0.777778  | 0.857143 | **0.815534** | 0.868122 |
| 3    | Random Forest       | Class Weight      | 0.961039  | 0.755102 | **0.845714** | 0.858928 |
| 4    | Random Forest       | SMOTE             | 0.826531  | 0.826531 | **0.826531** | 0.875015 |
| 5    | Logistic Regression | None              | 0.826667  | 0.632653 | **0.717647** | 0.741382 |
| 6    | Logistic Regression | Class Weight      | 0.060976  | 0.918367 | **0.114358** | 0.718971 |
| 7    | Logistic Regression | SMOTE             | 0.057803  | 0.918367 | **0.108761** | 0.724469 |
| 8    | Isolation Forest    | Anomaly Detection | 0.311321  | 0.336735 | **0.323529** | NaN      |

---

### 🔹 Ranked by PR-AUC

| Rank | Model               | Technique         | PR-AUC       |
| ---- | ------------------- | ----------------- | ------------ |
| 1    | Random Forest       | SMOTE             | **0.875015** |
| 2    | Random Forest       | None              | **0.873043** |
| 3    | XGBoost             | Scale_Pos_Weight  | **0.868122** |
| 4    | Random Forest       | Class Weight      | **0.858928** |
| 5    | Logistic Regression | None              | 0.741382     |
| 6    | Logistic Regression | SMOTE             | 0.724469     |
| 7    | Logistic Regression | Class Weight      | 0.718971     |
| 8    | Isolation Forest    | Anomaly Detection | NaN          |

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE, ADASYN)
* XGBoost
* Matplotlib, Seaborn
* SHAP

---

## 📂 Project Structure

```
├── data/
│   └── creditcard.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── src/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
├── README.md
├── requirements.txt
```

---

## ⚙️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Daniel-Godwin/Credit-Card-Fraud-Detection-Prediction..git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

---

## 🚀 Future Improvements

* Hyperparameter tuning using GridSearch / Optuna
* Deploy model using Streamlit or FastAPI
* Real-time fraud detection pipeline
* Integration into MLOps workflow (MLflow / Docker)
* Deep learning-based fraud detection

---

## 👤 Author

**Daniel Godwin**
Artificial Intelligence Engineering
Istanbul Okan University
📧 [dgodwin@stu.okan.edu.tr](mailto:dgodwin@stu.okan.edu.tr)

---

## ⭐ Acknowledgements

* Kaggle dataset contributors
* Scikit-learn & XGBoost communities
* SHAP library developers
* Open-source ML ecosystem

---

## 📌 License

This project is intended for academic and research purposes only.

---

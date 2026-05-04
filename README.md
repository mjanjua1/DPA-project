# 🩺 Diabetes Risk Prediction — DPA Project

A machine learning project to predict diabetes risk (No Diabetes, Pre-Diabetes, or Diabetes) using the **BRFSS 2015 Health Indicators Dataset**, built as part of our Data & Predictive Analytics course (April 2025).

---

## Team

 Name 
| Swetha Reddy Chilakala 
| Sabarish Dhayalan 
| Muhammad Rehan Munir Janjua 
| Trisha Chanda 

> **Note on collaboration:** Each team member worked independently on their assigned phase in Google Colab. All notebooks were developed and tested separately, then committed to this repository together at the end of the project.

---

## Repository Structure

```
DPA-project/
├── Phase_2_Implementation_(DPA).ipynb   # EDA + Data Preprocessing
├── dpa_3.ipynb                          # Modeling, Evaluation & Feature Analysis
└── README.md
```

---

## Project Overview

**Research Question:**
> Can self-reported health survey data predict an individual's diabetes status with clinically meaningful accuracy?

We used the CDC's BRFSS 2015 survey data (253,680 records, 22 features) to build and evaluate three classifiers for multi-class diabetes prediction.

---

## Workflow

### Phase 1 — Data Cleaning
- Removed 23,908 duplicate rows (229,772 records retained)
- Capped BMI outliers above 70 (data entry errors)
- Converted 22 float64 columns → int64
- Engineered a `GenHlth × Income` interaction feature

### Phase 2 — Exploratory Data Analysis
- Identified severe class imbalance: 82.7% No Diabetes / 15.3% Diabetes / 2.0% Pre-Diabetes
- Found top correlating features: General Health, High BP, BMI, Difficulty Walking, High Cholesterol
- Observed clear BMI shift and age effect across classes

### Phase 3 — Modeling
- Applied **SMOTE** to training set only (no data leakage)
- Trained 3 models: Logistic Regression, Random Forest, XGBoost
- Used 80/20 train-test split, 5-fold cross-validation, GridSearchCV for tuning

### Phase 4 — Feature Importance Analysis
- Both RF and XGBoost independently ranked the same top 5 features:
  `GenHlth → BMI → HighBP → Age → HighChol`

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | — | 0.688 |
| Random Forest | — | 0.769 |
| **XGBoost** | **81.2%** | **0.801** |

> Naive baseline (always predicting majority class) = 82.7% — but with **zero predictive value** on minority classes. All our models meaningfully outperformed it on Pre-Diabetes and Diabetes detection.

---

## Tech Stack

- Python (Google Colab)
- pandas, numpy, scikit-learn
- XGBoost
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

---

## Limitations

- Pre-Diabetes recall remains ~30–50% due to structural class overlap
- Random Forest showed signs of overfitting (Train F1 ~1.0 vs Val F1 ~0.64)
- BRFSS is self-reported data — subject to response bias

---

##  Future Work

- Threshold optimization for minority class detection
- Cost-sensitive learning (penalizing missed diabetic cases more heavily)
- Stacking ensemble to push performance further

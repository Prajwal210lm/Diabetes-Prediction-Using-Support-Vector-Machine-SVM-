# Diabetes Prediction Using Support Vector Machine (SVM)

## Overview

This project builds an **end-to-end machine learning pipeline** to predict whether a person is diabetic based on key medical attributes. The solution leverages **data preprocessing, feature standardization, model training, evaluation, and inference**, implemented using Python and scikit-learn.

The objective is to demonstrate a **practical supervised learning workflow**, moving from raw data to a deployable predictive system.

---

## Problem Statement

Diabetes is a chronic health condition that requires early detection and intervention. Using historical diagnostic data, this project aims to classify individuals as **diabetic or non-diabetic** based on physiological and demographic features.

This is framed as a **binary classification problem**.

---

## Dataset

* Source: PIMA Indians Diabetes Dataset
* Number of records: **768**
* Number of features: **8**
* Target variable: `Outcome`

  * `0` → Non-Diabetic
  * `1` → Diabetic

### Features Used

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

---

## Tech Stack

* **Python**
* **NumPy & Pandas** for data manipulation
* **scikit-learn** for preprocessing, modeling, and evaluation
* **Google Colab / Jupyter Notebook** for development and experimentation

---

## Methodology

### 1. Data Loading and Exploration

* Loaded dataset using Pandas
* Performed initial inspection using `.head()`, `.shape()`, `.describe()`
* Analyzed class distribution and feature means by outcome

### 2. Data Preprocessing

* Separated features (`X`) and target (`Y`)
* Standardized features using `StandardScaler`
* Ensured consistent scaling across training, testing, and inference data

### 3. Train-Test Split

* Split data into **80% training** and **20% testing**
* Applied stratification to preserve class balance
* Used a fixed random state for reproducibility

### 4. Model Training

* Algorithm: **Support Vector Machine (SVM)**
* Kernel: **Linear**
* Trained on standardized feature space

### 5. Model Evaluation

* Training Accuracy: **~78.7%**
* Test Accuracy: **~77.3%**
* Performance indicates reasonable generalization without major overfitting

### 6. Predictive System

* Built a prediction pipeline that:

  * Accepts raw user input
  * Applies feature scaling
  * Outputs diabetic or non-diabetic classification

---

## Results

| Metric            | Score  |
| ----------------- | ------ |
| Training Accuracy | ~78.7% |
| Test Accuracy     | ~77.3% |

The model demonstrates **stable performance** on unseen data and serves as a solid baseline for further optimization.

---

## Key Learnings

* Importance of feature standardization in distance-based models like SVM
* Impact of data quality and preprocessing on model performance
* End-to-end ML pipeline design from data ingestion to inference
* Practical evaluation of classification models using train-test splits

---

## Future Improvements

* Hyperparameter tuning using GridSearchCV
* Experimentation with non-linear kernels
* Handling zero or missing values more rigorously
* Model comparison with Logistic Regression, Random Forest, or XGBoost
* Deployment as a web or API-based prediction service


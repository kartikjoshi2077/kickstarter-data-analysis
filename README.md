# Kickstarter Project Success Classification and Clustering

## Overview
This project focuses on predicting the success of Kickstarter projects based on historical data and uncovering patterns within the data through clustering. The classification model identifies projects as either "successful" or "failed," while the clustering analysis groups similar projects to understand their defining characteristics.

## Goals
- **Classification:** Use machine learning techniques to predict whether a Kickstarter project will be successful or not.
- **Clustering:** Group projects into clusters based on shared attributes to uncover patterns and insights.

---

## Project Workflow

### 1. Data Preprocessing
- Extracted **campaign_duration** and **preparation_duration** as derived features from the `deadline`, `launched_at`, and `created_at` timestamps.
- Dropped irrelevant or redundant columns, including IDs, timestamps, and other non-predictive variables.
- Applied a **log transformation** to the `goal` column to handle skewness.
- Converted categorical features such as `state`, `country`, and `category` into one-hot encoded dummy variables.
- Filtered the dataset to include only "successful" and "failed" projects for binary classification.

### 2. Outlier Detection
- Employed an **Isolation Forest** to identify and remove outliers in the dataset, retaining 95% of the data.
- Verified the effectiveness of outlier removal by checking dataset dimensions and feature distributions.

### 3. Handling Class Imbalance
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset by generating synthetic samples for the minority class (`failed` projects).

### 4. Model Training
- Built a **Gradient Boosting Classifier** with optimized parameters:
  - `n_estimators=500`
  - `max_depth=4`
- Split the data into training (67%) and testing (33%) sets.
- Evaluated the model's performance using:
  - **Accuracy:** 82.96%
  - **Precision:** 83.95%
  - **Recall:** 81.92%
  - **F1 Score:** 82.93%
- Performed **cross-validation** with 5 folds to assess generalizability:
  - **Mean accuracy:** 83.12%
  - **Standard deviation:** 1.72%.

### 5. Overfitting Check
- Evaluated performance on the training set to ensure no overfitting:
  - **Accuracy:** 90.40%
  - **Precision:** 91.46%
  - **Recall:** 89.00%
  - **F1 Score:** 90.22%.

---

## Model Grading
A separate dataset (`Kickstarter-Grading.xlsx`) was processed using the same pipeline for evaluation. The trained model achieved:
- **Accuracy:** 84.65%
- **Precision:** 86.23%
- **Recall:** 83.45%
- **F1 Score:** 84.81%.

---

## Clustering Analysis

### Preprocessing for Clustering
- Removed outliers using an **Isolation Forest**.
- Standardized the data with **MinMaxScaler**.
- Reduced dimensions using **Principal Component Analysis (PCA)** for better visualization.

### Cluster Formation
- Used the **Elbow Method** and **Dendrograms** to determine the optimal number of clusters.
- Performed **K-Means clustering** with 3 clusters and evaluated results using the **Silhouette Score** (0.62).

### Cluster Insights
Key features analyzed within each cluster:
- **preparation_duration:** Mean preparation time varied significantly across clusters.
- **campaign_duration:** Campaign lengths were mostly consistent across clusters.
- **goal:** Goals in Cluster 0 were higher on average than in other clusters.
- **video** and **show_feature_image:** These features were more prevalent in successful clusters.
- **category_Documentary:** The proportion of documentary projects was minimal across all clusters.

---

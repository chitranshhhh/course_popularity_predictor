# Course Popularity Predictor

A machine learning pipeline that predicts course demand for Advanced learners
using real-world e-learning data. The system identifies the most popular courses
using a custom popularity score, clusters learners by experience level, and
applies regression and classification models to predict and classify course demand.

The project covers the full ML pipeline — data loading, PCA dimensionality
reduction, K-Means clustering, popularity score engineering, regression
(Random Forest + XGBoost), classification (KNN + ANN), and a final evaluation
dashboard built with Streamlit.



# Dataset

| Property       | Value                                 |
|----------------|---------------------------------------|
| Total Records  | 1,00,000                              |
| Scope          | Advanced Learners Only (27,439)       |
| Courses        | AI, Solidity, Project Management      |
| Date Range     | 2020 - 2024                           |
| File           | data/data_100000_records_daily.csv    |

# How to run
1. Install Dependencies
pip install -r requirements.txt

2. Run 
streamlit run app.py

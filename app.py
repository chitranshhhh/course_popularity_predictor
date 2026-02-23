import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(
    page_title="Course Popularity Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0D0D1A; color: #FFFFFF; }
    section[data-testid="stSidebar"] { background-color: #0A0A14; }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stMetric"] {
        background-color: #1A1A2E;
        border: 1px solid #2A2A4A;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] { color: #A0A0C0 !important; }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; }
    [data-testid="stDataFrame"] { background-color: #1A1A2E; }
    .stAlert { background-color: #1A1A2E; border-radius: 8px; }
    h1, h2, h3 { color: #4C9BE8 !important; }
    h4, h5 { color: #A0C4FF !important; }
    hr { border-color: #2A2A4A; }
    .conclusion-box {
        background-color: #0F2A1A;
        border-left: 4px solid #50FA7B;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #2A1A0F;
        border-left: 4px solid #FF4C4C;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .formula-box {
        background-color: #1A1A2E;
        border: 1px solid #4C9BE8;
        border-radius: 8px;
        padding: 15px;
        font-family: monospace;
        font-size: 16px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def embed_plot(filepath, height=550):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=height, scrolling=False)
    else:
        st.warning(f"Plot not found: {filepath}\nRun main.py first to generate all outputs.")


def section(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style=\'color:#A0A0C0;\'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("---")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Course Popularity Predictor")
    st.markdown("---")
    pages = {
        "Home"            : "Home",
        "PCA Analysis"    : "PCA",
        "Clustering"      : "Clustering",
        "Popularity Score": "Popularity",
        "Preprocessing"   : "Preprocessing",
        "Regression"      : "Regression",
        "KNN Classifier"  : "KNN",
        "ANN"             : "ANN",
        "Final Results"   : "Final",
    }
    selected = st.radio("Navigation", list(pages.keys()),
                        label_visibility="collapsed")
    page = pages[selected]

    st.markdown("---")
    st.markdown("**Pipeline Status**")
    steps = [
        "Data Loading", "PCA", "Clustering",
        "Popularity Score", "Preprocessing",
        "Regression", "KNN", "ANN", "Evaluation"
    ]
    for name in steps:
        st.markdown(f"- {name}: Done")


PLOTS   = "outputs/plots"
REPORTS = "outputs/reports"


# =============================================================================
# HOME
# =============================================================================
if page == "Home":
    st.markdown(
        "<h1 style=\'text-align:center;color:#4C9BE8;\'>📊 Course Popularity Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style=\'text-align:center;color:#A0A0C0;font-size:18px;\'>Predicting course demand for Advanced learners using PCA, Clustering, Regression, KNN and ANN</p>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",     "1,00,000")
    col2.metric("Advanced Learners", "27,439")
    col3.metric("Courses Analyzed",  "3")
    col4.metric("Features Used",     "13")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Dataset Overview")
        overview = pd.DataFrame({
            "Property": ["Skill Level", "Courses", "Date Range",
                         "Target (Regression)", "Target (Classification)"],
            "Value"   : ["Advanced Only", "AI, Solidity, Project Management",
                         "2020 - 2024", "Popularity Score (0-1)",
                         "High Demand (1) / Low Demand (0)"]
        })
        st.dataframe(overview, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Pipeline Steps")
        pipeline = pd.DataFrame({
            "Step" : list(range(1, 11)),
            "Stage": ["Load & Filter Data", "PCA Column Check",
                      "PCA Dimensionality Reduction", "K-Means Clustering",
                      "Popularity Score", "Train/Test Split",
                      "Regression (RF + XGBoost)", "KNN Classifier",
                      "ANN (TensorFlow)", "Final Evaluation"],
            "Status": ["Done"] * 10
        })
        st.dataframe(pipeline, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Quick Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Regression R2", "0.9998",  "Random Forest")
    col2.metric("KNN Accuracy",       "100.00%", "K=7")
    col3.metric("ANN Accuracy",       "100.00%", "50 epochs")
    col4.metric("Top Course",         "AI",      "Score: 0.6661")

    st.markdown("---")
    st.markdown("### Course Rankings at a Glance")
    rank_data = pd.DataFrame({
        "Rank"            : ["1st", "2nd", "3rd"],
        "Course"          : ["AI", "Solidity", "Project Management"],
        "Popularity Score": [0.6661, 0.4404, 0.2683],
        "Demand"          : ["HIGH", "HIGH", "LOW"],
        "Enrollments"     : ["9,135", "9,166", "9,138"]
    })
    st.dataframe(rank_data, use_container_width=True, hide_index=True)


# =============================================================================
# PCA
# =============================================================================
elif page == "PCA":
    section("PCA Analysis",
            "Principal Component Analysis - Dimensionality Reduction from 12 to 10 features")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Features", "12")
    col2.metric("PCA Components",    "10")
    col3.metric("Variance Retained", "99.41%")
    col4.metric("Variance Lost",     "0.59%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Original Columns (Before PCA)")
        orig = pd.DataFrame({
            "No."    : list(range(1, 13)),
            "Feature": ["Course_Enc", "Exp_Enc", "Device_Enc",
                        "Years_of_Experience", "Hours_Watched",
                        "Loss_of_Job_Score", "Course_Rating",
                        "Completion_Rate", "Dropout_Risk_Score",
                        "Salary_Expectation", "Month", "Quarter"]
        })
        st.dataframe(orig, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### New Columns (After PCA)")
        new_cols = pd.DataFrame({
            "Component" : [f"PC{i}" for i in range(1, 11)],
            "Variance %": [18.808, 16.442, 11.756, 11.691, 11.561,
                           5.931, 5.877, 5.828, 5.781, 5.736],
            "Status"    : ["Kept"] * 10
        })
        st.dataframe(new_cols, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Dropped Components")
    dropped = pd.DataFrame({
        "Component"   : ["PC11", "PC12"],
        "Variance %"  : [0.3605, 0.2286],
        "Top Features": ["Years_of_Experience (0.798), Loss_of_Job_Score (0.579)",
                         "Month (0.766), Quarter (-0.643)"],
        "Reason"      : ["< 0.5% variance - negligible",
                         "< 0.5% variance - negligible"]
    })
    st.dataframe(dropped, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Component Loadings")
    st.caption("What each PC is made of - which original features contribute most")
    loadings = pd.DataFrame({
        "Component"         : [f"PC{i}" for i in range(1, 11)],
        "Dominant Feature 1": ["Loss_of_Job_Score (0.77)", "Quarter (0.76)",
                               "Exp_Enc (0.78)", "Device_Enc (0.93)",
                               "Course_Enc (0.74)", "Hours_Watched (0.79)",
                               "Salary_Expectation (0.77)", "Dropout_Risk_Score (0.74)",
                               "Completion_Rate (0.66)", "Course_Rating (0.68)"],
        "Dominant Feature 2": ["Years_of_Experience (-0.60)", "Month (0.64)",
                               "Course_Enc (0.58)", "Exp_Enc (-0.15)",
                               "Exp_Enc (-0.55)", "Course_Rating (-0.50)",
                               "Completion_Rate (0.56)", "Hours_Watched (-0.49)",
                               "Salary_Expectation (-0.61)", "Completion_Rate (-0.32)"]
    })
    st.dataframe(loadings, use_container_width=True, hide_index=True)


# =============================================================================
# CLUSTERING
# =============================================================================
elif page == "Clustering":
    section("K-Means Clustering",
            "Grouping Advanced learners into 3 clusters by experience level")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters (K)",  "3")
    col2.metric("Cluster 0",     "Student",                "9,215 learners")
    col3.metric("Cluster 1",     "Mid-Level Professional", "9,103 learners")
    col4.metric("Cluster 2",     "Early Professional",     "9,121 learners")

    st.markdown("---")
    st.markdown("### Cluster Names and Profiles")
    cluster_df = pd.DataFrame({
        "Cluster"           : [0, 1, 2],
        "Name"              : ["Student", "Mid-Level Professional", "Early Professional"],
        "Avg Yrs Experience": [0.0, 10.4, 3.0],
        "Avg Completion %"  : [69.8, 69.9, 70.0],
        "Avg Dropout Risk"  : [0.50, 0.51, 0.50],
        "Learners"          : ["9,215", "9,103", "9,121"]
    })
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Elbow Method (2D)")
        st.caption("Chosen K=3 marked in green")
        embed_plot(f"{PLOTS}/clustering_elbow.html", height=450)
    with col2:
        st.markdown("### Cluster vs Course Heatmap (2D)")
        st.caption("Enrollment count per cluster per course")
        embed_plot(f"{PLOTS}/clustering_course_heatmap.html", height=450)

    st.markdown("---")
    st.markdown("### 3D Cluster Scatter - Learners in PCA Space")
    st.caption("Blue = Student | Red = Mid-Level Professional | Green = Early Professional")
    embed_plot(f"{PLOTS}/clustering_3d_scatter.html", height=600)


# =============================================================================
# POPULARITY SCORE
# =============================================================================
elif page == "Popularity":
    section("Popularity Score",
            "Weighted formula combining 5 signals to rank course demand")

    st.markdown("### Popularity Score Formula")
    st.markdown("""
    <div class="formula-box">
        Score = 0.25 x Hours_Watched + 0.25 x Course_Rating + 0.25 x Completion_Rate
                + 0.15 x (1 - Dropout_Risk) + 0.10 x Enrollment_Count
        <br><small style="color:#A0A0C0;">(All signals normalized to 0-1 before applying weights)</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Final Course Rankings")
    ranking_df = pd.DataFrame({
        "Rank"            : ["1st", "2nd", "3rd"],
        "Course"          : ["AI", "Solidity", "Project Management"],
        "Popularity Score": [0.6661, 0.4404, 0.2683],
        "Demand Label"    : ["HIGH", "HIGH", "LOW"],
        "Enrollments"     : ["9,135", "9,166", "9,138"],
        "Median Threshold": [0.4404, 0.4404, 0.4404]
    })
    st.dataframe(ranking_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Popularity Score 3D Bar")
        embed_plot(f"{PLOTS}/popularity_score_3d.html", height=500)
    with col2:
        st.markdown("### Score Breakdown per Course (2D)")
        st.caption("Each signal contribution per course")
        embed_plot(f"{PLOTS}/popularity_score_breakdown.html", height=500)


# =============================================================================
# PREPROCESSING
# =============================================================================
elif page == "Preprocessing":
    section("Preprocessing and Train/Test Split",
            "80/20 split with no data leakage - PCA fitted only on training data")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Features",       "13")
    col2.metric("PCA Components", "11")
    col3.metric("Training Rows",  "21,951")
    col4.metric("Test Rows",      "5,488")

    st.markdown("---")
    st.markdown("### Preprocessing Steps")
    steps_df = pd.DataFrame({
        "Step"        : list(range(1, 7)),
        "Action"      : [
            "Label encode Course_Name, Experience_Category, Device_Type",
            "Extract Month, Quarter, DayOfWeek from Date",
            "80/20 train-test split (random_state=42)",
            "MinMaxScaler - fit on train only, transform test",
            "PCA (95% variance) - fit on train only, transform test",
            "Return X_train_pca, X_test_pca, y_reg, y_cls"
        ],
        "Leakage Safe": ["Yes"] * 6
    })
    st.dataframe(steps_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Train vs Test Split (3D PCA Space)")
        st.caption("Blue = Train | Red = Test")
        embed_plot(f"{PLOTS}/preprocessing_train_test_split_3d.html", height=500)
    with col2:
        st.markdown("### Class Distribution (2D)")
        st.caption("Low Demand vs High Demand in Train and Test sets")
        embed_plot(f"{PLOTS}/preprocessing_class_distribution.html", height=500)


# =============================================================================
# REGRESSION
# =============================================================================
elif page == "Regression":
    section("Regression",
            "Predicting exact Popularity Score - Linear Regression vs RF vs XGBoost")

    st.markdown("""
    <div class="warning-box">
        <strong>Why Linear Regression Failed:</strong>
        The target (Popularity Score) has only <strong>3 unique values</strong> (0.27, 0.44, 0.67)
        - one per course. Linear Regression predicted <strong>372 unique values</strong> and
        achieved R2=0.32, proving it is not suitable for this problem.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Model Scorecard")
    scorecard = pd.DataFrame({
        "Model"  : ["Linear Regression", "Random Forest", "XGBoost"],
        "R2"     : [0.3238, 0.9998, 0.9992],
        "RMSE"   : [0.133285, 0.002349, 0.004636],
        "MAE"    : [0.125862, 0.000065, 0.000134],
        "Verdict": ["Failed - Baseline Proof", "Best Model", "Excellent"]
    })
    st.dataframe(scorecard, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Linear Regression Failure Proof (2D)")
        st.caption("Actual has 3 values - Linear predicts a spread. Clear failure.")
        embed_plot(f"{PLOTS}/regression_linear_failure_proof.html", height=450)
    with col2:
        st.markdown("### R2 Comparison (2D)")
        st.caption("Linear=0.32 vs RF/XGB=0.99 - proof of why we use RF+XGB")
        embed_plot(f"{PLOTS}/evaluator_regression_comparison.html", height=450)

    st.markdown("---")
    st.markdown("### Actual vs Predicted vs Residual (3D)")
    st.caption("All 3 models - rotate to see residual errors")
    embed_plot(f"{PLOTS}/regression_3d_actual_vs_predicted.html", height=580)


# =============================================================================
# KNN
# =============================================================================
elif page == "KNN":
    section("KNN Classifier",
            "K-Nearest Neighbours - predicting High/Low Demand (binary classification)")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best K",          "7")
    col2.metric("Accuracy",        "100.00%")
    col3.metric("F1 Score",        "1.00")
    col4.metric("Distance Metric", "Euclidean")

    st.markdown("---")
    st.markdown("### Classification Report")
    cls_report = pd.DataFrame({
        "Class"    : ["Low Demand", "High Demand", "Macro Avg", "Weighted Avg"],
        "Precision": [1.00, 1.00, 1.00, 1.00],
        "Recall"   : [1.00, 1.00, 1.00, 1.00],
        "F1-Score" : [1.00, 1.00, 1.00, 1.00],
        "Support"  : [1773, 3715, 5488, 5488]
    })
    st.dataframe(cls_report, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### K vs Accuracy (2D)")
        st.caption("Best K highlighted in red")
        embed_plot(f"{PLOTS}/knn_k_vs_accuracy.html", height=450)
    with col2:
        st.markdown("### Confusion Matrix (2D)")
        st.caption("Perfect classification - 0 errors")
        embed_plot(f"{PLOTS}/knn_confusion_matrix.html", height=450)


# =============================================================================
# ANN
# =============================================================================
elif page == "ANN":
    section("ANN - Artificial Neural Network",
            "TensorFlow deep learning model for High/Low Demand classification")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", "100.00%")
    col2.metric("Epochs Run",    "50")
    col3.metric("Optimizer",     "Adam")
    col4.metric("Parameters",    "3,777")

    st.markdown("---")
    st.markdown("### Architecture")
    arch_df = pd.DataFrame({
        "Layer"     : ["Input", "Dense + BN + Dropout", "Dense + BN + Dropout",
                       "Dense", "Output"],
        "Neurons"   : [11, 64, 32, 16, 1],
        "Activation": ["-", "ReLU", "ReLU", "ReLU", "Sigmoid"],
        "Dropout"   : ["-", "0.3", "0.2", "-", "-"],
        "Params"    : ["-", "768 + 256", "2,080 + 128", "528", "17"]
    })
    st.dataframe(arch_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Training Loss Curve (2D)")
        st.caption("Train vs Validation loss per epoch")
        embed_plot(f"{PLOTS}/ann_training_loss.html", height=430)
    with col2:
        st.markdown("### Training Accuracy Curve (2D)")
        st.caption("Train vs Validation accuracy per epoch")
        embed_plot(f"{PLOTS}/ann_training_accuracy.html", height=430)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Confusion Matrix (2D)")
        embed_plot(f"{PLOTS}/ann_confusion_matrix.html", height=430)
    with col2:
        st.markdown("### 3D Predicted Classes in PCA Space")
        embed_plot(f"{PLOTS}/ann_3d_predicted_classes.html", height=430)


# =============================================================================
# FINAL RESULTS
# =============================================================================
elif page == "Final":
    section("Final Results",
            "Complete model comparison, course rankings and project conclusion")

    st.markdown("### Final Course Popularity Ranking")
    embed_plot(f"{PLOTS}/evaluator_course_ranking.html", height=380)

    st.markdown("---")
    st.markdown("### Complete Model Scorecard")
    full_scorecard = pd.DataFrame({
        "Model"        : ["Linear Regression", "Random Forest", "XGBoost", "KNN", "ANN"],
        "Task"         : ["Regression", "Regression", "Regression",
                          "Classification", "Classification"],
        "R2 / Accuracy": [0.3238, 0.9998, 0.9992, 1.0000, 1.0000],
        "RMSE / F1"    : [0.1333, 0.0023, 0.0046, 1.0000, 1.0000],
        "Verdict"      : ["Baseline proof", "Best Regressor",
                          "Excellent", "Perfect", "Perfect"]
    })
    st.dataframe(full_scorecard, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### All Models Performance (3D)")
        embed_plot(f"{PLOTS}/evaluator_model_performance_3d.html", height=480)
    with col2:
        st.markdown("### Cluster Quality - Silhouette Score (2D)")
        embed_plot(f"{PLOTS}/evaluator_silhouette_score.html", height=480)

    st.markdown("---")
    st.markdown("### Project Conclusion")
    st.markdown("""
    <div class="conclusion-box">
        <h4 style="color:#50FA7B;">Key Findings</h4>
        <ul>
            <li><strong>AI</strong> is the most popular advanced course (Score: 0.6661) - driven by high hours watched, completion rate and course rating.</li>
            <li><strong>Solidity</strong> ranks 2nd (Score: 0.4404) - high enrollment count but moderate rating.</li>
            <li><strong>Project Management</strong> ranks 3rd (Score: 0.2683) - classified as Low Demand.</li>
            <li><strong>K-Means</strong> cleanly separated learners by experience level (Student / Early / Mid-Level Professional).</li>
            <li><strong>Linear Regression failed</strong> (R2=0.32) - target has only 3 discrete values, proving non-linear models are needed.</li>
            <li><strong>Random Forest</strong> is the best regression model (R2=0.9998, RMSE=0.0023).</li>
            <li><strong>KNN (K=7) and ANN</strong> both achieve <strong>100% classification accuracy</strong>.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
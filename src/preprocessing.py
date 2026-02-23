import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def preprocess(df_merged, plot_dir="outputs/plots"):

    df = df_merged.copy()

    # ── Encode categorical columns ────────────────────────────────────────────
    le_course = LabelEncoder()
    le_exp    = LabelEncoder()
    le_device = LabelEncoder()

    df["Course_Enc"] = le_course.fit_transform(df["Course_Name"])
    df["Exp_Enc"]    = le_exp.fit_transform(df["Experience_Category"])
    df["Device_Enc"] = le_device.fit_transform(df["Device_Type"])

    # ── Extract time features ─────────────────────────────────────────────────
    df["Month"]     = df["Date"].dt.month
    df["Quarter"]   = df["Date"].dt.quarter
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    # ── Define features and targets ───────────────────────────────────────────
    feature_cols = [
        "Course_Enc", "Exp_Enc", "Device_Enc",
        "Years_of_Experience", "Hours_Watched", "Loss_of_Job_Score",
        "Course_Rating", "Completion_Rate", "Dropout_Risk_Score",
        "Salary_Expectation", "Month", "Quarter", "DayOfWeek"
    ]

    X     = df[feature_cols].values
    y_reg = df["Popularity_Score"].values
    y_cls = df["Demand_Label"].values

    # ── Train/Test Split (80/20) — BEFORE scaling to prevent leakage ──────────
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test) = train_test_split(
        X, y_reg, y_cls,
        test_size=0.2,
        random_state=42
    )

    # ── Scale — fit on train only ─────────────────────────────────────────────
    scaler         = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── PCA — fit on train only ───────────────────────────────────────────────
    pca            = PCA(n_components=0.95, random_state=42)
    X_train_pca    = pca.fit_transform(X_train_scaled)
    X_test_pca     = pca.transform(X_test_scaled)

    # ── 3D Scatter: Train vs Test split visualisation ─────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=X_train_pca[:, 0],
        y=X_train_pca[:, 1],
        z=X_train_pca[:, 2],
        mode="markers",
        name=f"Train ({len(X_train_pca):,})",
        marker=dict(size=2, color="#4C9BE8", opacity=0.5)
    ))

    fig.add_trace(go.Scatter3d(
        x=X_test_pca[:, 0],
        y=X_test_pca[:, 1],
        z=X_test_pca[:, 2],
        mode="markers",
        name=f"Test ({len(X_test_pca):,})",
        marker=dict(size=2, color="#FF4C4C", opacity=0.6)
    ))

    fig.update_layout(
        title=dict(text="Train vs Test Split — PCA Space (3D)",
                   font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="PC1"),
            yaxis=dict(title="PC2"),
            zaxis=dict(title="PC3"),
            bgcolor="rgb(10,10,30)",
            xaxis_backgroundcolor="rgb(10,10,30)",
            yaxis_backgroundcolor="rgb(10,10,30)",
            zaxis_backgroundcolor="rgb(10,10,30)",
            xaxis_gridcolor="gray",
            yaxis_gridcolor="gray",
            zaxis_gridcolor="gray",
        ),
        paper_bgcolor="rgb(15,15,40)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)"),
        scene_camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))
    )
    fig.write_html(f"{plot_dir}/preprocessing_train_test_split_3d.html")

    # ── 2D Class Distribution Bar ─────────────────────────────────────────────
    train_counts = np.bincount(y_cls_train.astype(int))
    test_counts  = np.bincount(y_cls_test.astype(int))

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="Train", x=["Low Demand (0)", "High Demand (1)"],
        y=train_counts, marker_color="#4C9BE8"
    ))
    fig2.add_trace(go.Bar(
        name="Test", x=["Low Demand (0)", "High Demand (1)"],
        y=test_counts, marker_color="#FF4C4C"
    ))
    fig2.update_layout(
        barmode="group",
        title=dict(text="Class Distribution — Train vs Test (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Demand Label"),
        yaxis=dict(title="Count"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)"),
        xaxis_gridcolor="gray",
        yaxis_gridcolor="gray"
    )
    fig2.write_html(f"{plot_dir}/preprocessing_class_distribution.html")

    print("\n=== PREPROCESSING SUMMARY ===")
    print(f"  Features            : {len(feature_cols)}")
    print(f"  PCA components      : {pca.n_components_}")
    print(f"  Variance retained   : {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"  X_train_pca shape   : {X_train_pca.shape}")
    print(f"  X_test_pca shape    : {X_test_pca.shape}")
    print(f"  y_cls_train dist    : Low={train_counts[0]:,}  High={train_counts[1]:,}")
    print(f"  y_cls_test dist     : Low={test_counts[0]:,}   High={test_counts[1]:,}")
    print(f"\nPlots saved:")
    print(f"  {plot_dir}/preprocessing_train_test_split_3d.html")
    print(f"  {plot_dir}/preprocessing_class_distribution.html")

    return (X_train_pca, X_test_pca,
            y_reg_train, y_reg_test,
            y_cls_train, y_cls_test,
            scaler, pca, feature_cols,
            le_course, le_exp, le_device)
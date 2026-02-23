import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def prepare_features(df_adv):
    df = df_adv.copy()
    le_course = LabelEncoder()
    le_exp    = LabelEncoder()
    le_device = LabelEncoder()
    df["Course_Enc"] = le_course.fit_transform(df["Course_Name"])
    df["Exp_Enc"]    = le_exp.fit_transform(df["Experience_Category"])
    df["Device_Enc"] = le_device.fit_transform(df["Device_Type"])
    df["Month"]      = df["Date"].dt.month
    df["Quarter"]    = df["Date"].dt.quarter
    feature_cols = [
        "Course_Enc", "Exp_Enc", "Device_Enc",
        "Years_of_Experience", "Hours_Watched", "Loss_of_Job_Score",
        "Course_Rating", "Completion_Rate", "Dropout_Risk_Score",
        "Salary_Expectation", "Month", "Quarter"
    ]
    X        = df[feature_cols].values
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols, le_course, le_exp, le_device


def apply_pca(X_scaled, variance_threshold=0.95, plot_path=None):
    pca   = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    if plot_path:
        components = list(range(1, pca.n_components_ + 1))
        individual = pca.explained_variance_ratio_ * 100
        cumulative = np.cumsum(individual).tolist()

        fig = go.Figure()

        for i, (comp, var) in enumerate(zip(components, individual)):
            fig.add_trace(go.Scatter3d(
                x=[comp, comp], y=[0, var], z=[i, i],
                mode="lines",
                line=dict(color=f"rgba({30+i*20},{100+i*10},200,0.9)", width=20),
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[comp], y=[var], z=[i],
                mode="markers+text",
                marker=dict(size=6, color="white",
                            line=dict(color="black", width=1)),
                text=[f"{var:.1f}%"],
                textposition="top center",
                textfont=dict(size=10, color="white"),
                showlegend=False
            ))

        fig.add_trace(go.Scatter3d(
            x=components, y=cumulative, z=list(range(len(components))),
            mode="lines+markers", name="Cumulative Variance",
            line=dict(color="#FF4C4C", width=5),
            marker=dict(size=6, color="#FF4C4C")
        ))

        fig.add_trace(go.Scatter3d(
            x=[components[0], components[-1]], y=[95, 95],
            z=[0, len(components)-1],
            mode="lines", name="95% Threshold",
            line=dict(color="lightgreen", width=4, dash="dash")
        ))

        fig.update_layout(
            title=dict(text="PCA 3D — Explained Variance per Component",
                       font=dict(size=18)),
            scene=dict(
                xaxis=dict(title="Component", tickvals=components),
                yaxis=dict(title="Variance (%)"),
                zaxis=dict(title="Index"),
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
            scene_camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2))
        )

        fig.write_html(plot_path)

    return X_pca, pca

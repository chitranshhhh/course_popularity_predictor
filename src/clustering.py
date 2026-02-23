import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans


CLUSTER_NAME_MAP = {
    0: "Students",
    1: "Mid-Level Professionals",
    2: "Early Professionals"
}


def run_clustering(X_pca, df_adv, plot_dir="outputs/plots", n_clusters=3):

    # ── Elbow Method ──────────────────────────────────────────────────────────
    k_range  = list(range(2, 10))
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_pca)
        inertias.append(km.inertia_)

    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=k_range, y=inertias,
        mode="lines+markers",
        line=dict(color="#4C9BE8", width=3),
        marker=dict(size=9, color="#FF4C4C", line=dict(color="white", width=2)),
        name="Inertia"
    ))
    fig_elbow.add_vline(
        x=n_clusters, line_dash="dash", line_color="lightgreen",
        annotation_text=f"Chosen K={n_clusters}",
        annotation_font_color="lightgreen"
    )
    fig_elbow.update_layout(
        title=dict(text="Elbow Method — Optimal Number of Clusters",
                   font=dict(size=18)),
        xaxis=dict(title="Number of Clusters (K)", tickvals=k_range,
                   gridcolor="gray"),
        yaxis=dict(title="Inertia", gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white")
    )
    fig_elbow.write_html(f"{plot_dir}/clustering_elbow.html")

    # ── Train Final KMeans ────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    # ── Map cluster IDs to names based on Experience_Category ─────────────────
    df_temp   = df_adv.copy()
    df_temp["Cluster"] = labels
    exp_dist  = df_temp.groupby(["Cluster","Experience_Category"]).size().unstack(fill_value=0)

    # Dynamically assign names based on dominant Experience_Category per cluster
    cluster_label_map = {}
    for cluster_id in exp_dist.index:
        dominant = exp_dist.loc[cluster_id].idxmax()
        cluster_label_map[cluster_id] = dominant

    df_temp["Cluster_Name"] = df_temp["Cluster"].map(cluster_label_map)

    # ── 3D Cluster Scatter ────────────────────────────────────────────────────
    color_map = {
        "Student"               : "#4C9BE8",
        "Mid-Level Professional": "#FF4C4C",
        "Early Professional"    : "#50FA7B"
    }

    fig_scatter = go.Figure()
    for cluster_id in range(n_clusters):
        mask        = labels == cluster_id
        name        = cluster_label_map[cluster_id]
        color       = list(color_map.values())[cluster_id]
        hover_text  = [
            f"Cluster: {name}<br>Course: {c}<br>Completion: {comp:.1f}%"
            for c, comp in zip(
                df_adv["Course_Name"].values[mask],
                df_adv["Completion_Rate"].values[mask]
            )
        ]
        fig_scatter.add_trace(go.Scatter3d(
            x=X_pca[mask, 0],
            y=X_pca[mask, 1],
            z=X_pca[mask, 2],
            mode="markers",
            name=name,
            marker=dict(size=3, color=color, opacity=0.6,
                        line=dict(width=0)),
            text=hover_text,
            hoverinfo="text"
        ))

    centroids = kmeans.cluster_centers_
    fig_scatter.add_trace(go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode="markers+text",
        name="Centroids",
        marker=dict(size=10, color="white", symbol="diamond",
                    line=dict(color="black", width=2)),
        text=[cluster_label_map[i] for i in range(n_clusters)],
        textposition="top center",
        textfont=dict(size=11, color="white")
    ))

    fig_scatter.update_layout(
        title=dict(text="K-Means 3D — Learner Clusters by Experience",
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
    fig_scatter.write_html(f"{plot_dir}/clustering_3d_scatter.html")

    # ── 2D Heatmap: Cluster Name vs Course ───────────────────────────────────
    pivot = df_temp.groupby(["Cluster_Name","Course_Name"]).size().unstack(fill_value=0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values,
        texttemplate="%{text}",
        textfont=dict(size=14),
        hoverongaps=False
    ))
    fig_heat.update_layout(
        title=dict(text="Cluster (Learner Type) vs Course — Enrollment Count",
                   font=dict(size=18)),
        xaxis=dict(title="Course"),
        yaxis=dict(title="Learner Cluster"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white")
    )
    fig_heat.write_html(f"{plot_dir}/clustering_course_heatmap.html")

    print("\n=== CLUSTER NAMES ===")
    for cluster_id, name in cluster_label_map.items():
        count = (labels == cluster_id).sum()
        print(f"  Cluster {cluster_id} → '{name}'  ({count:,} learners)")

    print("\nPlots saved:")
    print(f"  {plot_dir}/clustering_elbow.html")
    print(f"  {plot_dir}/clustering_3d_scatter.html")
    print(f"  {plot_dir}/clustering_course_heatmap.html")

    return kmeans, labels, cluster_label_map
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def _normalize(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def compute_popularity(df_adv, plot_dir="outputs/plots"):

    # ── Aggregate per course ──────────────────────────────────────────────────
    agg = df_adv.groupby("Course_Name").agg(
        Hours_Watched      = ("Hours_Watched",      "mean"),
        Course_Rating      = ("Course_Rating",       "mean"),
        Completion_Rate    = ("Completion_Rate",     "mean"),
        Dropout_Risk_Score = ("Dropout_Risk_Score",  "mean"),
        Loss_of_Job_Score  = ("Loss_of_Job_Score",   "mean"),
        Salary_Expectation = ("Salary_Expectation",  "mean"),
        Enrollment_Count   = ("Learner_ID",          "count")
    ).reset_index()

    # ── Normalize each signal ─────────────────────────────────────────────────
    agg["HW_norm"]   = _normalize(agg["Hours_Watched"])
    agg["CR_norm"]   = _normalize(agg["Course_Rating"])
    agg["COMP_norm"] = _normalize(agg["Completion_Rate"])
    agg["DR_norm"]   = _normalize(agg["Dropout_Risk_Score"])
    agg["ENR_norm"]  = _normalize(agg["Enrollment_Count"])

    # ── Weighted Popularity Score ─────────────────────────────────────────────
    # Formula:
    # Score = 0.25*Hours_Watched + 0.25*Course_Rating + 0.25*Completion_Rate
    #       + 0.15*(1 - Dropout_Risk) + 0.10*Enrollment_Count
    agg["Popularity_Score"] = (
        0.25 * agg["HW_norm"]
      + 0.25 * agg["CR_norm"]
      + 0.25 * agg["COMP_norm"]
      + 0.15 * (1 - agg["DR_norm"])
      + 0.10 * agg["ENR_norm"]
    )

    # ── Binary Demand Label ───────────────────────────────────────────────────
    median_score        = agg["Popularity_Score"].median()
    agg["Demand_Label"] = (agg["Popularity_Score"] >= median_score).astype(int)
    agg                 = agg.sort_values("Popularity_Score", ascending=False).reset_index(drop=True)

    # ── Merge back to learner-level df ────────────────────────────────────────
    score_map   = agg.set_index("Course_Name")[["Popularity_Score","Demand_Label"]]
    df_merged   = df_adv.copy()
    df_merged   = df_merged.join(score_map, on="Course_Name")

    # ── 3D Bar Chart: Course vs Popularity Score ──────────────────────────────
    courses = agg["Course_Name"].tolist()
    scores  = agg["Popularity_Score"].tolist()
    colors  = ["#50FA7B" if d == 1 else "#FF4C4C"
               for d in agg["Demand_Label"].tolist()]

    fig = go.Figure()

    for i, (course, score, color) in enumerate(zip(courses, scores, colors)):
        fig.add_trace(go.Scatter3d(
            x=[i, i],
            y=[0, score],
            z=[0, 0],
            mode="lines",
            line=dict(color=color, width=25),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[i],
            y=[score],
            z=[0],
            mode="markers+text",
            marker=dict(size=8, color=color,
                        line=dict(color="white", width=1)),
            text=[f"{score:.4f}"],
            textposition="top center",
            textfont=dict(size=11, color="white"),
            showlegend=False
        ))

    # Legend traces
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=8, color="#50FA7B"),
        name="High Demand"
    ))
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(size=8, color="#FF4C4C"),
        name="Low Demand"
    ))

    fig.update_layout(
        title=dict(
            text="Course Popularity Score — Advanced Level (3D)",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title="Course", tickvals=list(range(len(courses))),
                       ticktext=courses),
            yaxis=dict(title="Popularity Score"),
            zaxis=dict(title=""),
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
    fig.write_html(f"{plot_dir}/popularity_score_3d.html")

    # ── 2D Breakdown: Score components per course ─────────────────────────────
    components_df = agg[["Course_Name","HW_norm","CR_norm",
                          "COMP_norm","DR_norm","ENR_norm"]].copy()
    components_df.columns = ["Course","Hours Watched","Course Rating",
                              "Completion Rate","Dropout Risk (inv)","Enrollment"]
    components_df["Dropout Risk (inv)"] = 1 - agg["DR_norm"]

    fig2 = go.Figure()
    signal_cols   = ["Hours Watched","Course Rating","Completion Rate",
                     "Dropout Risk (inv)","Enrollment"]
    signal_colors = ["#4C9BE8","#FF9F43","#50FA7B","#FF4C4C","#BD93F9"]

    for col, color in zip(signal_cols, signal_colors):
        fig2.add_trace(go.Bar(
            name=col,
            x=components_df["Course"],
            y=components_df[col],
            marker_color=color
        ))

    fig2.update_layout(
        barmode="group",
        title=dict(text="Popularity Score Breakdown per Course (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Course"),
        yaxis=dict(title="Normalised Value (0-1)"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)"),
        xaxis_gridcolor="gray",
        yaxis_gridcolor="gray"
    )
    fig2.write_html(f"{plot_dir}/popularity_score_breakdown.html")

    print("\n=== POPULARITY SCORE RESULTS ===\n")
    for _, row in agg.iterrows():
        label = "HIGH" if row["Demand_Label"] == 1 else "LOW"
        print(f"  {row['Course_Name']:<22}  "
              f"Score: {row['Popularity_Score']:.6f}  "
              f"Demand: {label}  "
              f"Enrollments: {int(row['Enrollment_Count']):,}")

    print(f"\n  Median threshold : {median_score:.6f}")
    print(f"\nPlots saved:")
    print(f"  {plot_dir}/popularity_score_3d.html")
    print(f"  {plot_dir}/popularity_score_breakdown.html")

    return agg, df_merged
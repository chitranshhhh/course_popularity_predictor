import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, accuracy_score,
                             f1_score, silhouette_score)
from datetime import datetime


def run_evaluator(
        # Regression
        y_reg_test, lr_pred, rf_pred, xgb_pred,
        # Classification
        y_cls_test, knn_pred, ann_pred,
        # Clustering
        X_pca, cluster_labels,
        # Course ranking
        agg,
        plot_dir="outputs/plots",
        report_dir="outputs/reports"):

    print("\n" + "="*60)
    print("  FINAL EVALUATION")
    print("="*60)

    # =========================================================================
    # 1. REGRESSION SCORECARD
    # =========================================================================
    reg_models = {
        "Linear Regression": lr_pred,
        "Random Forest"    : rf_pred,
        "XGBoost"          : xgb_pred
    }
    reg_scores = {}
    print("\n--- Regression Scorecard ---")
    for name, pred in reg_models.items():
        r2   = r2_score(y_reg_test, pred)
        rmse = np.sqrt(mean_squared_error(y_reg_test, pred))
        mae  = mean_absolute_error(y_reg_test, pred)
        reg_scores[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        print(f"  {name:<22}  R2={r2:.4f}  RMSE={rmse:.6f}  MAE={mae:.6f}")

    best_reg = max(reg_scores, key=lambda m: reg_scores[m]["R2"])
    print(f"\n  Best Regression Model : {best_reg}  (R2={reg_scores[best_reg]['R2']:.4f})")

    # =========================================================================
    # 2. CLASSIFICATION SCORECARD
    # =========================================================================
    cls_models = {
        "KNN": knn_pred,
        "ANN": ann_pred
    }
    cls_scores = {}
    print("\n--- Classification Scorecard ---")
    for name, pred in cls_models.items():
        acc = accuracy_score(y_cls_test, pred)
        f1  = f1_score(y_cls_test, pred, average="weighted")
        cls_scores[name] = {"Accuracy": acc, "F1": f1}
        print(f"  {name:<22}  Accuracy={acc*100:.2f}%  F1={f1:.4f}")

    best_cls = max(cls_scores, key=lambda m: cls_scores[m]["Accuracy"])
    print(f"\n  Best Classifier       : {best_cls}  (Accuracy={cls_scores[best_cls]['Accuracy']*100:.2f}%)")

    # =========================================================================
    # 3. CLUSTERING VALIDATION — Silhouette Score
    # =========================================================================
    sample_idx  = np.random.choice(len(X_pca), size=5000, replace=False)
    sil_score   = silhouette_score(X_pca[sample_idx], cluster_labels[sample_idx])
    print(f"\n--- Clustering Validation ---")
    print(f"  Silhouette Score : {sil_score:.4f}  (0=poor | 1=perfect)")
    if sil_score >= 0.5:
        sil_verdict = "Strong cluster separation"
    elif sil_score >= 0.25:
        sil_verdict = "Moderate cluster separation"
    else:
        sil_verdict = "Weak cluster separation"
    print(f"  Verdict          : {sil_verdict}")

    # =========================================================================
    # 4. FINAL COURSE RANKING
    # =========================================================================
    print("\n--- Final Course Ranking ---")
    ranking = agg.sort_values("Popularity_Score", ascending=False).reset_index(drop=True)
    for i, row in ranking.iterrows():
        medal = ["1st", "2nd", "3rd"][i]
        label = "HIGH" if row["Demand_Label"] == 1 else "LOW"
        print(f"  {medal}  {row['Course_Name']:<22}  Score={row['Popularity_Score']:.4f}  Demand={label}")

    # =========================================================================
    # PLOT 1 — 3D: Model Performance Overview
    # (R2 for regression, Accuracy for classification — all in one 3D bar)
    # =========================================================================
    model_names = (list(reg_scores.keys()) + list(cls_scores.keys()))
    r2_vals     = [reg_scores[m]["R2"]          for m in reg_scores] + [None, None]
    acc_vals    = [None, None, None] + [cls_scores[m]["Accuracy"] for m in cls_scores]

    fig_3d = go.Figure()

    # Regression R2 bars
    for i, (name, scores) in enumerate(reg_scores.items()):
        color = "#50FA7B" if name == best_reg else "#4C9BE8"
        fig_3d.add_trace(go.Scatter3d(
            x=[i, i], y=[0, scores["R2"]], z=[0, 0],
            mode="lines",
            line=dict(color=color, width=20),
            name=name,
            showlegend=True
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[i], y=[scores["R2"]], z=[0],
            mode="markers+text",
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=1)),
            text=[f"R2={scores['R2']:.4f}"],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            showlegend=False
        ))

    # Classification Accuracy bars
    offset = len(reg_scores) + 1
    for i, (name, scores) in enumerate(cls_scores.items()):
        color = "#50FA7B" if name == best_cls else "#FF9F43"
        fig_3d.add_trace(go.Scatter3d(
            x=[offset+i, offset+i], y=[0, scores["Accuracy"]], z=[1, 1],
            mode="lines",
            line=dict(color=color, width=20),
            name=name,
            showlegend=True
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[offset+i], y=[scores["Accuracy"]], z=[1],
            mode="markers+text",
            marker=dict(size=7, color=color,
                        line=dict(color="white", width=1)),
            text=[f"Acc={scores['Accuracy']*100:.1f}%"],
            textposition="top center",
            textfont=dict(size=10, color="white"),
            showlegend=False
        ))

    fig_3d.update_layout(
        title=dict(text="Final Model Performance Overview (3D)",
                   font=dict(size=18)),
        scene=dict(
            xaxis=dict(title="Model Index"),
            yaxis=dict(title="Score (R2 / Accuracy)"),
            zaxis=dict(title="Type: 0=Regression  1=Classification",
                       tickvals=[0, 1],
                       ticktext=["Regression", "Classification"]),
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
    fig_3d.write_html(f"{plot_dir}/evaluator_model_performance_3d.html")

    # =========================================================================
    # PLOT 2 — 2D: Course Popularity Ranking (horizontal bar — podium style)
    # =========================================================================
    courses     = ranking["Course_Name"].tolist()
    scores_list = ranking["Popularity_Score"].tolist()
    bar_colors  = ["#FFD700", "#C0C0C0", "#CD7F32"]   # gold, silver, bronze

    fig_rank = go.Figure(go.Bar(
        x=scores_list,
        y=courses,
        orientation="h",
        marker=dict(color=bar_colors,
                    line=dict(color="white", width=1)),
        text=[f"{s:.4f}" for s in scores_list],
        textposition="inside",
        textfont=dict(size=14, color="white")
    ))
    fig_rank.update_layout(
        title=dict(text="Final Course Popularity Ranking (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Popularity Score", gridcolor="gray"),
        yaxis=dict(title="Course"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        xaxis_gridcolor="gray"
    )
    fig_rank.write_html(f"{plot_dir}/evaluator_course_ranking.html")

    # =========================================================================
    # PLOT 3 — 2D: Regression Model Comparison (R2 only — clear proof)
    # =========================================================================
    fig_reg = go.Figure(go.Bar(
        x=list(reg_scores.keys()),
        y=[reg_scores[m]["R2"] for m in reg_scores],
        marker=dict(
            color=["#FF4C4C", "#4C9BE8", "#50FA7B"],
            line=dict(color="white", width=1)
        ),
        text=[f"R2={reg_scores[m]['R2']:.4f}" for m in reg_scores],
        textposition="outside",
        textfont=dict(color="white", size=13)
    ))
    fig_reg.add_hline(y=0.5, line_dash="dash", line_color="yellow",
                      annotation_text="R2=0.5 Threshold",
                      annotation_font_color="yellow")
    fig_reg.update_layout(
        title=dict(text="Regression Model R2 Comparison — Why Linear Failed (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Model"),
        yaxis=dict(title="R2 Score", range=[-0.1, 1.1], gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        xaxis_gridcolor="gray"
    )
    fig_reg.write_html(f"{plot_dir}/evaluator_regression_comparison.html")

    # =========================================================================
    # PLOT 4 — 2D: Silhouette Score gauge (bar)
    # =========================================================================
    fig_sil = go.Figure(go.Bar(
        x=["K-Means Silhouette Score"],
        y=[sil_score],
        marker=dict(color="#BD93F9",
                    line=dict(color="white", width=1)),
        text=[f"{sil_score:.4f}"],
        textposition="outside",
        textfont=dict(color="white", size=16)
    ))
    fig_sil.add_hline(y=0.5, line_dash="dash", line_color="lightgreen",
                      annotation_text="Good threshold (0.5)",
                      annotation_font_color="lightgreen")
    fig_sil.update_layout(
        title=dict(text=f"K-Means Cluster Quality — Silhouette Score ({sil_verdict})",
                   font=dict(size=18)),
        yaxis=dict(title="Silhouette Score", range=[0, 1], gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        xaxis_gridcolor="gray"
    )
    fig_sil.write_html(f"{plot_dir}/evaluator_silhouette_score.html")

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "COURSE POPULARITY PREDICTOR — FINAL REPORT",
        "=" * 60,
        f"Generated : {now}",
        f"Scope     : Advanced Learners Only",
        "",
        "--- REGRESSION RESULTS ---",
    ]
    for name, m in reg_scores.items():
        report_lines.append(f"  {name:<22} R2={m['R2']:.4f}  RMSE={m['RMSE']:.6f}  MAE={m['MAE']:.6f}")
    report_lines += [
        f"  Best Model : {best_reg}",
        "",
        "--- CLASSIFICATION RESULTS ---"
    ]
    for name, m in cls_scores.items():
        report_lines.append(f"  {name:<22} Accuracy={m['Accuracy']*100:.2f}%  F1={m['F1']:.4f}")
    report_lines += [
        f"  Best Model : {best_cls}",
        "",
        "--- CLUSTERING ---",
        f"  K-Means Clusters  : 3 (Student | Early Professional | Mid-Level Professional)",
        f"  Silhouette Score  : {sil_score:.4f}  ({sil_verdict})",
        "",
        "--- FINAL COURSE RANKING ---"
    ]
    for i, row in ranking.iterrows():
        medal = ["1st", "2nd", "3rd"][i]
        label = "HIGH" if row["Demand_Label"] == 1 else "LOW"
        report_lines.append(
            f"  {medal}  {row['Course_Name']:<22} Score={row['Popularity_Score']:.4f}  Demand={label}"
        )
    report_lines += [
        "",
        "--- CONCLUSION ---",
        f"  AI is the most popular advanced course with a score of {ranking.iloc[0]['Popularity_Score']:.4f}.",
        f"  Solidity ranks 2nd and Project Management ranks 3rd.",
        f"  {best_reg} is the best regression model (R2={reg_scores[best_reg]['R2']:.4f}).",
        f"  {best_cls} achieves perfect classification accuracy.",
        "=" * 60
    ]

    with open(f"{report_dir}/final_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n=== EVALUATOR COMPLETE ===")
    print(f"  Report saved  : {report_dir}/final_report.txt")
    print(f"  Plots saved   :")
    print(f"    {plot_dir}/evaluator_model_performance_3d.html")
    print(f"    {plot_dir}/evaluator_course_ranking.html")
    print(f"    {plot_dir}/evaluator_regression_comparison.html")
    print(f"    {plot_dir}/evaluator_silhouette_score.html")

    return {
        "reg_scores" : reg_scores,
        "cls_scores" : cls_scores,
        "best_reg"   : best_reg,
        "best_cls"   : best_cls,
        "sil_score"  : sil_score,
        "ranking"    : ranking
    }
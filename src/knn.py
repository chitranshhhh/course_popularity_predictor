import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def run_knn(X_train, X_test, y_train, y_test,
            plot_dir="outputs/plots",
            model_dir="outputs/models",
            report_dir="outputs/reports"):

    # =========================================================================
    # 1. Find Best K (K=1 to 20)
    # =========================================================================
    k_values   = list(range(1, 21))
    accuracies = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        knn.fit(X_train, y_train)
        accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

    best_k   = k_values[np.argmax(accuracies)]
    best_acc = max(accuracies)

    print(f"\n=== KNN CLASSIFIER ===")
    print(f"  Best K        : {best_k}")
    print(f"  Best Accuracy : {best_acc * 100:.2f}%")

    # =========================================================================
    # 2. Train Final KNN with Best K
    # =========================================================================
    knn_best  = KNeighborsClassifier(n_neighbors=best_k, metric="euclidean")
    knn_best.fit(X_train, y_train)
    knn_preds = knn_best.predict(X_test)
    joblib.dump(knn_best, f"{model_dir}/knn.pkl")

    report = classification_report(y_test, knn_preds,
                                   target_names=["Low Demand","High Demand"])
    cm     = confusion_matrix(y_test, knn_preds)
    print(f"\n  Classification Report:\n{report}")

    # =========================================================================
    # PLOT 1 — 2D: K vs Accuracy line chart
    # =========================================================================
    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=k_values,
        y=[a * 100 for a in accuracies],
        mode="lines+markers",
        line=dict(color="#4C9BE8", width=3),
        marker=dict(
            size=[12 if k == best_k else 7 for k in k_values],
            color=["#FF4C4C" if k == best_k else "#4C9BE8" for k in k_values],
            line=dict(color="white", width=1)
        ),
        name="Accuracy",
        text=[f"K={k}<br>Acc={a*100:.2f}%" for k, a in zip(k_values, accuracies)],
        hoverinfo="text"
    ))
    fig_k.add_vline(x=best_k, line_dash="dash", line_color="lightgreen",
                   annotation_text=f"Best K={best_k} ({best_acc*100:.2f}%)",
                   annotation_font_color="lightgreen")
    fig_k.update_layout(
        title=dict(text="KNN — K vs Accuracy (2D)", font=dict(size=18)),
        xaxis=dict(title="K (Neighbours)", tickvals=k_values, gridcolor="gray"),
        yaxis=dict(title="Accuracy (%)", gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"), bgcolor="rgba(255,255,255,0.1)")
    )
    fig_k.write_html(f"{plot_dir}/knn_k_vs_accuracy.html")

    # =========================================================================
    # PLOT 2 — 2D: Confusion Matrix Heatmap
    # =========================================================================
    labels = ["Low Demand", "High Demand"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=18)
    ))
    fig_cm.update_layout(
        title=dict(text="KNN — Confusion Matrix (2D)", font=dict(size=18)),
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="Actual Label"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white")
    )
    fig_cm.write_html(f"{plot_dir}/knn_confusion_matrix.html")

    # =========================================================================
    # PLOT 3 — 3D: Decision Boundary in PCA Space
    # =========================================================================
    colors_map = {0: "#FF4C4C", 1: "#50FA7B"}
    labels_map = {0: "Low Demand",  1: "High Demand"}

    fig_3d = go.Figure()
    for cls in [0, 1]:
        mask = knn_preds == cls
        fig_3d.add_trace(go.Scatter3d(
            x=X_test[mask, 0],
            y=X_test[mask, 1],
            z=X_test[mask, 2],
            mode="markers",
            name=f"Predicted: {labels_map[cls]}",
            marker=dict(size=3, color=colors_map[cls], opacity=0.55)
        ))

    # Highlight misclassifications
    wrong_mask = knn_preds != y_test
    if wrong_mask.sum() > 0:
        fig_3d.add_trace(go.Scatter3d(
            x=X_test[wrong_mask, 0],
            y=X_test[wrong_mask, 1],
            z=X_test[wrong_mask, 2],
            mode="markers",
            name="Misclassified",
            marker=dict(size=7, color="white", symbol="x",
                        line=dict(color="red", width=2))
        ))

    fig_3d.update_layout(
        title=dict(
            text=f"KNN 3D — Predicted Classes in PCA Space (K={best_k})",
            font=dict(size=18)
        ),
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
    fig_3d.write_html(f"{plot_dir}/knn_3d_decision_boundary.html")

    # =========================================================================
    # Save report
    # =========================================================================
    report_text = (f"KNN CLASSIFICATION REPORT\n" + "="*40 + "\n\n"
                   f"Best K    : {best_k}\n"
                   f"Accuracy  : {best_acc*100:.2f}%\n\n"
                   f"{report}")
    with open(f"{report_dir}/knn_report.txt", "w") as f:
        f.write(report_text)

    print(f"\n  Model saved   : {model_dir}/knn.pkl")
    print(f"  Report saved  : {report_dir}/knn_report.txt")
    print(f"  Plots saved   :")
    print(f"    {plot_dir}/knn_k_vs_accuracy.html")
    print(f"    {plot_dir}/knn_confusion_matrix.html")
    print(f"    {plot_dir}/knn_3d_decision_boundary.html")

    return knn_best, knn_preds, {"Accuracy": best_acc, "Best_K": best_k,
                                  "Report": report, "CM": cm}
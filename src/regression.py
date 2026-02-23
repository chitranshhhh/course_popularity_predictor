import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb


def _metrics(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"\n  [{name}]")
    print(f"    R²   : {r2:.6f}  (1.0 = perfect | <0.5 = poor)")
    print(f"    RMSE : {rmse:.6f}  (lower is better)")
    print(f"    MAE  : {mae:.6f}  (lower is better)")
    return {"R2": r2, "RMSE": rmse, "MAE": mae}


def run_regression(X_train, X_test, y_train, y_test,
                   plot_dir="outputs/plots",
                   model_dir="outputs/models",
                   report_dir="outputs/reports"):

    results  = {}
    all_preds = {}

    # =========================================================================
    # 1. LINEAR REGRESSION — Baseline (will fail — proof of work)
    # =========================================================================
    print("\n=== LINEAR REGRESSION (Baseline) ===")
    lr      = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    results["Linear Regression"] = _metrics("Linear Regression", y_test, lr_pred)
    all_preds["Linear Regression"] = lr_pred

    print(f"    Unique predicted values : {len(np.unique(lr_pred.round(4)))}")
    print(f"    Unique actual values    : {len(np.unique(y_test))}")
    print(f"    ⚠ Linear Regression predicts {len(np.unique(lr_pred.round(4)))} unique")
    print(f"      values but target only has {len(np.unique(y_test))} — PROOF OF FAILURE")

    # =========================================================================
    # 2. RANDOM FOREST
    # =========================================================================
    print("\n=== RANDOM FOREST REGRESSOR ===")
    rf      = RandomForestRegressor(n_estimators=200, max_depth=10,
                                    random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["Random Forest"] = _metrics("Random Forest", y_test, rf_pred)
    all_preds["Random Forest"] = rf_pred
    joblib.dump(rf, f"{model_dir}/random_forest.pkl")

    # =========================================================================
    # 3. XGBOOST
    # =========================================================================
    print("\n=== XGBOOST REGRESSOR ===")
    xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=6,
                                  learning_rate=0.05, random_state=42,
                                  verbosity=0)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    results["XGBoost"] = _metrics("XGBoost", y_test, xgb_pred)
    all_preds["XGBoost"] = xgb_pred
    joblib.dump(xgb_model, f"{model_dir}/xgboost.pkl")

    # =========================================================================
    # PLOT 1 — 3D: Actual vs Predicted vs Residual (all 3 models)
    # =========================================================================
    colors = {"Linear Regression": "#FF4C4C",
              "Random Forest"    : "#4C9BE8",
              "XGBoost"          : "#50FA7B"}

    fig3d = go.Figure()
    for name, pred in all_preds.items():
        residuals = y_test - pred
        fig3d.add_trace(go.Scatter3d(
            x=y_test,
            y=pred,
            z=residuals,
            mode="markers",
            name=name,
            marker=dict(size=3, color=colors[name], opacity=0.5)
        ))

    # Perfect prediction line
    mn, mx = y_test.min(), y_test.max()
    fig3d.add_trace(go.Scatter3d(
        x=[mn, mx], y=[mn, mx], z=[0, 0],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="white", width=4, dash="dash")
    ))

    fig3d.update_layout(
        title=dict(
            text="3D: Actual vs Predicted vs Residual — All Models",
            font=dict(size=18)
        ),
        scene=dict(
            xaxis=dict(title="Actual Score"),
            yaxis=dict(title="Predicted Score"),
            zaxis=dict(title="Residual (Error)"),
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
        scene_camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
    )
    fig3d.write_html(f"{plot_dir}/regression_3d_actual_vs_predicted.html")

    # =========================================================================
    # PLOT 2 — 2D: Linear Regression Failure Proof
    # (scatter of actual vs predicted — shows linear can't snap to 3 values)
    # =========================================================================
    fig_fail = go.Figure()
    fig_fail.add_trace(go.Scatter(
        x=y_test, y=lr_pred,
        mode="markers",
        marker=dict(color="#FF4C4C", size=4, opacity=0.4),
        name="Linear Regression"
    ))
    fig_fail.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        name="Perfect Prediction",
        line=dict(color="white", width=2, dash="dash")
    ))
    fig_fail.update_layout(
        title=dict(
            text="Linear Regression FAILURE — Actual vs Predicted (2D)",
            font=dict(size=18)
        ),
        xaxis=dict(title="Actual Score (only 3 values)",
                   gridcolor="gray"),
        yaxis=dict(title="Predicted Score (4,926 values — wrong!)",
                   gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)")
    )
    fig_fail.write_html(f"{plot_dir}/regression_linear_failure_proof.html")

    # =========================================================================
    # PLOT 3 — 2D: Model Comparison Bar (R², RMSE, MAE)
    # =========================================================================
    model_names = list(results.keys())
    r2_vals     = [results[m]["R2"]   for m in model_names]
    rmse_vals   = [results[m]["RMSE"] for m in model_names]
    mae_vals    = [results[m]["MAE"]  for m in model_names]

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="R²",   x=model_names, y=r2_vals,
                             marker_color="#4C9BE8"))
    fig_cmp.add_trace(go.Bar(name="RMSE", x=model_names, y=rmse_vals,
                             marker_color="#FF4C4C"))
    fig_cmp.add_trace(go.Bar(name="MAE",  x=model_names, y=mae_vals,
                             marker_color="#50FA7B"))
    fig_cmp.update_layout(
        barmode="group",
        title=dict(text="Regression Model Comparison — R², RMSE, MAE (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Model"),
        yaxis=dict(title="Score", gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)"),
        xaxis_gridcolor="gray"
    )
    fig_cmp.write_html(f"{plot_dir}/regression_model_comparison.html")

    # =========================================================================
    # PLOT 4 — 2D: Feature Importance (Random Forest)
    # =========================================================================
    feat_names  = [f"PC{i+1}" for i in range(X_train.shape[1])]
    importances = rf.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig_fi = go.Figure(go.Bar(
        x=importances[sorted_idx],
        y=[feat_names[i] for i in sorted_idx],
        orientation="h",
        marker=dict(
            color=importances[sorted_idx],
            colorscale="Blues",
            showscale=True
        )
    ))
    fig_fi.update_layout(
        title=dict(text="Random Forest — Feature Importance (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Importance Score", gridcolor="gray"),
        yaxis=dict(title="PCA Component"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        xaxis_gridcolor="gray"
    )
    fig_fi.write_html(f"{plot_dir}/regression_feature_importance.html")

    # =========================================================================
    # Save text report
    # =========================================================================
    report = "REGRESSION MODEL REPORT\n" + "="*40 + "\n\n"
    report += "NOTE: Linear Regression included as baseline proof of failure.\n"
    report += f"Target has only 3 unique values — Linear Regression is NOT suitable.\n\n"
    for name, m in results.items():
        report += f"Model : {name}\n"
        report += f"  R²   : {m['R2']:.6f}\n"
        report += f"  RMSE : {m['RMSE']:.6f}\n"
        report += f"  MAE  : {m['MAE']:.6f}\n\n"
    with open(f"{report_dir}/regression_report.txt", "w") as f:
        f.write(report)

    print("\n=== REGRESSION COMPLETE ===")
    print(f"  Models saved  : {model_dir}/random_forest.pkl")
    print(f"                  {model_dir}/xgboost.pkl")
    print(f"  Report saved  : {report_dir}/regression_report.txt")
    print(f"  Plots saved   :")
    print(f"    {plot_dir}/regression_3d_actual_vs_predicted.html")
    print(f"    {plot_dir}/regression_linear_failure_proof.html")
    print(f"    {plot_dir}/regression_model_comparison.html")
    print(f"    {plot_dir}/regression_feature_importance.html")

    return rf, xgb_model, results, rf_pred
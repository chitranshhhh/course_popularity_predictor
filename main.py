import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader    import load_and_sort_data, filter_advanced_learners
from src.pca_reduction  import prepare_features, apply_pca
from src.pca_check      import check_pca_columns
from src.clustering     import run_clustering
from src.popularity     import compute_popularity
from src.preprocessing  import preprocess
from src.regression     import run_regression
from src.knn            import run_knn
from src.ann            import run_ann
from src.evaluator      import run_evaluator


def main():
    DATA_PATH  = "data/data_100000_records_daily.csv"
    PLOTS_DIR  = "outputs/plots"
    MODELS_DIR = "outputs/models"
    REPORT_DIR = "outputs/reports"

    # Step 1 — Load & Sort
    df     = load_and_sort_data(DATA_PATH)
    df_adv = filter_advanced_learners(df)

    # Step 2 — PCA Column Check
    check_pca_columns(df_adv)

    # Step 3 — Prepare Features + Apply PCA
    X_scaled, scaler_init, feature_cols, le_course, le_exp, le_device = prepare_features(df_adv)
    X_pca, pca_init = apply_pca(
        X_scaled, variance_threshold=0.95,
        plot_path=f"{PLOTS_DIR}/pca_3d_interactive.html"
    )

    # Step 4 — Clustering
    kmeans, cluster_labels, cluster_name_map = run_clustering(
        X_pca, df_adv, plot_dir=PLOTS_DIR, n_clusters=3
    )

    # Step 5 — Popularity Score + Demand Label
    agg, df_merged = compute_popularity(df_adv, plot_dir=PLOTS_DIR)

    # Step 6 — Preprocessing + Train/Test Split
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test,
     scaler, pca, feat_cols,
     le_c, le_e, le_d) = preprocess(df_merged, plot_dir=PLOTS_DIR)

    # Step 7 — Regression
    rf_model, xgb_model, reg_results, rf_preds = run_regression(
        X_train, X_test, y_reg_train, y_reg_test,
        plot_dir=PLOTS_DIR, model_dir=MODELS_DIR, report_dir=REPORT_DIR
    )

    # Step 8 — KNN
    knn_model, knn_preds, knn_results = run_knn(
        X_train, X_test, y_cls_train, y_cls_test,
        plot_dir=PLOTS_DIR, model_dir=MODELS_DIR, report_dir=REPORT_DIR
    )

    # Step 9 — ANN
    ann_model, ann_preds, ann_results = run_ann(
        X_train, X_test, y_cls_train, y_cls_test,
        plot_dir=PLOTS_DIR, model_dir=MODELS_DIR, report_dir=REPORT_DIR,
        epochs=50, batch_size=256
    )

    # Step 10 — Final Evaluation
    # Get linear regression predictions for evaluator
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_reg_train)
    lr_preds = lr.predict(X_test)

    eval_results = run_evaluator(
        y_reg_test  = y_reg_test,
        lr_pred     = lr_preds,
        rf_pred     = rf_preds,
        xgb_pred    = xgb_model.predict(X_test),
        y_cls_test  = y_cls_test,
        knn_pred    = knn_preds,
        ann_pred    = ann_preds,
        X_pca       = X_pca,
        cluster_labels = cluster_labels,
        agg         = agg,
        plot_dir    = PLOTS_DIR,
        report_dir  = REPORT_DIR
    )

    print("\n" + "=" * 55)
    print("  FINAL PIPELINE SUMMARY")
    print("=" * 55)
    print(f"  Total records         : {df.shape[0]:,}")
    print(f"  Advanced records      : {df_adv.shape[0]:,}")
    print(f"  PCA components        : {pca.n_components_}")
    print(f"  Train / Test          : {X_train.shape[0]:,} / {X_test.shape[0]:,}")
    print(f"  Silhouette Score      : {eval_results['sil_score']:.4f}")
    print(f"\n  Regression Results:")
    for model, metrics in eval_results["reg_scores"].items():
        tag = " <- BEST" if model == eval_results["best_reg"] else ""
        print(f"    {model:<22} R2={metrics['R2']:.4f}{tag}")
    print(f"\n  Classification Results:")
    for model, metrics in eval_results["cls_scores"].items():
        tag = " <- BEST" if model == eval_results["best_cls"] else ""
        print(f"    {model:<22} Accuracy={metrics['Accuracy']*100:.2f}%{tag}")
    print(f"\n  Course Rankings:")
    for i, row in eval_results["ranking"].iterrows():
        medal = ["1st", "2nd", "3rd"][i]
        print(f"    {medal}  {row['Course_Name']:<22} Score={row['Popularity_Score']:.4f}")
    print(f"\n  All outputs saved to  : outputs/")
    print("=" * 55)


if __name__ == "__main__":
    main()
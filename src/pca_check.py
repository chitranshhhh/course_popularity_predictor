import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA


def check_pca_columns(df_adv, variance_threshold=0.95):

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

    pca_full = PCA(n_components=None, random_state=42)
    pca_full.fit(X_scaled)

    pca = PCA(n_components=variance_threshold, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("=" * 60)
    print("  ORIGINAL COLUMNS (Before PCA)")
    print("=" * 60)
    for i, col in enumerate(feature_cols):
        print(f"  [{i+1:02d}] {col}")

    print(f"\n  Total original features : {len(feature_cols)}")

    print("\n" + "=" * 60)
    print("  NEW COLUMNS (After PCA)")
    print("=" * 60)
    new_cols = [f"PC{i+1}" for i in range(pca.n_components_)]
    for i, col in enumerate(new_cols):
        var = pca.explained_variance_ratio_[i] * 100
        print(f"  [{i+1:02d}] {col}   →   variance explained: {var:.4f}%")

    print(f"\n  Total PCA components    : {pca.n_components_}")
    print(f"  Total variance retained : {sum(pca.explained_variance_ratio_)*100:.4f}%")

    print("\n" + "=" * 60)
    print("  DROPPED COMPONENTS")
    print("=" * 60)
    n_dropped = len(feature_cols) - pca.n_components_
    if n_dropped == 0:
        print("  None dropped.")
    else:
        for i in range(pca.n_components_, len(feature_cols)):
            var     = pca_full.explained_variance_ratio_[i] * 100
            pc      = pca_full.components_[i]
            top_idx = np.argsort(np.abs(pc))[::-1][:2]
            top     = ", ".join([f"{feature_cols[j]} ({pc[j]:.3f})"
                                 for j in top_idx])
            print(f"  PC{i+1:02d}  variance: {var:.4f}%   top features: {top}")

    print(f"\n  Total dropped           : {n_dropped}")
    print(f"  Total variance lost     : {sum(pca_full.explained_variance_ratio_[pca.n_components_:])*100:.4f}%")

    print("\n" + "=" * 60)
    print("  COMPONENT LOADINGS (What each PC is made of)")
    print("=" * 60)
    loading_df = pd.DataFrame(
        pca.components_,
        columns=feature_cols,
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    ).round(4)
    print(loading_df.to_string())

    print("\n" + "=" * 60)
    print("  X_pca SHAPE CONFIRMATION")
    print("=" * 60)
    print(f"  Original X shape : {X_scaled.shape}  → (rows, {len(feature_cols)} features)")
    print(f"  PCA X shape      : {X_pca.shape}  → (rows, {pca.n_components_} components)")

    return X_pca, pca, new_cols
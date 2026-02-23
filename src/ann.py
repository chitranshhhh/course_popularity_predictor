import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def build_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),

        layers.Dense(16, activation="relu"),

        layers.Dense(1, activation="sigmoid")
    ], name="ANN_Demand_Classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_ann(X_train, X_test, y_train, y_test,
            plot_dir="outputs/plots",
            model_dir="outputs/models",
            report_dir="outputs/reports",
            epochs=50,
            batch_size=256):

    input_dim = X_train.shape[1]
    model     = build_model(input_dim)

    print("\n=== ANN ARCHITECTURE ===")
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=5,
        restore_best_weights=True, verbose=1
    )
    lr_reduce = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\n=== TRAINING ===")
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, lr_reduce],
        verbose=1
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_test).flatten()
    y_pred      = (y_pred_prob >= 0.5).astype(int)
    acc         = accuracy_score(y_test, y_pred)
    report      = classification_report(y_test, y_pred,
                                        target_names=["Low Demand", "High Demand"])
    cm          = confusion_matrix(y_test, y_pred)

    print(f"\n=== ANN RESULTS ===")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"\n{report}")

    model.save(f"{model_dir}/ann_model.keras")

    # ── PLOT 1 — 2D: Loss curve ────────────────────────────────────────────────
    epochs_ran = list(range(1, len(history.history["loss"]) + 1))

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=epochs_ran, y=history.history["loss"],
        mode="lines", name="Train Loss",
        line=dict(color="#4C9BE8", width=2)
    ))
    fig_loss.add_trace(go.Scatter(
        x=epochs_ran, y=history.history["val_loss"],
        mode="lines", name="Val Loss",
        line=dict(color="#FF4C4C", width=2, dash="dash")
    ))
    fig_loss.update_layout(
        title=dict(text="ANN - Training vs Validation Loss (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Epoch", gridcolor="gray"),
        yaxis=dict(title="Loss", gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)")
    )
    fig_loss.write_html(f"{plot_dir}/ann_training_loss.html")

    # ── PLOT 2 — 2D: Accuracy curve ───────────────────────────────────────────
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(
        x=epochs_ran, y=[a*100 for a in history.history["accuracy"]],
        mode="lines", name="Train Accuracy",
        line=dict(color="#50FA7B", width=2)
    ))
    fig_acc.add_trace(go.Scatter(
        x=epochs_ran, y=[a*100 for a in history.history["val_accuracy"]],
        mode="lines", name="Val Accuracy",
        line=dict(color="#FF9F43", width=2, dash="dash")
    ))
    fig_acc.update_layout(
        title=dict(text="ANN - Training vs Validation Accuracy (2D)",
                   font=dict(size=18)),
        xaxis=dict(title="Epoch", gridcolor="gray"),
        yaxis=dict(title="Accuracy (%)", gridcolor="gray"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white"),
        legend=dict(font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)")
    )
    fig_acc.write_html(f"{plot_dir}/ann_training_accuracy.html")

    # ── PLOT 3 — 2D: Confusion Matrix ─────────────────────────────────────────
    labels = ["Low Demand", "High Demand"]
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=18)
    ))
    fig_cm.update_layout(
        title=dict(text="ANN - Confusion Matrix (2D)", font=dict(size=18)),
        xaxis=dict(title="Predicted Label"),
        yaxis=dict(title="Actual Label"),
        paper_bgcolor="rgb(15,15,40)",
        plot_bgcolor="rgb(25,25,55)",
        font=dict(color="white")
    )
    fig_cm.write_html(f"{plot_dir}/ann_confusion_matrix.html")

    # ── PLOT 4 — 3D: Predicted classes in PCA space ───────────────────────────
    colors_map = {0: "#FF4C4C", 1: "#50FA7B"}
    labels_map = {0: "Low Demand", 1: "High Demand"}

    fig_3d = go.Figure()
    for cls in [0, 1]:
        mask = y_pred == cls
        fig_3d.add_trace(go.Scatter3d(
            x=X_test[mask, 0],
            y=X_test[mask, 1],
            z=X_test[mask, 2],
            mode="markers",
            name=f"Predicted: {labels_map[cls]}",
            marker=dict(size=3, color=colors_map[cls], opacity=0.55)
        ))

    wrong_mask = y_pred != y_test
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
        title=dict(text="ANN 3D - Predicted Classes in PCA Space",
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
    fig_3d.write_html(f"{plot_dir}/ann_3d_predicted_classes.html")

    # ── Save report (utf-8 to handle all characters) ──────────────────────────
    report_text = (
        "ANN CLASSIFICATION REPORT\n" + "="*40 + "\n\n"
        f"Architecture  : 64 -> 32 -> 16 -> 1 (Sigmoid)\n"
        f"Optimizer     : Adam (lr=0.001)\n"
        f"Epochs run    : {len(epochs_ran)}\n"
        f"Test Accuracy : {acc*100:.2f}%\n\n"
        f"{report}"
    )
    with open(f"{report_dir}/ann_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\n  Model saved   : {model_dir}/ann_model.keras")
    print(f"  Report saved  : {report_dir}/ann_report.txt")
    print(f"  Plots saved   :")
    print(f"    {plot_dir}/ann_training_loss.html")
    print(f"    {plot_dir}/ann_training_accuracy.html")
    print(f"    {plot_dir}/ann_confusion_matrix.html")
    print(f"    {plot_dir}/ann_3d_predicted_classes.html")

    return model, y_pred, {
        "Accuracy"  : acc,
        "Report"    : report,
        "CM"        : cm,
        "History"   : history.history,
        "Epochs_ran": len(epochs_ran)
    }
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, classification_report


def create_model_leaderboard(results):
    rows = []

    for model_name, values in results.items():
        rows.append({
            "Model": model_name,
            "MAE": round(values.get("mae", 0), 4),
            "RMSE": round(values.get("rmse", 0), 4),
            "R2 Score": round(values.get("r2", 0), 4),
            "Training Time": round(values.get("train_time", 0), 2)
        })

    return pd.DataFrame(rows).sort_values("R2 Score", ascending=False)


def create_residual_plot(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(residuals))),
        y=residuals,
        mode="markers",
        marker=dict(color="#00d4ff", size=5),
        name="Residuals"
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#ff6b35")

    fig.update_layout(
        title="Residual Error Analysis",
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff"),
        xaxis_title="Sample Index",
        yaxis_title="Residual Error"
    )

    return fig


def create_prediction_distribution(y_true, y_pred):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=y_true,
        name="Actual",
        opacity=0.65,
        marker_color="#00d4ff"
    ))

    fig.add_trace(go.Histogram(
        x=y_pred,
        name="Predicted",
        opacity=0.65,
        marker_color="#ff6b35"
    ))

    fig.update_layout(
        title="Actual vs Predicted Distribution",
        barmode="overlay",
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff")
    )

    return fig


def create_confusion_matrix_from_severity(df):
    """
    Rule-based actual/predicted severity comparison.
    Works even if classification model is not present.
    """

    actual = df["severity"].astype(str).values[-300:]

    kp = df["kp_index"].values[-300:]
    pred = []

    for v in kp:
        if v < 3:
            pred.append("Low")
        elif v < 5:
            pred.append("Moderate")
        elif v < 7:
            pred.append("High")
        else:
            pred.append("Extreme")

    labels = ["Low", "Moderate", "High", "Extreme"]
    cm = confusion_matrix(actual, pred, labels=labels)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Plasma",
        text=cm,
        texttemplate="%{text}",
        textfont=dict(color="white")
    ))

    fig.update_layout(
        title="Severity Classification Confusion Matrix",
        xaxis_title="Predicted Severity",
        yaxis_title="Actual Severity",
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff")
    )

    report = classification_report(actual, pred, labels=labels, zero_division=0)

    return fig, report
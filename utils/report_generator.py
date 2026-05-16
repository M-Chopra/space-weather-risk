"""
=============================================================================
  SPACE WEATHER PREDICTOR — Downloadable Report Generator
=============================================================================
Generates a PDF or CSV summary report of current predictions and alerts.
"""

import os
import csv
import io
import logging
from datetime import datetime
from typing import List
logger = logging.getLogger(__name__)


def generate_csv_report(predictions: dict, alerts: list) -> bytes:
    """
    Generate a downloadable CSV report as bytes.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["SPACE WEATHER PREDICTION REPORT"])
    writer.writerow(["Generated UTC", datetime.utcnow().isoformat()])
    writer.writerow([])

    writer.writerow(["=== PREDICTIONS ==="])
    writer.writerow(["Metric", "Value", "Unit"])
    for k, v in predictions.items():
        writer.writerow([k, round(float(v), 3) if isinstance(v, (int, float)) else v, ""])

    writer.writerow([])
    writer.writerow(["=== ACTIVE ALERTS ==="])
    writer.writerow(["Level", "Metric", "Value", "Threshold", "Message"])
    for alert in alerts:
        a = alert if isinstance(alert, dict) else alert.to_dict()
        writer.writerow([a.get("level"), a.get("metric"), a.get("value"),
                         a.get("threshold"), a.get("message")])

    return output.getvalue().encode("utf-8")


def generate_txt_report(predictions: dict, alerts: list,
                         model_scores: dict = None) -> str:
    """
    Plain-text report for display / download.
    """
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=" * 65,
        "  🛰  SPACE WEATHER PREDICTION REPORT",
        f"  Generated: {now}",
        "=" * 65,
        "",
        "── CURRENT PREDICTIONS ─────────────────────────────────────",
    ]

    for k, v in predictions.items():
        label = k.replace("_", " ").title()
        val   = f"{v:.2f}" if isinstance(v, float) else str(v)
        lines.append(f"  {label:<40} {val}")

    lines += ["", "── ACTIVE ALERTS ────────────────────────────────────────────"]
    if not alerts:
        lines.append("  ✅  No alerts — space weather is quiet.")
    else:
        for alert in alerts:
            a = alert if isinstance(alert, dict) else alert.to_dict()
            lines.append(f"  [{a['level']:8s}]  {a['message']}  (value={a['value']})")

    if model_scores:
        lines += ["", "── MODEL PERFORMANCE ────────────────────────────────────────"]
        lines.append(f"  {'Model':<22} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
        lines.append("  " + "-" * 45)
        for name, sc in model_scores.items():
            if isinstance(sc, dict) and "mae" in sc:
                lines.append(f"  {name:<22} {sc['mae']:>7.3f} {sc['rmse']:>7.3f} {sc['r2']:>7.4f}")

    lines += ["", "=" * 65,
              "  Data: NOAA SWPC / NASA DONKI / Synthetic ML Dataset",
              "  Model: XGBoost + Random Forest Ensemble",
              "=" * 65]

    return "\n".join(lines)

"""
=============================================================================
  SPACE WEATHER PREDICTOR — Visualization Utilities
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os

VISUAL_DIR = "visuals"
os.makedirs(VISUAL_DIR, exist_ok=True)

_BG   = "#0d1117"
_C1   = "#00d4ff"   # cyan
_C2   = "#ff6b35"   # orange
_C3   = "#39ff14"   # neon green
_C4   = "#bf5fff"   # purple
_C5   = "#ffd700"   # gold

plt.rcParams.update({
    "figure.facecolor"  : _BG,
    "axes.facecolor"    : _BG,
    "axes.edgecolor"    : "#2a2a3e",
    "axes.labelcolor"   : "white",
    "xtick.color"       : "#aaa",
    "ytick.color"       : "#aaa",
    "text.color"        : "white",
    "grid.color"        : "#1f1f2e",
    "grid.linestyle"    : "--",
    "grid.alpha"        : 0.5,
    "legend.facecolor"  : "#1a1a2e",
    "legend.edgecolor"  : "#333",
    "legend.labelcolor" : "white",
    "font.family"       : "monospace",
})


def plot_solar_activity_overview(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Multi-panel overview of solar activity indicators."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True, facecolor=_BG)

    pairs = [
        ("sunspot_number",    "Sunspot Number (SSN)", _C1),
        ("f107_solar_flux",   "F10.7 Solar Flux (sfu)", _C2),
        ("kp_index",          "Kp Index", _C3),
        ("solar_wind_speed",  "Solar Wind Speed (km/s)", _C4),
    ]

    for ax, (col, label, color) in zip(axes, pairs):
        ax.plot(df["timestamp"], df[col], color=color, lw=0.8, alpha=0.9)
        ax.fill_between(df["timestamp"], df[col], alpha=0.15, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True)
        # Storm threshold lines
        if col == "kp_index":
            ax.axhline(5, color=_C5, lw=1, linestyle=":", alpha=0.7, label="Storm threshold")
            ax.axhline(7, color=_C2, lw=1, linestyle=":", alpha=0.7, label="Severe threshold")
            ax.legend(fontsize=8)

    axes[-1].set_xlabel("Date")
    plt.suptitle("☀  Solar Activity Overview", fontsize=16, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(VISUAL_DIR, "solar_activity_overview.png"),
                    dpi=150, bbox_inches="tight", facecolor=_BG)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Correlation matrix of key space-weather features."""
    cols = ["sunspot_number", "f107_solar_flux", "solar_wind_speed",
            "imf_bz", "kp_index", "dst_index",
            "satellite_disruption_risk", "gps_blackout_prob",
            "flare_intensity", "proton_flux"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "space", ["#002244", "#0d1117", "#00d4ff", "#39ff14"])

    fig, ax = plt.subplots(figsize=(12, 10), facecolor=_BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, annot=True,
                fmt=".2f", linewidths=0.5, linecolor="#1a1a2e",
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=14, pad=15)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(VISUAL_DIR, "correlation_heatmap.png"),
                    dpi=150, bbox_inches="tight", facecolor=_BG)
    return fig


def plot_risk_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Distribution plots for risk scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=_BG)

    for ax, col, color, title in [
        (axes[0], "satellite_disruption_risk", _C2, "Satellite Disruption Risk"),
        (axes[1], "gps_blackout_prob",         _C3, "GPS Blackout Probability"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor="none", density=True)
        # KDE overlay
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(vals)
        xs  = np.linspace(vals.min(), vals.max(), 200)
        ax.plot(xs, kde(xs), color="white", lw=1.5)
        ax.set_title(title)
        ax.set_xlabel("Risk Score (0–100)")
        ax.set_ylabel("Density")
        ax.grid(True)

    plt.suptitle("Risk Score Distributions", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(VISUAL_DIR, "risk_distributions.png"),
                    dpi=150, bbox_inches="tight", facecolor=_BG)
    return fig


def plot_storm_events(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Highlight geomagnetic storm periods on Kp timeline."""
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=_BG)

    ax.plot(df["timestamp"], df["kp_index"], color=_C1, lw=0.7, alpha=0.8)
    storm_mask = df["kp_index"] >= 5
    ax.fill_between(df["timestamp"], df["kp_index"],
                    where=storm_mask, alpha=0.4, color=_C2, label="Storm (Kp≥5)")
    ax.axhline(5, color=_C5, lw=1, linestyle="--", alpha=0.7)
    ax.axhline(7, color=_C2, lw=1, linestyle="--", alpha=0.7)
    ax.set_title("Geomagnetic Storm Events (Kp Index)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Kp Index")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(VISUAL_DIR, "storm_events.png"),
                    dpi=150, bbox_inches="tight", facecolor=_BG)
    return fig


def plot_severity_pie(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Pie chart of severity distribution."""
    counts = df["severity"].value_counts()
    colors = {"Low": _C3, "Moderate": _C5, "High": _C2, "Extreme": "#ff0040"}
    clrs   = [colors.get(s, "#aaa") for s in counts.index]

    fig, ax = plt.subplots(figsize=(7, 7), facecolor=_BG)
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=clrs, startangle=140, wedgeprops={"edgecolor": _BG, "linewidth": 2})
    for t in texts + autotexts:
        t.set_color("white")
    ax.set_title("Space Weather Severity Distribution", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(VISUAL_DIR, "severity_pie.png"),
                    dpi=150, bbox_inches="tight", facecolor=_BG)
    return fig


def generate_all_visuals(df: pd.DataFrame):
    """Run all visualization functions."""
    plot_solar_activity_overview(df)
    plot_correlation_heatmap(df)
    plot_risk_distribution(df)
    plot_storm_events(df)
    plot_severity_pie(df)
    print(f"✅  All visuals saved to /{VISUAL_DIR}/")

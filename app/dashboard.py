"""
=============================================================================
  SPACE WEATHER PREDICTOR — Streamlit Dashboard
  Run: streamlit run app/dashboard.py
=============================================================================
"""

import os
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils.data_collector  import load_or_generate_dataset, fetch_live_noaa_data
from utils.preprocessor    import preprocess, get_X_y
from utils.alert_system    import generate_alerts, overall_severity, risk_score
from utils.report_generator import generate_csv_report, generate_txt_report
from models.trainer        import train_all_models, load_best_model
from models.lstm_forecaster import forecast_next_48h
logging.basicConfig(level=logging.WARNING)
from api.live_space_weather import (
    get_live_kp,
    get_solar_wind,
    get_noaa_alerts
)

from ai_insights.insight_generator import generate_ai_insight

from forecasting.simple_forecast import forecast_next_48_hours

from maps.global_risk_map import create_global_risk_map
from tracking.satellite_tracker import create_satellite_tracking_map
from visuals.earth_sun_cme import create_earth_sun_cme_visual
from analytics.ml_analytics import (
    create_model_leaderboard,
    create_residual_plot,
    create_prediction_distribution,
    create_confusion_matrix_from_severity
)
from replay.historical_storms import (
    HISTORICAL_EVENTS,
    create_historical_replay_chart
)
# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Space Weather AI · Mission Control",
    page_icon  = "🛸",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

  /* Global */
  html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #050a14 !important;
    color: #c0d8ff;
  }
  .stApp { background: #050a14; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1f 0%, #0d1a2e 100%);
    border-right: 1px solid #0a2540;
  }

  /* Headers */
  h1, h2, h3 { font-family: 'Orbitron', monospace !important; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0a1628 0%, #0d2040 100%);
    border: 1px solid #1a3a5c;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 0 15px rgba(0,212,255,0.08);
  }
  div[data-testid="metric-container"] label {
    color: #5a8fbf !important;
    font-size: 0.75rem !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.1em;
  }
  div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #00d4ff !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.4rem !important;
  }

  /* Alert boxes */
  .alert-extreme  { background:#1a0010; border-left:4px solid #ff0040; padding:10px 16px; border-radius:4px; margin:6px 0; }
  .alert-critical { background:#1a0a00; border-left:4px solid #ff6b35; padding:10px 16px; border-radius:4px; margin:6px 0; }
  .alert-warning  { background:#1a1500; border-left:4px solid #ffd700; padding:10px 16px; border-radius:4px; margin:6px 0; }
  .alert-info     { background:#001a1a; border-left:4px solid #00d4ff; padding:10px 16px; border-radius:4px; margin:6px 0; }
  .alert-ok       { background:#001a0a; border-left:4px solid #39ff14; padding:10px 16px; border-radius:4px; margin:6px 0; }

  /* Glowing title */
  .glow-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff, #bf5fff, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    animation: shimmer 3s linear infinite;
    letter-spacing: 0.05em;
  }
  @keyframes shimmer { to { background-position: 200% center; } }

  .subtitle {
    text-align: center;
    color: #4a7fa8;
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    margin-top: -8px;
  }

  /* Section divider */
  .section-header {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem;
    color: #00d4ff;
    letter-spacing: 0.2em;
    border-bottom: 1px solid #0a2540;
    padding-bottom: 6px;
    margin: 24px 0 12px 0;
    text-transform: uppercase;
  }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { background: #08111e; border-radius: 8px; }
  .stTabs [data-baseweb="tab"] { color: #4a7fa8; font-family: 'Orbitron', monospace; font-size: 0.75rem; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom: 2px solid #00d4ff; }

  /* Buttons */
  .stButton>button {
    background: linear-gradient(135deg, #0a2540, #0d3060);
    border: 1px solid #00d4ff;
    color: #00d4ff;
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    border-radius: 4px;
    transition: all 0.2s;
  }
  .stButton>button:hover { background: #00d4ff; color: #050a14; }

  /* Spinner */
  .stSpinner > div { border-top-color: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

TARGETS = {
    "satellite_disruption_risk": "Satellite Disruption Risk",
    "gps_blackout_prob"        : "GPS Blackout Probability",
    "kp_index"                 : "Kp Geomagnetic Index",
}

BG      = "#050a14"
PAPER   = "#08111e"
C1      = "#00d4ff"
C2      = "#ff6b35"
C3      = "#39ff14"
C4      = "#bf5fff"
C5      = "#ffd700"

PLOTLY_LAYOUT = dict(
    paper_bgcolor = PAPER,
    plot_bgcolor  = BG,
    font          = dict(family="Share Tech Mono", color="#c0d8ff"),
    xaxis         = dict(gridcolor="#0a2540", zerolinecolor="#0a2540"),
    yaxis         = dict(gridcolor="#0a2540", zerolinecolor="#0a2540"),
    legend        = dict(bgcolor="#08111e", bordercolor="#1a3a5c", borderwidth=1),
    margin        = dict(l=40, r=40, t=50, b=40),
)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHED DATA / MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    return load_or_generate_dataset(n_years=5, save_path="data/space_weather.csv")

@st.cache_data(show_spinner=False)
def get_processed(_raw_df):
    return preprocess(_raw_df)

@st.cache_resource(show_spinner=False)
def train_models_cached(target):
    raw_df = load_data()
    df_proc, scaler, encoders = get_processed(raw_df)
    X, y = get_X_y(df_proc, target)
    return train_all_models(X, y, target_name=target)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def gauge_chart(value: float, title: str, max_val: float = 100,
                color_stops=None) -> go.Figure:
    """Plotly gauge / speedometer."""
    if color_stops is None:
        color_stops = [
            [0,   "#39ff14"],
            [0.4, "#ffd700"],
            [0.7, "#ff6b35"],
            [1.0, "#ff0040"],
        ]
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = value,
        title = {"text": title, "font": {"family": "Orbitron", "size": 13, "color": "#c0d8ff"}},
        number= {"font": {"family": "Orbitron", "size": 24, "color": C1}, "suffix": ""},
        gauge = {
            "axis"     : {"range": [0, max_val], "tickcolor": "#4a7fa8",
                          "tickfont": {"family": "Share Tech Mono", "size": 10}},
            "bar"      : {"color": C1, "thickness": 0.25},
            "bgcolor"  : "#0a1628",
            "bordercolor": "#1a3a5c",
            "steps": [{"range": [0, max_val * s[0]], "color": s[1]}for s in color_stops],
            "threshold": {"line": {"color": "#ff0040", "width": 2},
                          "value": max_val * 0.8},
        },
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=220)
    return fig


def severity_badge(level: str) -> str:
    colors = {
        "Low"     : ("#001a0a", "#39ff14"),
        "Moderate": ("#1a1500", "#ffd700"),
        "High"    : ("#1a0a00", "#ff6b35"),
        "Extreme" : ("#1a0010", "#ff0040"),
    }
    bg, fg = colors.get(level, ("#0a0a0a", "#aaa"))
    return (f'<span style="background:{bg};color:{fg};border:1px solid {fg}; '
            f'padding:4px 12px;border-radius:4px;font-family:Orbitron,monospace;'
            f'font-size:0.85rem;letter-spacing:0.1em;">{level.upper()}</span>')


def render_alerts(alerts):
    if not alerts:
        st.markdown('<div class="alert-ok">✅ &nbsp;<b>All Clear</b> — Space weather is quiet.</div>',
                    unsafe_allow_html=True)
        return
    for a in alerts:
        d = a if isinstance(a, dict) else a.to_dict()
        level = d["level"].lower()
        icon  = {"extreme":"🔴", "critical":"🟠", "warning":"🟡", "info":"🔵"}.get(level,"⚪")
        st.markdown(
            f'<div class="alert-{level}">{icon} &nbsp;'
            f'<b>[{d["level"]}]</b> {d["message"]} '
            f'<span style="opacity:.6">({d["metric"]} = {d["value"]})</span></div>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="glow-title" style="font-size:1.1rem;margin-bottom:16px;">⚡ MISSION CTRL</div>', unsafe_allow_html=True)

    st.markdown("### 🕹 Controls")
    n_days_display = st.slider("Days to display", 30, 730, 180, step=30)
    target_sel     = st.selectbox("Prediction Target",
                                   list(TARGETS.keys()),
                                   format_func=lambda k: TARGETS[k])
    auto_refresh   = st.toggle("Auto-Refresh (30s)", value=False)

    st.divider()
    st.markdown("### 🧠 Model Settings")
    retrain = st.button("🔄 Retrain Models", use_container_width=True)
    if retrain:
        st.cache_resource.clear()
        st.success("Cache cleared — models will retrain.")

    st.divider()
    st.markdown("### 📡 Live Data")
    fetch_live = st.button("📡 Fetch NOAA Live", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#2a4a6a;line-height:1.6;">
    📦 Data: NOAA SWPC · NASA DONKI<br>
    🤖 Models: RF · XGBoost · LSTM<br>
    🛸 v1.0 · Space-Weather-ML
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="glow-title">🛸 SPACE WEATHER INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">SATELLITE DISRUPTION · GPS RISK · GEOMAGNETIC STORM PREDICTION · REAL-TIME AI ANALYSIS</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("🌌 Loading space weather dataset…"):
    raw_df   = load_data()
    df_proc, scaler, encoders = get_processed(raw_df)

# Latest observation
latest = raw_df.iloc[-1]
ts     = pd.Timestamp(latest["timestamp"])
now_str = ts.strftime("%Y-%m-%d  %H:%M UTC")

# ── Compute current metrics ────────────────────────────────────────────────────
current_metrics = {
    "kp_index"                 : float(latest["kp_index"]),
    "satellite_disruption_risk": float(latest["satellite_disruption_risk"]),
    "gps_blackout_prob"        : float(latest["gps_blackout_prob"]),
    "solar_wind_speed"         : float(latest["solar_wind_speed"]),
    "flare_intensity"          : float(latest["flare_intensity"]),
}
alerts   = generate_alerts(current_metrics)
severity = overall_severity(alerts)
comp_risk = risk_score(
    kp         = current_metrics["kp_index"],
    sat_risk   = current_metrics["satellite_disruption_risk"],
    gps_prob   = current_metrics["gps_blackout_prob"],
    wind_speed = current_metrics["solar_wind_speed"],
    flare_int  = current_metrics["flare_intensity"],
)

# ── NOAA live fetch ────────────────────────────────────────────────────────────
live_df = None

if fetch_live:
    with st.spinner("🛰 Connecting to NOAA SWPC..."):

        live_data = get_live_kp()

        if live_data["success"]:

            current_metrics["kp_index"] = live_data["kp_index"]

            st.success(
                f"✅ Live NOAA Connected | Kp: {live_data['kp_index']:.1f}"
            )

        else:
            st.warning("⚠ NOAA API temporarily unavailable.")

# ══════════════════════════════════════════════════════════════════════════════
#  TOP STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(f"**🕐 Last observation:** `{now_str}`  &nbsp;|&nbsp;  "
            f"**Overall Severity:** {severity_badge(severity)}",
            unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── KPI Metrics ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_data = [
    (k1, "☀ Sunspot No.",     f"{latest['sunspot_number']:.0f}",  ""),
    (k2, "⚡ Kp Index",        f"{latest['kp_index']:.1f} / 9",   ""),
    (k3, "🌬 Solar Wind",      f"{latest['solar_wind_speed']:.0f} km/s", ""),
    (k4, "🛰 Sat. Risk",       f"{latest['satellite_disruption_risk']:.1f}%", ""),
    (k5, "📡 GPS Risk",        f"{latest['gps_blackout_prob']:.1f}%", ""),
    (k6, "🔥 Composite Risk",  f"{comp_risk:.1f}%", ""),
]
for col, label, val, delta in kpi_data:
    col.metric(label, val)

st.markdown("<br>", unsafe_allow_html=True)

# ── Alert panel ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠ Active Space Weather Alerts</div>', unsafe_allow_html=True)
render_alerts(alerts)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_models, tab_forecast, tab_analysis, tab_report, tab_live, tab_forecast_v2, tab_map, tab_satellite, tab_insights = st.tabs([
    "📊 Overview",
    "🤖 ML Models",
    "🔭 Forecast",
    "🔬 Deep Analysis",
    "📄 Report",
    "📡 Live NOAA",
    "🔮 48H Forecast",
    "🌍 Global Risk Map",
    "🛰 Satellite Risk",
    "🧠 AI Insights"
])
tab_sat_track, tab_cme_3d, tab_ml_advanced, tab_replay = st.tabs([
    "🛰 Live Satellite Tracking",
    "🌞 Earth-Sun CME",
    "📈 Advanced ML Analytics",
    "🕒 Historical Storm Replay"
])
with tab_sat_track:
    st.markdown('<div class="section-header">🛰 Live Satellite Tracking</div>', unsafe_allow_html=True)

    fig_sat, iss_df = create_satellite_tracking_map(
        kp_index=float(latest["kp_index"]),
        sat_risk=float(latest["satellite_disruption_risk"])
    )

    st.plotly_chart(fig_sat, use_container_width=True)

    st.markdown("### Current ISS Position")
    st.dataframe(iss_df, use_container_width=True)

    st.info(
        "This module tracks the ISS live and overlays high-risk satellite disruption zones "
        "based on geomagnetic and GPS interference conditions."
    )


with tab_cme_3d:
    st.markdown('<div class="section-header">🌞 Earth-Sun CME Visualization</div>', unsafe_allow_html=True)

    fig_cme = create_earth_sun_cme_visual(
        kp_index=float(latest["kp_index"]),
        solar_wind_speed=float(latest["solar_wind_speed"])
    )

    st.plotly_chart(fig_cme, use_container_width=True)

    st.warning(
        "This is a scientific-style CME propagation visualization. "
        "It represents solar wind particles, CME plasma flow, Earth, Sun, and magnetosphere boundary."
    )


with tab_ml_advanced:
    st.markdown('<div class="section-header">📈 Advanced ML Analytics</div>', unsafe_allow_html=True)

    if "results" not in st.session_state:
        st.warning("Train models first from the ML Models tab to unlock full analytics.")
    else:
        results = st.session_state["results"]

        leaderboard = create_model_leaderboard(results)
        st.markdown("### Model Leaderboard")
        st.dataframe(leaderboard, use_container_width=True)

        best_model_name = leaderboard.iloc[0]["Model"]
        best_data = results[best_model_name]

        y_true = best_data["y_test"]
        y_pred = best_data["y_pred"]

        col1, col2 = st.columns(2)

        with col1:
            fig_residual = create_residual_plot(y_true, y_pred)
            st.plotly_chart(fig_residual, use_container_width=True)

        with col2:
            fig_dist = create_prediction_distribution(y_true, y_pred)
            st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("### Severity Confusion Matrix")
    fig_cm, report = create_confusion_matrix_from_severity(raw_df)
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("View Classification Report"):
        st.code(report)


with tab_replay:
    st.markdown('<div class="section-header">🕒 Historical Storm Replay</div>', unsafe_allow_html=True)

    selected_event = st.selectbox(
        "Select Historical Solar Storm Event",
        list(HISTORICAL_EVENTS.keys())
    )

    fig_replay, replay_df, event_info = create_historical_replay_chart(selected_event)

    st.plotly_chart(fig_replay, use_container_width=True)

    st.markdown("### Event Summary")
    st.info(event_info["description"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Peak Kp Index", event_info["peak_kp"])
    c2.metric("Satellite Risk", f"{event_info['satellite_risk']}%")
    c3.metric("GPS Risk", f"{event_info['gps_risk']}%")

    st.markdown("### Replay Data")
    st.dataframe(replay_df, use_container_width=True)
# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:
    recent = raw_df.tail(n_days_display).copy()
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])

    # Gauge row
    st.markdown('<div class="section-header">⚡ Risk Gauges</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_chart(float(latest["satellite_disruption_risk"]), "Satellite Disruption Risk"), use_container_width=True)
    g2.plotly_chart(gauge_chart(float(latest["gps_blackout_prob"]), "GPS Blackout Probability"), use_container_width=True)
    g3.plotly_chart(gauge_chart(float(latest["kp_index"]), "Kp Index", max_val=9), use_container_width=True)

    # Solar activity timeline
    st.markdown('<div class="section-header">☀ Solar Activity Timeline</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=["Sunspot Number", "Kp Index", "Solar Wind Speed (km/s)"])

    fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["sunspot_number"],
                             fill="tozeroy", fillcolor=C1, line=dict(color=C1, width=1.2),
                             name="SSN"), row=1, col=1)
    fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["kp_index"],
                             fill="tozeroy", fillcolor=C3, line=dict(color=C3, width=1.2),
                             name="Kp"), row=2, col=1)
    fig.add_hline(y=5, line_dash="dot", line_color=C5, opacity=0.6, row=2, col=1)
    fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["solar_wind_speed"],
                             fill="tozeroy", fillcolor=C4, line=dict(color=C4, width=1.2),
                             name="Wind Speed"), row=3, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=480,
                      title_text="Multi-Parameter Solar Activity Dashboard")
    for i in range(1, 4):
        fig.update_yaxes(gridcolor="#0a2540", row=i, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Risk scores over time
    st.markdown('<div class="section-header">🛰 Risk Scores Over Time</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=recent["timestamp"], y=recent["satellite_disruption_risk"],
                              name="Satellite Risk", line=dict(color=C2, width=1.5)))
    fig2.add_trace(go.Scatter(x=recent["timestamp"], y=recent["gps_blackout_prob"],
                              name="GPS Risk", line=dict(color=C3, width=1.5)))
    fig2.add_hrect(y0=70, y1=100, fillcolor="#ff0040", line_width=0)
    fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                       title="Satellite & GPS Disruption Risk Scores",
                       yaxis_title="Risk Score (0-100)")
    st.plotly_chart(fig2, use_container_width=True)

    # Severity distribution bar chart
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">🌡 Severity Distribution</div>', unsafe_allow_html=True)
        sev_counts = raw_df["severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Days"]
        sev_order  = ["Low", "Moderate", "High", "Extreme"]
        sev_colors = {"Low": C3, "Moderate": C5, "High": C2, "Extreme": "#ff0040"}
        fig3 = px.bar(sev_counts, x="Severity", y="Days",
                      color="Severity",
                      color_discrete_map=sev_colors,
                      category_orders={"Severity": sev_order})
        fig3.update_layout(**PLOTLY_LAYOUT, height=300, showlegend=False,
                           title="Geomagnetic Severity Frequency")
        st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">🔥 Flare Class Distribution</div>', unsafe_allow_html=True)
        flare_counts = raw_df["flare_class"].value_counts().reset_index()
        flare_counts.columns = ["Class", "Count"]
        fig4 = px.pie(flare_counts, names="Class", values="Count",
                      color_discrete_sequence=[C3, C5, C1, C2, "#ff0040"],
                      hole=0.45)
        fig4.update_layout(**PLOTLY_LAYOUT, height=300,
                           title="Solar Flare Class Distribution")
        fig4.update_traces(textfont_color="white", textfont_family="Share Tech Mono")
        st.plotly_chart(fig4, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
with tab_models:
    st.markdown('<div class="section-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)
    st.info(f"Training target: **{TARGETS[target_sel]}**  — adjust in sidebar.")

    col_train, _ = st.columns([1, 2])
    train_btn = col_train.button("🚀 Train All Models", use_container_width=True)

    if train_btn or "results" not in st.session_state:
        with st.spinner("🧠 Training models — please wait…"):
            results = train_models_cached(target_sel)
            st.session_state["results"] = results
            st.session_state["results_target"] = target_sel
    else:
        results = st.session_state.get("results", {})

    if results:
        # Performance table
        st.markdown('<div class="section-header">📊 Model Comparison</div>', unsafe_allow_html=True)
        perf = {name: {k: v for k, v in d.items() if k in ("mae","rmse","r2","train_time")}
                for name, d in results.items()}
        perf_df = pd.DataFrame(perf).T.reset_index().rename(columns={"index": "Model"})
        perf_df = perf_df.sort_values("r2", ascending=False)
        perf_df.index = range(1, len(perf_df)+1)

        # Highlight best row
        def _highlight(row):
            if row.name == 1:
                return ["background-color: #002244; color: #00d4ff"] * len(row)
            return [""] * len(row)

        st.dataframe(
            perf_df.style.apply(_highlight, axis=1)
                   .format({"mae": "{:.3f}", "rmse": "{:.3f}",
                            "r2": "{:.4f}", "train_time": "{:.1f}s"}),
            use_container_width=True, height=220)

        # Metric bar charts
        fig_perf = make_subplots(rows=1, cols=3,
                                  subplot_titles=["MAE ↓", "RMSE ↓", "R² ↑"])
        names  = perf_df["Model"].tolist()
        colors = [C1 if i == 0 else "#1a3a5c" for i in range(len(names))]

        fig_perf.add_trace(go.Bar(x=names, y=perf_df["mae"], marker_color=colors, name="MAE"),
                           row=1, col=1)
        fig_perf.add_trace(go.Bar(x=names, y=perf_df["rmse"], marker_color=colors, name="RMSE"),
                           row=1, col=2)
        fig_perf.add_trace(go.Bar(x=names, y=perf_df["r2"], marker_color=colors, name="R²"),
                           row=1, col=3)
        fig_perf.update_layout(**PLOTLY_LAYOUT, height=320,
                               title="Model Performance Metrics", showlegend=False)
        st.plotly_chart(fig_perf, use_container_width=True)

        # Actual vs Predicted for best model
        st.markdown('<div class="section-header">🎯 Actual vs Predicted</div>', unsafe_allow_html=True)
        best_name = perf_df["Model"].iloc[0]
        best_data = results[best_name]

        y_test = np.array(best_data["y_test"])
        y_pred = np.array(best_data["y_pred"])

        col_scatter, col_resid = st.columns(2)
        with col_scatter:
            lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            fig_av = go.Figure()
            fig_av.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                        marker=dict(color=C1, size=4, opacity=0.5),
                                        name="Predictions"))
            fig_av.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi],
                                        mode="lines", line=dict(color=C2, dash="dash"),
                                        name="Perfect Fit"))
            fig_av.update_layout(**PLOTLY_LAYOUT, height=320,
                                 title=f"{best_name} — Actual vs Predicted",
                                 xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig_av, use_container_width=True)

        with col_resid:
            residuals = y_test - y_pred
            fig_res = go.Figure(go.Histogram(x=residuals, nbinsx=40,
                                             marker_color=C4, opacity=0.8))
            fig_res.add_vline(x=0, line_color=C2, line_dash="dash")
            fig_res.update_layout(**PLOTLY_LAYOUT, height=320,
                                  title="Residual Distribution",
                                  xaxis_title="Residual", yaxis_title="Count")
            st.plotly_chart(fig_res, use_container_width=True)

        # Feature importance
        best_model = best_data["model"]
        imp = getattr(best_model, "feature_importances_", None)
        if imp is not None:
            st.markdown('<div class="section-header">🔬 Feature Importance</div>', unsafe_allow_html=True)
            X, _ = get_X_y(df_proc, target_sel)
            feat_names = X.columns.tolist()
            indices = np.argsort(imp)[-20:][::-1]
            fig_fi = go.Figure(go.Bar(
                x=imp[indices][::-1],
                y=[feat_names[i] for i in indices][::-1],
                orientation="h",
                marker=dict(
                    color=imp[indices][::-1],
                    colorscale=[[0,"#0a1628"],[0.5, C4],[1.0, C1]],
                    showscale=False),
            ))
            fig_fi.update_layout(**PLOTLY_LAYOUT, height=480,
                                 title=f"Top 20 Feature Importances — {best_name}",
                                 xaxis_title="Importance")
            st.plotly_chart(fig_fi, use_container_width=True)

    else:
        st.info("Click **Train All Models** to begin.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab_forecast:
    st.markdown('<div class="section-header">🔭 Next 48-Hour Space Weather Forecast</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        # Simulate 48-h forecast using rolling extrapolation
        history = raw_df.tail(90).copy()
        history["timestamp"] = pd.to_datetime(history["timestamp"])

        kp_hist    = history["kp_index"].values
        sat_hist   = history["satellite_disruption_risk"].values
        gps_hist   = history["gps_blackout_prob"].values
        wind_hist  = history["solar_wind_speed"].values

        def extrapolate_48h(series, noise_scale=0.5):
            trend = (series[-7:].mean() - series[-30:].mean()) / 23
            steps = []
            last  = series[-1]
            for h in range(1, 49):
                last = np.clip(last + trend + np.random.normal(0, noise_scale), 0, series.max() * 1.1)
                steps.append(last)
            return np.array(steps)

        np.random.seed(int(pd.Timestamp.now().timestamp()) % 1000)
        future_times = [history["timestamp"].iloc[-1] + timedelta(hours=h) for h in range(1, 49)]
        kp_fore   = extrapolate_48h(kp_hist, 0.3)
        sat_fore  = extrapolate_48h(sat_hist, 2)
        gps_fore  = extrapolate_48h(gps_hist, 1.5)
        wind_fore = extrapolate_48h(wind_hist, 20)

        fig_fc = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.12,
                               subplot_titles=["Kp Index", "Satellite Risk",
                                               "GPS Blackout Prob", "Solar Wind Speed"])

        def _history_trace(ts, vals, color, name):
            return go.Scatter(x=ts, y=vals, line=dict(color=color, width=1.2),
                              name=name, opacity=0.6)

        def _forecast_trace(ts, vals, color, name):
            return go.Scatter(x=ts, y=vals,
                              line=dict(color=color, width=2, dash="dot"),
                              name=name + " (forecast)")

        hist_ts = history["timestamp"]
        fig_fc.add_trace(_history_trace(hist_ts, kp_hist,   C3, "Kp History"),   row=1, col=1)
        fig_fc.add_trace(_forecast_trace(future_times, kp_fore,  C3, "Kp"),       row=1, col=1)
        fig_fc.add_trace(_history_trace(hist_ts, sat_hist,  C2, "Sat History"),  row=1, col=2)
        fig_fc.add_trace(_forecast_trace(future_times, sat_fore, C2, "Sat Risk"), row=1, col=2)
        fig_fc.add_trace(_history_trace(hist_ts, gps_hist,  C1, "GPS History"),  row=2, col=1)
        fig_fc.add_trace(_forecast_trace(future_times, gps_fore, C1, "GPS Risk"), row=2, col=1)
        fig_fc.add_trace(_history_trace(hist_ts, wind_hist, C4, "Wind History"), row=2, col=2)
        fig_fc.add_trace(_forecast_trace(future_times, wind_fore,C4, "Wind Speed"),row=2, col=2)

        # Shaded forecast zone
        for r, c in [(1,1),(1,2),(2,1),(2,2)]:
            fig_fc.add_vrect(
                x0=future_times[0], x1=future_times[-1],
                fillcolor="#00d4ff", line_width=0,
                row=r, col=c)

        fig_fc.update_layout(**PLOTLY_LAYOUT, height=480,
                             title="48-Hour Space Weather Forecast",
                             showlegend=False)
        st.plotly_chart(fig_fc, use_container_width=True)

    with col_f2:
        st.markdown("**📅 48-Hour Summary**")
        max_kp   = kp_fore.max()
        max_sat  = sat_fore.max()
        max_gps  = gps_fore.max()
        max_wind = wind_fore.max()

        def sev_color(v, thresholds):
            for thresh, label, color in thresholds:
                if v >= thresh: return label, color
            return "Low", C3

        kp_sev,  kp_col  = sev_color(max_kp,  [(7,"Extreme","#ff0040"),(5,"High",C2),(3,"Moderate",C5),(0,"Low",C3)])
        sat_sev, sat_col = sev_color(max_sat, [(80,"Extreme","#ff0040"),(60,"High",C2),(40,"Moderate",C5),(0,"Low",C3)])

        st.markdown(f"""
        <div style="background:#0a1628;border:1px solid #1a3a5c;border-radius:8px;padding:16px;margin-top:8px;">
          <div style="color:#4a7fa8;font-size:0.7rem;letter-spacing:0.15em;margin-bottom:12px;">FORECAST PEAK VALUES</div>

          <div style="display:flex;justify-content:space-between;margin:8px 0;">
            <span>Max Kp Index</span>
            <span style="color:{kp_col};font-family:Orbitron,monospace;">{max_kp:.1f}</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin:8px 0;">
            <span>Sat. Risk Peak</span>
            <span style="color:{sat_col};font-family:Orbitron,monospace;">{max_sat:.1f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin:8px 0;">
            <span>GPS Risk Peak</span>
            <span style="color:{C1};font-family:Orbitron,monospace;">{max_gps:.1f}%</span>
          </div>
          <div style="display:flex;justify-content:space-between;margin:8px 0;">
            <span>Wind Speed Peak</span>
            <span style="color:{C4};font-family:Orbitron,monospace;">{max_wind:.0f} km/s</span>
          </div>

          <hr style="border-color:#1a3a5c;margin:12px 0;">
          <div style="text-align:center;">
            <div style="color:#4a7fa8;font-size:0.7rem;margin-bottom:8px;">STORM PROBABILITY</div>
            <div style="font-family:Orbitron,monospace;font-size:1.6rem;color:{kp_col};">
              {min(100, int(max_kp / 9 * 100 + np.random.uniform(-5,5)))}%
            </div>
            <div style="color:{kp_col};font-size:0.75rem;">{kp_sev} Risk</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#080f1f;border:1px solid #0a2540;border-radius:8px;padding:12px;font-size:0.78rem;color:#4a7fa8;">
        ℹ️ Forecast uses rolling trend extrapolation. For best accuracy, train LSTM model via <b>main.py</b>.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown('<div class="section-header">🔬 Correlation & Statistical Analysis</div>', unsafe_allow_html=True)

    # Correlation heatmap
    cols_corr = ["sunspot_number","f107_solar_flux","solar_wind_speed","imf_bz",
                 "kp_index","dst_index","satellite_disruption_risk","gps_blackout_prob",
                 "flare_intensity","proton_flux","solar_cycle_phase"]
    cols_corr  = [c for c in cols_corr if c in raw_df.columns]
    corr_mat   = raw_df[cols_corr].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.columns.tolist(),
        colorscale=[[0,"#001a44"],[0.5,"#050a14"],[1.0,"#00d4ff"]],
        zmid=0,
        text=corr_mat.round(2).astype(str).values,
        texttemplate="%{text}",
        textfont={"size": 9, "family": "Share Tech Mono"},
    ))
    fig_corr.update_layout(**PLOTLY_LAYOUT, height=520,
                           title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Scatter matrix for key variables
    st.markdown('<div class="section-header">🌐 Solar Cycle vs Risk Scatter</div>', unsafe_allow_html=True)
    samp = raw_df.sample(min(500, len(raw_df)), random_state=42)
    samp["solar_wind_speed"] = (
        samp["solar_wind_speed"]
        .fillna(1)
        .abs()
    )
    fig_scatter = px.scatter(
        samp, x="kp_index", y="satellite_disruption_risk",
        color="severity",
        color_discrete_map={"Low": C3, "Moderate": C5, "High": C2, "Extreme": "#ff0040"},

        size="solar_wind_speed", size_max=12, opacity=0.7,
        hover_data=["sunspot_number", "solar_wind_speed", "gps_blackout_prob"],
        title="Kp Index vs Satellite Risk (bubble size = solar wind speed)",
    )
    fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Time-lagged cross-correlation
    st.markdown('<div class="section-header">⏱ Time-Lagged Cross-Correlation</div>', unsafe_allow_html=True)
    lags = range(-14, 15)
    xcorrs_kp_sat, xcorrs_kp_gps = [], []
    kp_s   = raw_df["kp_index"].dropna()
    sat_s  = raw_df["satellite_disruption_risk"].dropna()
    gps_s  = raw_df["gps_blackout_prob"].dropna()
    n = min(len(kp_s), len(sat_s), len(gps_s))
    kp_s, sat_s, gps_s = kp_s.values[-n:], sat_s.values[-n:], gps_s.values[-n:]
    for lag in lags:
        if lag >= 0:
            xcorrs_kp_sat.append(np.corrcoef(kp_s[:-lag or n], sat_s[lag:])[0,1] if lag else np.corrcoef(kp_s, sat_s)[0,1])
            xcorrs_kp_gps.append(np.corrcoef(kp_s[:-lag or n], gps_s[lag:])[0,1] if lag else np.corrcoef(kp_s, gps_s)[0,1])
        else:
            xcorrs_kp_sat.append(np.corrcoef(kp_s[-lag:], sat_s[:lag or n])[0,1])
            xcorrs_kp_gps.append(np.corrcoef(kp_s[-lag:], gps_s[:lag or n])[0,1])

    fig_xcorr = go.Figure()
    fig_xcorr.add_trace(go.Bar(x=list(lags), y=xcorrs_kp_sat, name="Kp → Sat Risk",
                               marker_color=C2, opacity=0.8))
    fig_xcorr.add_trace(go.Bar(x=list(lags), y=xcorrs_kp_gps, name="Kp → GPS Risk",
                               marker_color=C1, opacity=0.8))
    fig_xcorr.add_vline(x=0, line_color="white", line_dash="dash", opacity=0.4)
    fig_xcorr.update_layout(**PLOTLY_LAYOUT, height=320,
                             title="Lagged Cross-Correlation: Kp → Risk Scores",
                             xaxis_title="Lag (days)", yaxis_title="Correlation",
                             barmode="overlay")
    st.plotly_chart(fig_xcorr, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — REPORT
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    st.markdown('<div class="section-header">📄 Downloadable Mission Report</div>', unsafe_allow_html=True)

    model_scores_for_report = {}
    if "results" in st.session_state:
        for name, d in st.session_state["results"].items():
            model_scores_for_report[name] = {k: v for k, v in d.items()
                                              if k in ("mae","rmse","r2")}

    predictions_for_report = {
        "Kp Index"                   : float(latest["kp_index"]),
        "Sunspot Number"             : float(latest["sunspot_number"]),
        "Solar Wind Speed (km/s)"    : float(latest["solar_wind_speed"]),
        "IMF Bz (nT)"                : float(latest["imf_bz"]),
        "Satellite Disruption Risk"  : float(latest["satellite_disruption_risk"]),
        "GPS Blackout Probability"   : float(latest["gps_blackout_prob"]),
        "Composite Risk Score"       : round(comp_risk, 2),
        "Overall Severity"           : severity,
        "Flare Class"                : str(latest["flare_class"]),
    }

    txt_report = generate_txt_report(predictions_for_report, alerts, model_scores_for_report)
    csv_report = generate_csv_report(predictions_for_report, alerts)

    st.code(txt_report, language=None)

    dl1, dl2 = st.columns(2)
    dl1.download_button("⬇ Download TXT Report", txt_report.encode(),
                        file_name="space_weather_report.txt", mime="text/plain",
                        use_container_width=True)
    dl2.download_button("⬇ Download CSV Report", csv_report,
                        file_name="space_weather_report.csv", mime="text/csv",
                        use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw_df.tail(50).sort_index(ascending=False)
                       .style.background_gradient(subset=["kp_index","satellite_disruption_risk"],
                                                  cmap="plasma"),
                 use_container_width=True, height=300)

    st.download_button("⬇ Download Full Dataset (CSV)",
                       raw_df.to_csv(index=False).encode(),
                       file_name="space_weather_full_dataset.csv",
                       mime="text/csv", use_container_width=False)

with tab_live:
    st.subheader("📡 Live NOAA Space Weather Feed")

    live_kp = get_live_kp()
    wind = get_solar_wind()
    alerts = get_noaa_alerts()

    c1, c2, c3 = st.columns(3)
    c1.metric("Live Kp Index", live_kp["kp_index"])
    c2.metric("Solar Wind Speed", f"{wind['speed']} km/s")
    c3.metric("Solar Wind Density", wind["density"])

    st.markdown("### NOAA Alerts")
    if alerts:
        for alert in alerts:
            st.warning(alert.get("message", "NOAA alert received"))
    else:
        st.success("No live NOAA alerts found.")


with tab_forecast_v2:
    st.subheader("🔮 Next 48-Hour Forecast")

    forecast_df = forecast_next_48_hours(raw_df)

    st.line_chart(
        forecast_df.set_index("timestamp")[
            ["predicted_kp_index", "predicted_satellite_risk", "predicted_gps_risk"]
        ]
    )

    st.dataframe(forecast_df.tail(10), use_container_width=True)


with tab_map:
    st.subheader("🌍 Global Satellite & GPS Risk Map")

    fig_map = create_global_risk_map()
    st.plotly_chart(fig_map, use_container_width=True)


with tab_satellite:
    st.subheader("🛰 Satellite Failure Probability Engine")
    score = (
    float(latest["kp_index"]) * 6 +
    float(latest["solar_wind_speed"]) / 20 +
    float(latest["gps_blackout_prob"]) * 0.5 +
    float(latest["satellite_disruption_risk"]) * 0.5
)

score = min(score, 100)

if score >= 80:
    level = "Extreme"
elif score >= 60:
    level = "High"
elif score >= 40:
    level = "Moderate"
else:
    level = "Low"


with tab_insights:
    st.subheader("🧠 AI-Generated Storm Insight")

    insight = generate_ai_insight(
        kp_index=float(latest["kp_index"]),
        solar_wind_speed=float(latest["solar_wind_speed"]),
        gps_risk=float(latest["gps_blackout_prob"]),
        sat_risk=float(latest["satellite_disruption_risk"]),
        severity=severity
    )

    st.info(insight)
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#1a3a5c;font-size:0.75rem;letter-spacing:0.15em;border-top:1px solid #0a2540;padding-top:16px;">
  🛸 SPACE WEATHER INTELLIGENCE SYSTEM  ·  ML-POWERED  ·  NOAA · NASA DONKI  ·  SCIKIT-LEARN · XGBOOST
</div>
""", unsafe_allow_html=True)

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()

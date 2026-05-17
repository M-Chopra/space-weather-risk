import pandas as pd
import numpy as np
import plotly.graph_objects as go


HISTORICAL_EVENTS = {
    "Carrington Event 1859": {
        "description": "One of the most powerful recorded geomagnetic storms. Telegraph systems failed across Europe and North America.",
        "peak_kp": 9,
        "satellite_risk": 100,
        "gps_risk": 100
    },
    "Halloween Storms 2003": {
        "description": "A major solar storm period that affected satellites, GPS, aviation, and power systems.",
        "peak_kp": 9,
        "satellite_risk": 95,
        "gps_risk": 90
    },
    "Quebec Blackout 1989": {
        "description": "A geomagnetic storm caused a major power grid failure in Quebec, Canada.",
        "peak_kp": 8,
        "satellite_risk": 82,
        "gps_risk": 75
    },
    "St. Patrick's Day Storm 2015": {
        "description": "The strongest geomagnetic storm of Solar Cycle 24.",
        "peak_kp": 8,
        "satellite_risk": 78,
        "gps_risk": 70
    }
}


def generate_historical_event_timeline(event_name):
    event = HISTORICAL_EVENTS[event_name]

    hours = np.arange(0, 72)
    peak_hour = 36

    kp = event["peak_kp"] * np.exp(-((hours - peak_hour) ** 2) / (2 * 10 ** 2))
    sat = event["satellite_risk"] * np.exp(-((hours - peak_hour) ** 2) / (2 * 12 ** 2))
    gps = event["gps_risk"] * np.exp(-((hours - peak_hour) ** 2) / (2 * 14 ** 2))

    df = pd.DataFrame({
        "hour": hours,
        "kp_index": np.round(kp, 2),
        "satellite_risk": np.round(sat, 2),
        "gps_risk": np.round(gps, 2)
    })

    return df


def create_historical_replay_chart(event_name):
    df = generate_historical_event_timeline(event_name)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["kp_index"],
        mode="lines",
        name="Kp Index",
        line=dict(color="#39ff14", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["satellite_risk"],
        mode="lines",
        name="Satellite Risk",
        line=dict(color="#ff6b35", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df["hour"],
        y=df["gps_risk"],
        mode="lines",
        name="GPS Risk",
        line=dict(color="#00d4ff", width=3)
    ))

    fig.update_layout(
        title=f"Historical Storm Replay: {event_name}",
        xaxis_title="Hours Since Event Start",
        yaxis_title="Intensity / Risk",
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff"),
        height=450
    )

    return fig, df, HISTORICAL_EVENTS[event_name]
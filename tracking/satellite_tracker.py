import requests
import pandas as pd
import plotly.express as px


def get_iss_location():
    """
    Fetch current ISS latitude and longitude.
    Fallback is used if API fails.
    """
    try:
        url = "http://api.open-notify.org/iss-now.json"
        data = requests.get(url, timeout=10).json()

        lat = float(data["iss_position"]["latitude"])
        lon = float(data["iss_position"]["longitude"])

        return pd.DataFrame({
            "satellite": ["International Space Station"],
            "latitude": [lat],
            "longitude": [lon],
            "risk_zone": ["Live Tracking"],
            "altitude_km": [420]
        })

    except Exception:
        return pd.DataFrame({
            "satellite": ["International Space Station"],
            "latitude": [20.5],
            "longitude": [78.9],
            "risk_zone": ["Fallback Position"],
            "altitude_km": [420]
        })


def create_satellite_tracking_map(kp_index=3.0, sat_risk=40.0):
    iss_df = get_iss_location()

    risk_points = pd.DataFrame({
        "satellite": [
            "Polar Orbit Risk Zone",
            "GPS Interference Zone",
            "Auroral Disruption Zone"
        ],
        "latitude": [75, 60, -65],
        "longitude": [0, -100, 120],
        "risk_zone": ["Extreme", "High", "Moderate"],
        "altitude_km": [800, 20200, 1000]
    })

    df = pd.concat([iss_df, risk_points], ignore_index=True)

    fig = px.scatter_geo(
        df,
        lat="latitude",
        lon="longitude",
        size="altitude_km",
        color="risk_zone",
        hover_name="satellite",
        projection="natural earth",
        title="Live Satellite Tracking & Space Weather Risk Overlay",
        color_discrete_map={
            "Live Tracking": "#00d4ff",
            "Fallback Position": "#ffd700",
            "Extreme": "#ff0040",
            "High": "#ff6b35",
            "Moderate": "#ffd700"
        }
    )

    fig.update_layout(
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff"),
        geo=dict(
            bgcolor="#050a14",
            showland=True,
            landcolor="#101827",
            showocean=True,
            oceancolor="#050a14",
            showcountries=True,
            countrycolor="#1a3a5c"
        )
    )

    return fig, iss_df
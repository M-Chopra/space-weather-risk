import pandas as pd
import plotly.express as px

def create_global_risk_map():
    data = pd.DataFrame({
        "region": ["North America", "Europe", "India", "Japan", "Australia", "Arctic Zone"],
        "lat": [37.1, 54.5, 22.9, 36.2, -25.2, 75.0],
        "lon": [-95.7, 15.2, 78.9, 138.2, 133.7, 0.0],
        "risk": [55, 62, 48, 58, 45, 88],
        "status": ["Moderate", "High", "Moderate", "High", "Moderate", "Extreme"]
    })

    fig = px.scatter_geo(
        data,
        lat="lat",
        lon="lon",
        size="risk",
        color="status",
        hover_name="region",
        projection="natural earth",
        title="Global GPS & Satellite Disruption Risk Map",
        color_discrete_map={
            "Low": "#39ff14",
            "Moderate": "#ffd700",
            "High": "#ff6b35",
            "Extreme": "#ff0040"
        }
    )

    fig.update_layout(
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff")
    )

    return fig
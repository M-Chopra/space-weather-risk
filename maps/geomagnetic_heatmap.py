import pandas as pd
import plotly.express as px
import numpy as np


def create_geomagnetic_heatmap():

    np.random.seed(42)

    locations = [
        "USA",
        "Canada",
        "Russia",
        "Norway",
        "Sweden",
        "Finland",
        "Greenland",
        "Iceland",
        "UK",
        "Germany",
        "India",
        "Japan",
        "Australia",
        "Brazil"
    ]

    risk = np.random.randint(20, 100, len(locations))

    df = pd.DataFrame({
        "country": locations,
        "risk": risk
    })

    fig = px.choropleth(
        df,
        locations="country",
        locationmode="country names",
        color="risk",
        hover_name="country",
        color_continuous_scale="Turbo",
        title="🌐 Global Geomagnetic Disturbance Heatmap"
    )

    fig.update_layout(
        paper_bgcolor="#050a14",
        plot_bgcolor="#050a14",
        font_color="white",
        geo=dict(
            bgcolor="#050a14",
            showframe=False,
            showcoastlines=True,
            projection_type="orthographic"
        ),
        height=700
    )

    return fig
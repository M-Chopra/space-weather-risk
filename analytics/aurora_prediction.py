import pandas as pd
import plotly.express as px
import numpy as np


def generate_aurora_prediction(kp_index):

    regions = [
        "Alaska",
        "Northern Canada",
        "Iceland",
        "Norway",
        "Sweden",
        "Finland",
        "Greenland",
        "Scotland"
    ]

    visibility = []

    for _ in regions:

        score = min(
            100,
            kp_index * 10 + np.random.randint(5, 25)
        )

        visibility.append(score)

    aurora_df = pd.DataFrame({

        "Region": regions,
        "Visibility": visibility
    })

    fig = px.bar(

        aurora_df,
        x="Region",
        y="Visibility",
        color="Visibility",

        title="🌌 Aurora Visibility Prediction",

        color_continuous_scale="Turbo"
    )

    fig.update_layout(

        paper_bgcolor="#050a14",
        plot_bgcolor="#050a14",
        font_color="white",
        height=600
    )

    return aurora_df, fig
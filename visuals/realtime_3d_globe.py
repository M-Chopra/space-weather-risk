import numpy as np
import plotly.graph_objects as go


def create_realtime_3d_globe():

    # Earth sphere
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(theta), np.sin(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.ones(100), np.cos(phi))

    earth = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale="Blues",
        showscale=False,
        opacity=0.95
    )

    # Simulated satellite positions
    sat_x = [1.2, -1.3, 0.8, -0.7]
    sat_y = [0.4, -0.6, 1.1, -1.2]
    sat_z = [0.8, -0.9, 0.2, 1.0]

    satellites = go.Scatter3d(
        x=sat_x,
        y=sat_y,
        z=sat_z,
        mode="markers+text",
        marker=dict(
            size=5,
            color="#ff0040"
        ),
        text=["ISS", "GPS-1", "STARLINK", "WEATHER-SAT"],
        textposition="top center",
        name="Satellites"
    )

    # Auroral ring
    aurora_theta = np.linspace(0, 2 * np.pi, 200)

    aurora_x = 1.05 * np.cos(aurora_theta)
    aurora_y = 1.05 * np.sin(aurora_theta)
    aurora_z = np.full_like(aurora_theta, 0.7)

    aurora = go.Scatter3d(
        x=aurora_x,
        y=aurora_y,
        z=aurora_z,
        mode="lines",
        line=dict(
            color="#39ff14",
            width=6
        ),
        name="Auroral Zone"
    )

    fig = go.Figure(data=[earth, satellites, aurora])

    fig.update_layout(
        title="🌍 Real-Time 3D Space Weather Globe",
        paper_bgcolor="#050a14",
        scene=dict(
            bgcolor="#050a14",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        height=700
    )

    return fig
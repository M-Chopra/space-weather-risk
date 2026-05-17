import numpy as np
import plotly.graph_objects as go


def create_earth_sun_cme_visual(kp_index=4.0, solar_wind_speed=450.0):
    """
    3D Earth-Sun-CME visualization.
    Lightweight Plotly version for Streamlit Cloud.
    """

    # Sun
    sun = go.Scatter3d(
        x=[-5], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=35, color="#ffcc00"),
        text=["SUN"],
        textposition="top center",
        name="Sun"
    )

    # Earth
    earth = go.Scatter3d(
        x=[5], y=[0], z=[0],
        mode="markers+text",
        marker=dict(size=18, color="#00d4ff"),
        text=["EARTH"],
        textposition="top center",
        name="Earth"
    )

    # CME trajectory
    t = np.linspace(-4.5, 4.5, 80)
    wave_y = np.sin(t * 2) * 0.4
    wave_z = np.cos(t * 2) * 0.4

    cme = go.Scatter3d(
        x=t,
        y=wave_y,
        z=wave_z,
        mode="lines",
        line=dict(color="#ff0040", width=8),
        name="CME Plasma Stream"
    )

    # Magnetic shield around Earth
    theta = np.linspace(0, 2 * np.pi, 100)
    shield_x = np.full_like(theta, 5)
    shield_y = np.cos(theta) * (1.2 + kp_index / 10)
    shield_z = np.sin(theta) * (1.2 + kp_index / 10)

    shield = go.Scatter3d(
        x=shield_x,
        y=shield_y,
        z=shield_z,
        mode="lines",
        line=dict(color="#39ff14", width=4),
        name="Magnetosphere Boundary"
    )

    # Solar wind particles
    particle_x = np.linspace(-4, 4, 30)
    particle_y = np.random.normal(0, 1.5, 30)
    particle_z = np.random.normal(0, 1.5, 30)

    particles = go.Scatter3d(
        x=particle_x,
        y=particle_y,
        z=particle_z,
        mode="markers",
        marker=dict(
            size=4,
            color=solar_wind_speed,
            colorscale="Plasma",
            opacity=0.8
        ),
        name="Solar Wind Particles"
    )

    fig = go.Figure(data=[sun, earth, cme, shield, particles])

    fig.update_layout(
        title="3D Earth-Sun CME Propagation Visualization",
        paper_bgcolor="#08111e",
        plot_bgcolor="#050a14",
        font=dict(color="#c0d8ff"),
        height=600,
        scene=dict(
            xaxis=dict(backgroundcolor="#050a14", gridcolor="#1a3a5c"),
            yaxis=dict(backgroundcolor="#050a14", gridcolor="#1a3a5c"),
            zaxis=dict(backgroundcolor="#050a14", gridcolor="#1a3a5c"),
        )
    )

    return fig
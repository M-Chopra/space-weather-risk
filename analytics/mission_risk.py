import pandas as pd


def calculate_mission_risk(
    kp_index,
    solar_wind,
    gps_risk,
    sat_risk
):

    score = (
        kp_index * 8 +
        solar_wind / 30 +
        gps_risk * 0.9 +
        sat_risk * 1.1
    )

    score = min(score, 100)

    if score >= 80:
        level = "EXTREME"
        color = "red"

    elif score >= 60:
        level = "HIGH"
        color = "orange"

    elif score >= 40:
        level = "MODERATE"
        color = "yellow"

    else:
        level = "LOW"
        color = "green"

    missions = pd.DataFrame({

        "Mission System": [
            "ISS Operations",
            "GPS Navigation",
            "Satellite Communications",
            "Polar Aviation",
            "Deep Space Missions"
        ],

        "Risk Level": [
            level,
            level,
            level,
            "HIGH" if kp_index > 6 else "MODERATE",
            "EXTREME" if kp_index > 8 else level
        ]
    })

    return score, level, color, missions
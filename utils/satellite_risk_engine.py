def satellite_failure_probability(kp_index, solar_wind_speed, gps_risk, sat_risk):
    score = (
        kp_index * 8 +
        min(solar_wind_speed / 900, 1) * 25 +
        gps_risk * 0.35 +
        sat_risk * 0.45
    )

    score = max(0, min(score, 100))

    if score < 30:
        level = "Low"
    elif score < 55:
        level = "Moderate"
    elif score < 75:
        level = "High"
    else:
        level = "Extreme"

    return round(score, 2), level
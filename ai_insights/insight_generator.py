def generate_ai_insight(kp_index, solar_wind_speed, gps_risk, sat_risk, severity):
    insight = []

    if kp_index >= 7:
        insight.append("Extreme geomagnetic activity detected. Satellite and GPS systems may experience severe disruption.")
    elif kp_index >= 5:
        insight.append("Moderate to strong geomagnetic storm conditions are active. Communication systems should be monitored.")
    else:
        insight.append("Current geomagnetic activity is relatively stable.")

    if solar_wind_speed > 650:
        insight.append("Solar wind speed is elevated, increasing the probability of geomagnetic disturbances.")

    if gps_risk > 60:
        insight.append("GPS interference probability is high, especially in polar and high-latitude regions.")

    if sat_risk > 60:
        insight.append("Satellite disruption risk is elevated. Communication and navigation satellites may be affected.")

    insight.append(f"Overall system severity is currently classified as {severity}.")

    return " ".join(insight)
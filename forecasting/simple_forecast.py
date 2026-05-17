import numpy as np
import pandas as pd
from datetime import timedelta

def forecast_next_48_hours(df):
    recent = df.tail(30).copy()
    last_time = pd.to_datetime(recent["timestamp"].iloc[-1])

    kp_base = recent["kp_index"].mean()
    sat_base = recent["satellite_disruption_risk"].mean()
    gps_base = recent["gps_blackout_prob"].mean()

    future = []

    for h in range(1, 49):
        noise = np.random.normal(0, 0.8)

        kp_pred = max(0, min(9, kp_base + noise))
        sat_pred = max(0, min(100, sat_base + noise * 5))
        gps_pred = max(0, min(100, gps_base + noise * 4))

        future.append({
            "timestamp": last_time + timedelta(hours=h),
            "predicted_kp_index": round(kp_pred, 2),
            "predicted_satellite_risk": round(sat_pred, 2),
            "predicted_gps_risk": round(gps_pred, 2)
        })

    return pd.DataFrame(future)
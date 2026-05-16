"""
=============================================================================
  SPACE WEATHER PREDICTOR — Alert & Severity Engine
=============================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List
import numpy as np


@dataclass
class SpaceWeatherAlert:
    level    : str          # "INFO" | "WARNING" | "CRITICAL" | "EXTREME"
    message  : str
    metric   : str
    value    : float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return self.__dict__


# ── Threshold table ───────────────────────────────────────────────────────────
THRESHOLDS = {
    "kp_index": [
        (9.0, "EXTREME",  "Extreme geomagnetic storm — G5 event!"),
        (7.0, "CRITICAL", "Severe geomagnetic storm — G4 event"),
        (5.0, "WARNING",  "Moderate geomagnetic storm — G3 event"),
        (3.0, "INFO",     "Minor geomagnetic activity — G1/G2"),
    ],
    "satellite_disruption_risk": [
        (90,  "EXTREME",  "Extreme satellite disruption risk"),
        (70,  "CRITICAL", "High satellite disruption risk"),
        (50,  "WARNING",  "Elevated satellite disruption risk"),
        (25,  "INFO",     "Low satellite disruption risk"),
    ],
    "gps_blackout_prob": [
        (85,  "EXTREME",  "GPS blackout imminent"),
        (65,  "CRITICAL", "Severe GPS interference expected"),
        (40,  "WARNING",  "GPS accuracy degradation possible"),
        (20,  "INFO",     "Minor GPS scintillation possible"),
    ],
    "solar_wind_speed": [
        (800, "EXTREME",  "Extreme solar wind speed — CME arrival likely"),
        (600, "CRITICAL", "Very high solar wind speed"),
        (500, "WARNING",  "Elevated solar wind speed"),
        (450, "INFO",     "Slightly elevated solar wind"),
    ],
    "flare_intensity": [
        (4.5, "EXTREME",  "X-class solar flare — major radio blackout"),
        (3.5, "CRITICAL", "M-class flare — possible radio blackout"),
        (2.5, "WARNING",  "C-class flare — minor radio interference"),
        (1.5, "INFO",     "B-class flare activity"),
    ],
}

LEVEL_ORDER = {"INFO": 0, "WARNING": 1, "CRITICAL": 2, "EXTREME": 3}


def generate_alerts(metrics: dict) -> List[SpaceWeatherAlert]:
    """
    Evaluate a dict of {metric_name: value} against threshold table.
    Returns a list of SpaceWeatherAlert objects, sorted by severity.
    """
    alerts = []
    for metric, value in metrics.items():
        if metric not in THRESHOLDS or value is None:
            continue
        for threshold, level, message in THRESHOLDS[metric]:
            if value >= threshold:
                alerts.append(SpaceWeatherAlert(
                    level=level, message=message,
                    metric=metric, value=round(float(value), 2),
                    threshold=threshold))
                break   # only most severe alert per metric

    alerts.sort(key=lambda a: LEVEL_ORDER.get(a.level, 0), reverse=True)
    return alerts


def overall_severity(alerts: List[SpaceWeatherAlert]) -> str:
    """Derive overall severity from alert list."""
    if not alerts:
        return "Low"
    top = alerts[0].level
    return {"INFO": "Low", "WARNING": "Moderate",
            "CRITICAL": "High", "EXTREME": "Extreme"}.get(top, "Low")


def risk_score(kp: float, sat_risk: float, gps_prob: float,
               wind_speed: float, flare_int: float) -> float:
    """Composite 0-100 risk score."""
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    norms   = [kp / 9 * 100, sat_risk, gps_prob,
               (wind_speed - 280) / 620 * 100,
               flare_int / 5 * 100]
    return float(np.clip(sum(w * v for w, v in zip(weights, norms)), 0, 100))

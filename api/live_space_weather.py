import requests
import pandas as pd
from datetime import datetime, timedelta

NOAA_BASE = "https://services.swpc.noaa.gov/json"


def safe_float(value):
    try:
        value = str(value)

        cleaned = ""

        for ch in value:
            if ch.isdigit() or ch == "." or ch == "-":
                cleaned += ch

        if cleaned == "" or cleaned == "." or cleaned == "-":
            return 0.0

        return float(cleaned)

    except:
        return 0.0


def get_live_kp():
    url = f"{NOAA_BASE}/planetary_k_index_1m.json"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {
                "success": False,
                "time": None,
                "kp_index": 0.0
            }

        data = response.json()

        if not data or len(data) == 0:
            return {
                "success": False,
                "time": None,
                "kp_index": 0.0
            }

        latest = data[-1]

        kp_raw = latest.get("kp_index", latest.get("kp", 0))

        kp_value = safe_float(kp_raw)

        return {
            "success": True,
            "time": latest.get("time_tag"),
            "kp_index": kp_value
        }

    except Exception as e:
        print("NOAA ERROR:", e)

        return {
            "success": False,
            "time": None,
            "kp_index": 0.0
        }


def get_solar_wind():
    url = f"{NOAA_BASE}/rtsw/rtsw_wind_1m.json"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return {
                "success": False,
                "time": None,
                "speed": 0.0,
                "density": 0.0
            }

        data = response.json()

        if not data or len(data) == 0:
            return {
                "success": False,
                "time": None,
                "speed": 0.0,
                "density": 0.0
            }

        latest = data[-1]

        return {
            "success": True,
            "time": latest.get("time_tag"),
            "speed": safe_float(latest.get("speed", 0)),
            "density": safe_float(latest.get("density", 0))
        }

    except Exception as e:
        print("SOLAR WIND ERROR:", e)

        return {
            "success": False,
            "time": None,
            "speed": 0.0,
            "density": 0.0
        }


def get_noaa_alerts():
    url = f"{NOAA_BASE}/alerts.json"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return []

        data = response.json()

        return data[:5]

    except Exception as e:
        print("NOAA ALERT ERROR:", e)
        return []
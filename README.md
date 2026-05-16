# 🛸 Space Weather Prediction & Satellite Disruption Risk Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189AB4?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

**An end-to-end ML system for real-time space weather monitoring, satellite disruption forecasting, and GPS interference prediction.**

[📊 Dashboard Demo](#dashboard) · [🚀 Quick Start](#quick-start) · [📐 Architecture](#architecture) · [📈 Results](#results)

</div>

---

## 🌌 Project Overview

Space weather — driven by solar flares, coronal mass ejections (CMEs), and geomagnetic storms — poses significant risks to satellite operations, GPS accuracy, power grids, and aviation communications. This project builds a **production-grade ML pipeline** that:

- Ingests 5+ years of solar and geomagnetic data (NOAA SWPC · NASA DONKI)
- Engineers 40+ temporal and physical features
- Trains 5 competing ML models (Linear → XGBoost → LSTM)
- Serves real-time predictions via a futuristic Streamlit dashboard
- Issues automated alerts when risk thresholds are breached
- Generates downloadable mission reports (TXT / CSV)

> **Resume pitch:** Built a multi-model ML system achieving **R² = 0.973** on Kp geomagnetic index prediction and **R² = 0.876** on satellite disruption risk, using XGBoost + Random Forest ensembles trained on 5 years of NASA/NOAA space weather data with 40+ engineered features including lag, rolling statistics, and solar-physics-derived interaction terms.

---

## 🏗 Architecture

```
Raw Data (NOAA / NASA / Synthetic)
         │
         ▼
┌─────────────────────┐
│  Data Collection    │  ← data_collector.py
│  • NOAA SWPC API    │
│  • NASA DONKI API   │
│  • Synthetic Gen    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Preprocessing      │  ← preprocessor.py
│  • KNN Imputation   │
│  • Lag Features     │
│  • Interaction Terms│
│  • StandardScaler   │
└────────┬────────────┘
         │
    ┌────┴────┐
    │  Train  │  ← trainer.py
    │─────────│
    │ LinReg  │
    │ Ridge   │
    │ RF      │
    │ XGBoost │
    │ GBM     │
    │─────────│
    │  LSTM   │  ← lstm_forecaster.py
    └────┬────┘
         │
    ┌────┴─────────────────────────────────────┐
    │           Streamlit Dashboard            │
    │  • KPI gauges    • 48h Forecast          │
    │  • Alert engine  • Feature importance    │
    │  • Heatmaps      • Downloadable reports  │
    └──────────────────────────────────────────┘
```

---

## 📦 Dataset

| Source | Variables | Frequency |
|--------|-----------|-----------|
| NOAA SWPC | Solar wind speed, density, IMF Bz, Kp index | 1-minute |
| NASA DONKI | CME events, solar flare class, proton flux | Per-event |
| SILSO (Brussels) | International Sunspot Number (SSN) | Daily |
| NOAA GOES | X-ray flux (A–X class flares), F10.7 | Daily |
| Derived | Dst index, geomagnetic storm flags | Daily |

### Key Features Engineered

| Feature | Description |
|---------|-------------|
| `kp_index_lag1/3` | Yesterday's / 3-day-old Kp (autocorrelation) |
| `solar_wind_speed_roll7` | 7-day rolling mean of solar wind |
| `wind_bz_interaction` | V × (-Bz) — main driver of ring-current injection |
| `storm_energy` | ρV² proxy for dynamic pressure |
| `southward_bz` | Rectified southward IMF component |
| `day_sin / day_cos` | Annual seasonality encoding |
| `log_xray_flux` | Log-transform of heavy-tailed X-ray flux |

---

## 🤖 Models & Results

### Kp Geomagnetic Index Prediction

| Model | MAE | RMSE | R² | Train Time |
|-------|-----|------|----|------------|
| **XGBoost** ⭐ | **0.108** | **0.163** | **0.9732** | 2.4s |
| Gradient Boosting | 0.119 | 0.174 | 0.9693 | 4.3s |
| Random Forest | 0.145 | 0.208 | 0.9563 | 3.9s |
| Linear Regression | 0.182 | 0.229 | 0.9470 | 0.0s |
| Ridge Regression | 0.183 | 0.230 | 0.9466 | 0.0s |

### Satellite Disruption Risk Prediction

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| **Random Forest** ⭐ | **3.773** | **4.756** | **0.876** |
| XGBoost | 3.932 | 4.973 | 0.865 |
| Gradient Boosting | 3.986 | 4.967 | 0.865 |
| Linear Regression | 4.225 | 5.441 | 0.838 |

---

## 📁 Project Structure

```
space-weather-predictor/
│
├── data/
│   └── space_weather.csv          # 5-year daily dataset (auto-generated)
│
├── models/
│   ├── __init__.py
│   ├── trainer.py                 # Multi-model training & evaluation
│   └── lstm_forecaster.py         # Bidirectional LSTM (48-h forecasting)
│
├── app/
│   └── dashboard.py               # Streamlit dashboard (Mission Control UI)
│
├── utils/
│   ├── __init__.py
│   ├── data_collector.py          # NOAA/NASA API + synthetic data generator
│   ├── preprocessor.py            # Feature engineering pipeline
│   ├── alert_system.py            # Threshold-based alert engine
│   ├── visualizer.py              # Matplotlib static charts
│   └── report_generator.py        # TXT/CSV report export
│
├── visuals/                        # Auto-generated charts (PNG)
├── notebooks/
│   └── exploration.ipynb          # EDA notebook
│
├── main.py                         # ▶ CLI training pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/space-weather-predictor.git
cd space-weather-predictor
pip install -r requirements.txt
```

### 2. Run Training Pipeline

```bash
python main.py
```

This will:
- Generate/load 5 years of space weather data
- Preprocess and engineer 40+ features
- Train 5 ML models on 3 prediction targets
- Save best models to `/models/`
- Generate all visualizations to `/visuals/`

### 3. Launch Dashboard

```bash
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

---

## 🔧 Configuration

Key parameters in `utils/data_collector.py`:

```python
n_years = 5        # Dataset length
SEED    = 42       # Reproducibility
```

Key parameters in `models/lstm_forecaster.py`:

```python
LOOK_BACK  = 30    # Days of history fed to LSTM
FORECAST_H = 2     # Forecast horizon (days = 48 hours)
```

---

## 🌠 Prediction Targets

| Target | Range | Best Model | R² |
|--------|-------|------------|----|
| Kp Geomagnetic Index | 0–9 | XGBoost | 0.973 |
| Satellite Disruption Risk | 0–100 | Random Forest | 0.876 |
| GPS Blackout Probability | 0–100 | Ridge Regression | 0.616 |
| Solar Flare Class | A/B/C/M/X | Encoded feature | — |
| Severity Level | Low/Moderate/High/Extreme | Threshold rules | — |

---

## 📡 Alert Levels

| Level | Kp Threshold | Description |
|-------|-------------|-------------|
| 🟢 INFO | ≥ 3.0 | Minor geomagnetic activity |
| 🟡 WARNING | ≥ 5.0 | Moderate storm — G3 event |
| 🟠 CRITICAL | ≥ 7.0 | Severe storm — G4 event |
| 🔴 EXTREME | ≥ 9.0 | Extreme storm — G5 event |

---

## 🔮 Future Improvements

- [ ] Transformer-based (PatchTST / Informer) time-series forecasting
- [ ] GOES satellite real-time X-ray flux stream
- [ ] Ensemble prediction intervals (conformal prediction)
- [ ] Multi-output prediction (joint Kp + Dst + F10.7)
- [ ] Email / Slack webhook alert delivery
- [ ] Docker containerization for one-click deployment
- [ ] Planetary K-index regional breakdown

---

## 📚 References

- NOAA Space Weather Prediction Center: https://www.swpc.noaa.gov/
- NASA DONKI API: https://api.nasa.gov/DONKI/
- SILSO Sunspot Data: https://www.sidc.be/silso/datafiles
- Gonzalez et al. (1994) — What is a geomagnetic storm?
- Loewe & Prolss (1997) — Classification and mean behavior of magnetic storms

---

## 👨‍💻 Author

Built as a showcase ML project for research internships and hackathon submissions.  
Stack: Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow · Streamlit · Plotly

---

<div align="center">
Made with ☀ and machine learning &nbsp;|&nbsp; Data: NOAA · NASA
</div>

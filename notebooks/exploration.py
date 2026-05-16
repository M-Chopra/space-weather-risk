# %% [markdown]
# # 🛸 Space Weather EDA & Feature Analysis
# **Space Weather Prediction & Satellite Disruption Risk Analysis**
#
# This notebook walks through:
# 1. Dataset overview & statistics
# 2. Solar cycle analysis
# 3. Correlation analysis
# 4. Feature engineering rationale
# 5. Storm event case studies

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from utils.data_collector import load_or_generate_dataset
from utils.preprocessor   import preprocess, get_X_y

# Dark theme
plt.rcParams["figure.facecolor"]  = "#0d1117"
plt.rcParams["axes.facecolor"]    = "#0d1117"
plt.rcParams["text.color"]        = "white"
plt.rcParams["axes.labelcolor"]   = "white"
plt.rcParams["xtick.color"]       = "#aaa"
plt.rcParams["ytick.color"]       = "#aaa"

# %%  [markdown]
# ## 1. Load Dataset

# %%
df = load_or_generate_dataset(n_years=5, save_path="../data/space_weather.csv")
print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
df.head()

# %% [markdown]
# ## 2. Summary Statistics

# %%
print("=== Numerical Summary ===")
num_cols = df.select_dtypes(include=[np.number]).columns
print(df[num_cols].describe().round(2).to_string())

# %%
print("\n=== Missing Values ===")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)
print(f"\nTotal NaN rate: {df.isnull().mean().mean():.1%}")

# %% [markdown]
# ## 3. Solar Cycle Visualization

# %%
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, facecolor="#0d1117")

axes[0].plot(df["timestamp"], df["sunspot_number"], color="#00d4ff", lw=0.8)
axes[0].fill_between(df["timestamp"], df["sunspot_number"], alpha=0.2, color="#00d4ff")
axes[0].set_ylabel("Sunspot Number", color="white")
axes[0].set_title("Solar Cycle Activity Indicators", color="white", fontsize=13)

axes[1].plot(df["timestamp"], df["kp_index"], color="#39ff14", lw=0.8)
axes[1].axhline(5, color="#ffd700", lw=1, linestyle="--", alpha=0.7, label="Storm threshold")
axes[1].fill_between(df["timestamp"], df["kp_index"], where=df["kp_index"]>=5,
                      color="#ff6b35", alpha=0.3, label="Storm periods")
axes[1].set_ylabel("Kp Index", color="white")
axes[1].legend(loc="upper right", facecolor="#1a1a2e", labelcolor="white")

axes[2].plot(df["timestamp"], df["dst_index"], color="#bf5fff", lw=0.8)
axes[2].axhline(-100, color="#ff0040", lw=1, linestyle="--", alpha=0.7, label="Severe storm")
axes[2].fill_between(df["timestamp"], df["dst_index"], where=df["dst_index"]<=-50,
                      color="#ff0040", alpha=0.3)
axes[2].set_ylabel("Dst Index (nT)", color="white")
axes[2].set_xlabel("Date", color="white")

for ax in axes:
    ax.set_facecolor("#0d1117")
    ax.grid(color="#1f1f2e", linestyle="--", alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig("../visuals/notebook_solar_cycle.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

# %% [markdown]
# ## 4. Storm Event Analysis

# %%
storm_days = df[df["kp_index"] >= 5]
print(f"Total storm days (Kp ≥ 5): {len(storm_days)}")
print(f"Storm rate: {len(storm_days)/len(df):.1%}")
print(f"\nSeverity breakdown:")
print(df["severity"].value_counts())

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")

# Flare class vs storm probability
flare_storm = df.groupby("flare_class")["storm_flag"].mean() * 100
order = ["A","B","C","M","X"]
vals  = [flare_storm.get(c, 0) for c in order]
axes[0].bar(order, vals, color=["#39ff14","#00d4ff","#ffd700","#ff6b35","#ff0040"],
            edgecolor="none", alpha=0.85)
axes[0].set_title("Storm Probability by Flare Class", color="white")
axes[0].set_ylabel("Storm Probability (%)", color="white")
axes[0].set_facecolor("#0d1117")
for spine in axes[0].spines.values(): spine.set_visible(False)

# Kp vs Satellite risk
scatter = axes[1].scatter(df["kp_index"], df["satellite_disruption_risk"],
                           c=df["gps_blackout_prob"], cmap="plasma",
                           s=8, alpha=0.5)
plt.colorbar(scatter, ax=axes[1], label="GPS Blackout Probability")
axes[1].set_xlabel("Kp Index", color="white")
axes[1].set_ylabel("Satellite Disruption Risk", color="white")
axes[1].set_title("Kp vs Satellite Risk (color = GPS risk)", color="white")
axes[1].set_facecolor("#0d1117")
for spine in axes[1].spines.values(): spine.set_visible(False)

plt.tight_layout()
plt.savefig("../visuals/notebook_storm_analysis.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

# %% [markdown]
# ## 5. Feature Engineering Check

# %%
df_proc, scaler, encoders = preprocess(df)
print(f"Features after engineering: {len(encoders['feature_cols'])}")
print("\nTop features (sample):")
for f in encoders["feature_cols"][:15]:
    print(f"  • {f}")

# %% [markdown]
# ## 6. Correlation Analysis

# %%
key_cols = ["sunspot_number","f107_solar_flux","solar_wind_speed","imf_bz",
            "kp_index","dst_index","satellite_disruption_risk","gps_blackout_prob",
            "flare_intensity","proton_flux","solar_cycle_phase"]
corr = df[[c for c in key_cols if c in df.columns]].corr()

fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d1117")
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0,
            linewidths=0.3, linecolor="#1a1a2e", ax=ax,
            annot_kws={"size": 8, "color": "white"})
ax.set_title("Feature Correlation Matrix", color="white", fontsize=13, pad=15)
ax.tick_params(colors="white", labelsize=8)
plt.tight_layout()
plt.savefig("../visuals/notebook_correlation.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()

print("\n✅ EDA complete. All visuals saved to /visuals/")

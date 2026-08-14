import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # Optional: for Savitzky-Golay filtering

file_path_3 = "recordings\session_20260814_104425\session_metrics.csv"
df = pd.read_csv(file_path_3)
df.columns = df.columns.str.strip()

# start time at 0
df['timecode'] = df['timecode'] - df['timecode'].iloc[0]
plt.figure(figsize=(12, 6), dpi=100)
clean_cop = df['cop2bos_dist_cm'].interpolate(method='linear').bfill().ffill()
clean_xcom = df['xcom2bos_dist_cm'].interpolate(method='linear').bfill().ffill()

# Smoothing
df['cop2bos_smooth'] = savgol_filter(clean_cop, window_length=11, polyorder=2)
df['xcom2bos_smooth'] = savgol_filter(clean_xcom, window_length=11, polyorder=2)

# Ensures CoP never drops below 0
df['cop2bos_smooth'] = df['cop2bos_smooth'].clip(lower=0)
# Ensures XCoM never drops below -5
df['xcom2bos_smooth'] = df['xcom2bos_smooth'].clip(lower=-5)

plt.plot(
    df['timecode'], 
    df['cop2bos_smooth'], 
    label='COP to BOS (Smoothed)', 
    color='#F2340F', 
    linewidth=2
)

plt.plot(
    df['timecode'], 
    df['xcom2bos_smooth'], 
    label='XCoM to BOS (Smoothed)', 
    color='#F2C80F', 
    linewidth=2
)

# Semi transparent raw data points behind smoothed lines to visually compare
# plt.plot(df['timecode'], df['cop2bos_dist_cm'], color='#1f77b4', alpha=0.25, linestyle='--', label='COP (Raw)')
# plt.plot(df['timecode'], df['xcom2bos_dist_cm'], color='#ff7f0e', alpha=0.25, linestyle='--', label='XCoM (Raw)')

# Case A: CoM < XcoM < CoP < BoSmax: 
plt.fill_between(
    df['timecode'],
    df['cop2bos_smooth'],
    df['xcom2bos_smooth'],
    where=(df['xcom2bos_smooth'] >= df['cop2bos_smooth']),
    color='#2ca02c',  # Green
    alpha=0.3,         # Transparency (30%)
    interpolate=True,  
    label='case A: stable'
)

# Case B: CoM < CoP < XcoM < BoSmax
plt.fill_between(
    df['timecode'],
    df['cop2bos_smooth'],
    df['xcom2bos_smooth'],
    where=(df['xcom2bos_smooth'] < df['cop2bos_smooth']),
    color='#ffe119',  # Yellow
    alpha=0.35,        # Transparency (35%)
    interpolate=True,
    label='case B: unstable'
)

# Case C: XcoM > BoSmax
plt.fill_between(
    df['timecode'],
    df['cop2bos_smooth'],
    df['xcom2bos_smooth'],
    where=(df['xcom2bos_smooth'] < 0),
    color='#ff2617',  # Red
    alpha=0.35,        # Transparency (35%)
    interpolate=True,
    label='case C: fall iminent'
)

# 3. Customizations
plt.title('COP & XCoM Distance to Base of Support (Smoothed)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Timecode (seconds)', fontsize=12)
plt.ylabel('Distance (cm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True, fontsize=10)
plt.ticklabel_format(useOffset=False, style='plain', axis='x')

plt.tight_layout()
plt.show()

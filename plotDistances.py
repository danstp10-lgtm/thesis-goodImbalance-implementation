import pandas as pd
import matplotlib.pyplot as plt

file_path = 'recordings\session_20260813_130115\session_metrics.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# start time at 0
df['timecode'] = df['timecode'] - df['timecode'].iloc[0]
plt.figure(figsize=(12, 6), dpi=100)
plt.plot(
    df['timecode'], 
    df['cop2bos_dist_cm'], 
    label='COP to BOS Distance (cm)', 
    color='#1f77b4', 
    linewidth=1.8
)
plt.plot(
    df['timecode'], 
    df['xcom2bos_dist_cm'], 
    label='XCoM to BOS Distance (cm)', 
    color='#ff7f0e', 
    linewidth=1.8
)

plt.title('COP & XCoM Distance to Base of Support (BOS) Over Time', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Timecode (seconds)', fontsize=12)
plt.ylabel('Distance (cm)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='upper right', frameon=True, fontsize=11)
# Prevent x-axis values from rendering in scientific notation offset
plt.ticklabel_format(useOffset=False, style='plain', axis='x')
plt.tight_layout()
plt.show()

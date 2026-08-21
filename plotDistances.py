import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.patches import Polygon
from scipy.signal import savgol_filter  # Optional: for Savitzky-Golay filtering
from scipy.spatial import ConvexHull

def make_distance_plot(df,ax):
    df['timecode'] = df['timecode'] - df['timecode'].iloc[0] # start time at 0
    clean_cop = df['cop2bos_dist_cm'].interpolate(method='linear').bfill().ffill()
    clean_com = df['com2bos_dist_cm'].interpolate(method='linear').bfill().ffill()
    clean_xcom = df['xcom2bos_dist_cm'].interpolate(method='linear').bfill().ffill()

    # Smoothing
    df['cop2bos_smooth'] = savgol_filter(clean_cop, window_length=6, polyorder=2)
    df['com2bos_smooth'] = savgol_filter(clean_com, window_length=6, polyorder=2)
    df['xcom2bos_smooth'] = savgol_filter(clean_xcom, window_length=6, polyorder=2)

    # Remove CoP below 0
    df['cop2bos_smooth'] = df['cop2bos_smooth'].clip(lower=-5)
    # Remove XCoM velow -5
    df['com2bos_smooth'] = df['com2bos_smooth'].clip(lower=-5)
    df['xcom2bos_smooth'] = df['xcom2bos_smooth'].clip(lower=-5)

    ax.plot(
        df['timecode'], 
        df['cop2bos_smooth'], 
        label='b_CoP', 
        color='#F2340F', 
        linewidth=2
    )

    ax.plot(
        df['timecode'], 
        df['com2bos_smooth'], 
        label='b_CoM', 
        color='#039e00', 
        linewidth=2
    )

    ax.plot(
        df['timecode'], 
        df['xcom2bos_smooth'], 
        label='b_XCoM', 
        color='#F2C80F', 
        linewidth=2
    )

    # Semi transparent raw data points behind smoothed lines to visually compare
    # ax.plot(df['timecode'], df['cop2bos_dist_cm'], color='#1f77b4', alpha=0.25, linestyle='--', label='COP (Raw)')
    # ax.plot(df['timecode'], df['xcom2bos_dist_cm'], color='#ff7f0e', alpha=0.25, linestyle='--', label='XCoM (Raw)')

    # Case A: CoM < XcoM < CoP < BoSmax: 
    ax.fill_between(
        df['timecode'],
        df['cop2bos_smooth'],
        df['xcom2bos_smooth'],
        where=(df['xcom2bos_smooth'] >= df['cop2bos_smooth']),
        color='#2ca02c',  # Green
        alpha=0.3,         # Transparency (30%)
        interpolate=True,  
        label='A'
    )

    # Case B: CoM < CoP < XcoM < BoSmax
    ax.fill_between(
        df['timecode'],
        df['cop2bos_smooth'],
        df['xcom2bos_smooth'],
        where=(df['xcom2bos_smooth'] < df['cop2bos_smooth']),
        color='#ffe119',  # Yellow
        alpha=0.35,        # Transparency (35%)
        interpolate=True,
        label='B'
    )

    # Case C: XcoM > BoSmax
    ax.fill_between(
        df['timecode'],
        df['cop2bos_smooth'],
        df['xcom2bos_smooth'],
        where=(df['xcom2bos_smooth'] < 0),
        color='#ff2617',  # Red
        alpha=0.35,        # Transparency (35%)
        interpolate=True,
        label='C'
    )
    
    ax.set_title('CoP & XCoM Distance to Base of Support')
    ax.set_xlabel('Timecode (seconds)')
    ax.set_ylabel('Distance (cm)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

def make_CoM_path_plot(df,ax):
    com_x = df["com_x"].interpolate(method='linear').bfill().ffill()
    com_y = df["com_y"].interpolate(method='linear').bfill().ffill()
    def parse_bos(val):
        # Check if the value is missing or not a string
        if pd.isna(val) or not isinstance(val, str):
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    df['bos_parsed'] = df['bos'].apply(parse_bos)

    com_points = np.column_stack((com_x, com_y))
    hull = ConvexHull(com_points)

    # plot bos
    for bos_points in df['bos_parsed']:
        # Ensure there are at least 3 bos_points to form a closed polygon
        if len(bos_points) >= 3:
            poly = Polygon(
                bos_points, 
                closed=True, 
                facecolor='#377aed', 
                edgecolor='purple', 
                alpha=0.01  # Translucency (0 = fully transparent, 1 = opaque)
            )
            ax.add_patch(poly)
        elif len(bos_points) == 2:
            # Fallback for 2-point lines (e.g., frames 0 & 3 in your sample data)
            x_coords, y_coords = zip(*bos_points)
            ax.plot(x_coords, y_coords, color='purple', alpha=0.01, linestyle='--')

    # plot points and convex hull
    ax.scatter(com_x, com_y, c='#37ed4c', label=f'CoM', alpha=0.7, edgecolors='k')
    for simplex in hull.simplices:
        ax.plot(com_points[simplex, 0], com_points[simplex, 1], 'r--', alpha=0.8)
    ax.fill(com_points[hull.vertices, 0], com_points[hull.vertices, 1], 'red', alpha=0.15, label=f'Convex Hull Area: {hull.volume:.2f} cm^2')

    ax.set_title('CoM within BoS')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_xlim([0, 46])   # Replace with your desired [xmin, xmax]
    # ax.set_ylim([0, 18])
    ax.invert_yaxis()
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)

file_path_1 = "recordings\session_20260817_142128\session_metrics.csv"
file_path_2 = "recordings\session_20260817_142647\session_metrics.csv"
file_path_3 = "recordings\session_20260819_114549\session_metrics.csv"
file_path_4 = "recordings\session_20260819_115434\session_metrics.csv"
file_path_5 = "recordings\session_20260819_131339\session_metrics.csv"
file_path_6 = "recordings\session_20260819_131603\session_metrics.csv"
file_path_7 = "recordings\session_20260819_132657\session_metrics.csv"

file_path_8 = "recordings\session_20260821_103622\session_metrics.csv"

file_path_9 = "recordings\session_20260821_134854\session_metrics.csv"
file_path_10 = "recordings\session_20260821_135127\session_metrics.csv"
file_path_11 = "recordings\session_20260821_141750\session_metrics.csv"
# frames_path = "recordings\session_20260814_104425\\frames"
df = pd.read_csv(file_path_10)
df.columns = df.columns.str.strip()

fig, axes = plt.subplots(
        2, 1, 
        figsize=(10, 6),
        gridspec_kw={
        'width_ratios': [1],
        'height_ratios': [1,2]
        }
    )
make_distance_plot(df,axes[0])
make_CoM_path_plot(df,axes[1])
plt.tight_layout()
plt.show()

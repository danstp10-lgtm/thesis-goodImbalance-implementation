import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ast
from matplotlib.patches import Polygon
from scipy.signal import savgol_filter  
from scipy.spatial import ConvexHull

def make_distance_plot(df,ax):
    df['timecode'] = df['timecode'] - df['timecode'].iloc[0] # start time at 0
    clean_cop = df['cop2bos_dist_cm'].interpolate(method='linear').bfill().ffill()
    # clean_com = df['com2bos_dist_cm'].interpolate(method='linear').bfill().ffill()
    clean_xcom = df['xcom2bos_dist_cm'].interpolate(method='linear').bfill().ffill()

    # Smoothing
    df['cop2bos_smooth'] = savgol_filter(clean_cop, window_length=6, polyorder=2)
    # df['com2bos_smooth'] = savgol_filter(clean_com, window_length=6, polyorder=2)
    df['xcom2bos_smooth'] = savgol_filter(clean_xcom, window_length=6, polyorder=2)

    # Remove CoP below 0, cases when there is no boundry
    df['cop2bos_smooth'] = df['cop2bos_smooth'].clip(lower=0)
    # Remove XCoM velow -5, too far away from TSP's
    # df['com2bos_smooth'] = df['com2bos_smooth'].clip(lower=-5)
    df['xcom2bos_smooth'] = df['xcom2bos_smooth'].clip(lower=-5)

    # Plot distances
    ax.plot(df['timecode'], df['cop2bos_smooth'], label='b_CoP', color='#F2340F', linewidth=2)
    # ax.plot(df['timecode'], df['com2bos_smooth'], label='b_CoM', color='#039e00', linewidth=2)
    ax.plot(df['timecode'], df['xcom2bos_smooth'], label='b_XCoM', color='#F2C80F', linewidth=2)

    # Semi transparent raw data points
    # ax.plot(df['timecode'], df['cop2bos_dist_cm'], color='#1f77b4', alpha=0.25, linestyle='--', label='COP (Raw)')
    # ax.plot(df['timecode'], df['xcom2bos_dist_cm'], color='#ff7f0e', alpha=0.25, linestyle='--', label='XCoM (Raw)')

    # Case A: CoM < XcoM < CoP < BoSmax: 
    ax.fill_between(
        df['timecode'],
        df['cop2bos_smooth'],
        df['xcom2bos_smooth'],
        where=(df['xcom2bos_smooth'] >= df['cop2bos_smooth']),
        color='#2ca02c',  # Green
        alpha=0.35,         # Transparency (35%)
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
    
    # Legend
    ax.set_title('CoP & XCoM Distance to Base of Support')
    ax.set_xlabel('Timecode (seconds)')
    ax.set_ylabel('Distance (cm)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.ticklabel_format(useOffset=False, style='plain', axis='x')

def make_CoM_path_plot(df,ax):

    # filtered_data = df[(df["timecode"] >= 16.7) & (df["timecode"] <= 36.5)][["timecode", "com_x", "com_y", "bos"]]
    # com_x = df["com_x"].interpolate(method='linear').bfill().ffill() # all points
    # com_y = df["com_y"].interpolate(method='linear').bfill().ffill() # all points
    df['timecode'] = df['timecode'] - df['timecode'].iloc[0]
    com_x_upright = df[(df['timecode'] >= 16.7) & (df['timecode'] <= 36.5)][["com_x"]].interpolate(method='linear').bfill().ffill()
    com_y_upright = df[(df['timecode'] >= 16.7) & (df['timecode'] <= 36.5)][["com_y"]].interpolate(method='linear').bfill().ffill()
    com_x_regular = df[(df['timecode'] >= 37) & (df['timecode'] <= 85)][["com_x"]].interpolate(method='linear').bfill().ffill()
    com_y_regular = df[(df['timecode'] >= 37) & (df['timecode'] <= 85)][["com_y"]].interpolate(method='linear').bfill().ffill()


    def parse_bos(val):
        if pd.isna(val) or not isinstance(val, str):
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []

    df['bos_parsed'] = df['bos'].apply(parse_bos)

    # com_points_all = np.column_stack((com_x, com_y))
    com_points_upright = np.column_stack((com_x_upright, com_y_upright))
    com_points_regular = np.column_stack((com_x_regular, com_y_regular))
    # hull_all = ConvexHull(com_points_all)
    hull_upright = ConvexHull(com_points_upright)
    hull_regular = ConvexHull(com_points_regular)

    # Plot bos
    for bos_points in df['bos_parsed']:
        if len(bos_points) >= 3: # minimum three points for a polygon
            poly = Polygon(
                bos_points, 
                closed=True, 
                facecolor='#377aed', 
                edgecolor='purple', 
                alpha=0.01  # 0 = fully transparent, 1 = opaque
            )
            ax.add_patch(poly)
        elif len(bos_points) == 2: # fallback for less than 3 points
            x_coords, y_coords = zip(*bos_points)
            ax.plot(x_coords, y_coords, color='purple', alpha=0.01, linestyle='--')

    # Plot points and convex hull
    # ax.scatter(com_x, com_y, c='#37ed4c', label=f'CoM', alpha=0.7, edgecolors='k')
    # for simplex in hull_all.simplices:
    #     ax.plot(com_points_all[simplex, 0], com_points_all[simplex, 1], 'r--', alpha=0.8)
    # ax.fill(com_points_all[hull_all.vertices, 0], com_points_all[hull_all.vertices, 1], 'red', alpha=0.15, label=f'Convex Hull Area: {hull_all.volume:.2f} cm^2')
    ax.scatter(com_x_upright, com_y_upright, c='#8feb34', label=f'CoM upright', alpha=0.7, edgecolors='k')
    ax.scatter(com_x_regular, com_y_regular, c='#eb4034', label=f'CoM regular', alpha=0.7, edgecolors='k')
    for simplex in hull_upright.simplices:
        ax.plot(com_points_upright[simplex, 0], com_points_upright[simplex, 1], 'y--', alpha=0.8)
    ax.fill(com_points_upright[hull_upright.vertices, 0], com_points_upright[hull_upright.vertices, 1], 'green', alpha=0.15, label=f'Convex Hull Area: {hull_upright.volume:.2f} cm^2')

def make_CoM_path_segregate_plot(df,ax):
    # Define condition
    condition = (
        (df['xcom2bos_dist_cm'] >= df['cop2bos_dist_cm']) | 
        (df['xcom2bos_dist_cm'] < df['cop2bos_dist_cm'])
        ) #& (~(df['xcom2bos_dist_cm'] < 0))    
    block_ids = condition.ne(condition.shift()).cumsum()
    true_segments = []
    time_period = []
    for _, group in df.groupby(block_ids):
        com_x = group['com_x']
        com_y = group['com_y']
        if condition.loc[group.index[0]]:
            valid_group = group.dropna(subset=['com_x', 'com_y']) # drop nans
            if len(valid_group) >= 3: # need min 3 points for hull
                t_start = valid_group['timecode'].iloc[0]
                t_end = valid_group['timecode'].iloc[-1]
                
                true_segments.append({
                    'x': valid_group['com_x'].to_numpy(),
                    'y': valid_group['com_y'].to_numpy(),
                    'time': valid_group['timecode'].to_numpy(),
                    'start_time': t_start,
                    'finish_time': t_end,
                    'duration': t_end - t_start
                })
    def parse_bos(val):
        if pd.isna(val) or not isinstance(val, str):
            return []
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    df['bos_parsed'] = df['bos'].apply(parse_bos)

    # Plot bos
    for bos_points in df['bos_parsed']:
        if len(bos_points) >= 3: # minimum three points for a polygon
            poly = Polygon(
                bos_points, 
                closed=True, 
                fill=False, 
                edgecolor='purple', 
                alpha=0.01  # 0 = fully transparent, 1 = opaque
            )
            ax.add_patch(poly)
        elif len(bos_points) == 2: # fallback for less than 3 points
            x_coords, y_coords = zip(*bos_points)
            ax.plot(x_coords, y_coords, color='purple', alpha=0.01, linestyle='--')

    # Plot segmented balance periods
    colors = plt.cm.tab10.colors  # Uses 10 distinct colors
    for i, seg in enumerate(true_segments):
        x_data, y_data = seg['x'], seg['y']        
        segment_color = colors[i % len(colors)]
        com_points = np.column_stack((x_data, y_data))
        hull = ConvexHull(com_points)

        ax.scatter(x_data, y_data, color=segment_color, alpha=0.3, edgecolors='k', linewidths=0.5)
        # Fill the convex hull area (hull.volume gives area in 2D)
        ax.fill(
            com_points[hull.vertices, 0], 
            com_points[hull.vertices, 1], 
            color=segment_color, 
            alpha=0.15,
            label=f'Seg {i+1} ({seg['start_time']:.1f}s - {seg['finish_time']:.1f}s Area: {hull.volume:.2f} cm²'
        )
        
        # Draw hull contour outline
        for simplex in hull.simplices:
            ax.plot(
                com_points[simplex, 0], 
                com_points[simplex, 1], 
                color=segment_color, 
                linestyle='--', 
                alpha=0.9
            )

    # Legend
    ax.set_title('CoM within BoS')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_xlim([0, 46])  
    # ax.set_ylim([0, 18])
    ax.invert_yaxis()
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)


file_path="recordings\session_20260817_142647\session_metrics.csv" # custom upright example
file_path2="recordings\session_20260819_131339\session_metrics.csv" # custom regular
file_path3="recordings\session_20260824_111952\session_metrics.csv" # built-in chair
file_path4="recordings\session_20260824_111023\session_metrics.csv" # built-in regular

# frames_path = "recordings\session_20260814_104425\\frames"
df = pd.read_csv(file_path4)
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
make_CoM_path_segregate_plot(df,axes[1])
plt.tight_layout()
plt.show()

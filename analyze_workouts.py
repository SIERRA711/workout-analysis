import pandas as pd
import argparse
import sys
from datetime import timedelta
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import io
import base64
import os
import numpy as np
import re
import math

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        # Parse dates
        df['start_time'] = pd.to_datetime(df['start_time'], format='%d %b %Y, %H:%M')
        df['end_time'] = pd.to_datetime(df['end_time'], format='%d %b %Y, %H:%M')
        
        # Ensure numeric columns are numeric
        numeric_cols = ['weight_lbs', 'reps', 'distance_miles', 'duration_seconds', 'rpe']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_muscle_group(exercise_name):
    ex = exercise_name.lower()
    patterns = {
        'Legs': r'squat|leg press|leg curl|calf|leg extension|hamstring|quad|glute|lunge|deadlift',
        'Pull': r'row|pulldown|pull.?up|chin.?up|lat|posterior|back|curl|bicep|face pull',
        'Push': r'bench|press|push.?up|dip|chest|shoulder|tricep|fly|extension|ohp|pec|delt',
        'Core': r'plank|crunch|sit.?up|ab|core|russian twist|hollow|l.sit|hanging',
        'Cardio': r'run|jog|sprint|cycle|bike|rowing|elliptical|cardio|hiit|interval'
    }
    
    for group, pattern in patterns.items():
        if re.search(pattern, ex):
            return group
    return 'Other'

def preprocess_data(df):
    # Calculate E1RM (Epley Formula): w * (1 + r/30)
    mask = df['weight_lbs'].notna() & df['reps'].notna() & (df['reps'] > 0)
    df.loc[mask, 'e1rm'] = df.loc[mask, 'weight_lbs'] * (1 + df.loc[mask, 'reps'] / 30)
    
    # Calculate Volume
    df.loc[mask, 'volume'] = df.loc[mask, 'weight_lbs'] * df.loc[mask, 'reps']
    
    # Assign Muscle Groups
    df['muscle_group'] = df['exercise_title'].apply(get_muscle_group)
    
    return df

def analyze_consistency(df):
    workouts = df[['title', 'start_time']].drop_duplicates()
    workouts.set_index('start_time', inplace=True)
    workout_dates = sorted(workouts.index)
    
    # Resample to weekly
    weekly_frequency = workouts.resample('W').size()
    
    # Fill missing weeks with 0
    if not weekly_frequency.empty:
        full_idx = pd.date_range(start=weekly_frequency.index.min(), end=weekly_frequency.index.max(), freq='W')
        weekly_frequency = weekly_frequency.reindex(full_idx, fill_value=0)
    
    recent_weeks = weekly_frequency.tail(4)
    avg_freq = recent_weeks.mean() if not recent_weeks.empty else 0
    total_workouts_last_4_weeks = recent_weeks.sum()
    
    # Consistency Score Calculation (0-100)
    intervals = []
    if len(workout_dates) > 1:
        for i in range(1, len(workout_dates)):
            interval = (workout_dates[i] - workout_dates[i-1]).days
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        std_dev = np.std(intervals)
        
        # Base score starts at 100 and penalties applied
        # Ideal interval is ~2-3 days (frequency of 2-3x/week)
        # Penalty for irregularity (std_dev) and too long gaps (avg_interval)
        
        regularity_penalty = min(50, std_dev * 5) # Penalize inconsistent gaps
        frequency_penalty = 0
        
        if avg_interval > 5: # If avg gap > 5 days, penalty
             frequency_penalty = (avg_interval - 5) * 5
        
        score = max(0, min(100, 100 - regularity_penalty - frequency_penalty))
    else:
        score = 0
        if len(workout_dates) == 1: score = 50 # Just started
    
    return {
        'avg_weekly_freq': avg_freq,
        'last_4_weeks_count': total_workouts_last_4_weeks,
        'weekly_trend': weekly_frequency,
        'consistency_score': int(score)
    }

def analyze_volume(df):
    # Group by week and sum volume
    df_vol = df.set_index('start_time')
    weekly_volume = df_vol['volume'].resample('W').sum()
    
    if not weekly_volume.empty:
        full_idx = pd.date_range(start=weekly_volume.index.min(), end=weekly_volume.index.max(), freq='W')
        weekly_volume = weekly_volume.reindex(full_idx, fill_value=0)
        
    return weekly_volume

def analyze_muscle_distribution(df):
    # Total volume by muscle group
    muscle_vol = df.groupby('muscle_group')['volume'].sum()
    # Remove 'Other' if it's too small or keep it
    return muscle_vol

def analyze_volume_heatmap(df):
    # Aggregate volume by Week and Muscle Group
    df['week_start'] = df['start_time'].dt.to_period('W').apply(lambda r: r.start_time)
    heatmap_data = df.groupby(['week_start', 'muscle_group'])['volume'].sum().unstack(fill_value=0)
    
    # Ensure all main groups exist
    for group in ['Push', 'Pull', 'Legs', 'Core', 'Cardio']:
        if group not in heatmap_data.columns:
            heatmap_data[group] = 0
            
    # Keep only last 12 weeks for readability
    heatmap_data = heatmap_data.tail(12)
    return heatmap_data

def analyze_progress(df, top_n=7, months=6):
    top_exercises = df['exercise_title'].value_counts().head(top_n).index.tolist()
    progress_report = []
    
    for exercise in top_exercises:
        ex_data = df[df['exercise_title'] == exercise].dropna(subset=['e1rm'])
        if ex_data.empty:
            continue
            
        ex_data = ex_data.sort_values('start_time')
        workout_maxes = ex_data.groupby('start_time')['e1rm'].max()
        
        if len(workout_maxes) < 2:
            continue
            
        current_max = workout_maxes.iloc[-1]
        
        # Compare to X months ago
        past_date = ex_data['start_time'].max() - timedelta(days=30*months)
        older_data = workout_maxes[workout_maxes.index <= past_date]
        
        # If no data exactly X months ago, find the oldest available in that window or just oldest record
        prev_max = older_data.iloc[-1] if not older_data.empty else workout_maxes.iloc[0]
        change = current_max - prev_max
        roi = "Growing" if change > 0 else "Stagnant/Regressing"
        
        progress_report.append({
            'exercise': exercise,
            'current_e1rm': current_max,
            'change': change,
            'status': roi,
            'history': workout_maxes 
        })
        
    return progress_report

def detect_prs(df, days_lookback=30):
    recent_prs = []
    exercises = df['exercise_title'].unique()
    cutoff_date = df['start_time'].max() - timedelta(days=days_lookback)
    
    for ex in exercises:
        ex_data = df[df['exercise_title'] == ex].dropna(subset=['weight_lbs', 'e1rm'])
        if ex_data.empty:
            continue
            
        # 1. Check E1RM PRs (Strength)
        e1rm_maxes = ex_data.groupby('start_time')['e1rm'].max().sort_index()
        if len(e1rm_maxes) >= 1:
            recent_workouts = e1rm_maxes[e1rm_maxes.index >= cutoff_date]
            historical_data = e1rm_maxes[e1rm_maxes.index < cutoff_date]
            
            if not recent_workouts.empty:
                recent_max = recent_workouts.max()
                recent_max_date = recent_workouts.idxmax()
                
                is_pr = False
                improvement = 0
                
                if not historical_data.empty:
                    historical_max = historical_data.max()
                    if recent_max > historical_max:
                        is_pr = True
                        improvement = recent_max - historical_max
                elif recent_max == e1rm_maxes.max() and recent_max > 0: 
                     is_pr = True 
                
                if is_pr:
                    recent_prs.append({
                        'type': 'E1RM',
                        'exercise': ex,
                        'value': recent_max,
                        'date': recent_max_date,
                        'improvement': improvement
                    })

        # 2. Check Absolute Weight PRs (Heaviest Lift)
        weight_maxes = ex_data.groupby('start_time')['weight_lbs'].max().sort_index()
        if len(weight_maxes) >= 1:
            recent_workouts = weight_maxes[weight_maxes.index >= cutoff_date]
            historical_data = weight_maxes[weight_maxes.index < cutoff_date]
            
            if not recent_workouts.empty:
                recent_max = recent_workouts.max()
                recent_max_date = recent_workouts.idxmax()
                
                is_pr = False
                improvement = 0
                
                if not historical_data.empty:
                    historical_max = historical_data.max()
                    if recent_max > historical_max:
                        is_pr = True
                        improvement = recent_max - historical_max
                elif recent_max == weight_maxes.max() and recent_max > 0:
                     is_pr = True
                
                if is_pr:
                    recent_prs.append({
                        'type': 'Weight',
                        'exercise': ex,
                        'value': recent_max,
                        'date': recent_max_date,
                        'improvement': improvement
                    })
                 
    return recent_prs

def analyze_splits(df):
    workouts = df[['title', 'start_time']].drop_duplicates()
    split_dist = workouts['title'].value_counts()
    return split_dist

def plot_to_base64(fig):
    buf = io.BytesIO()
    # Save with transparent background to blend with CSS
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_plots(consistency_data, progress_data, volume_data, muscle_dist, heatmap_data):
    # Enable Dark Mode for Matplotlib
    plt.style.use('dark_background')
    
    # Custom colors for dark mode
    c_bar = '#2ecc71' # Green
    c_line = '#3498db' # Blue
    c_fill = '#3498db'
    c_radar = '#e74c3c' # Red
    
    plots = {}
    
    # 1. Consistency Plot
    if not consistency_data['weekly_trend'].empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        # Ensure figure bg is transparent (handled in savefig) but set axes facecolor just in case
        ax.set_facecolor('none') 
        fig.patch.set_alpha(0.0)
        
        data_to_plot = consistency_data['weekly_trend'].tail(12)
        dates = [d.strftime('%Y-%m-%d') for d in data_to_plot.index]
        bars = ax.bar(dates, data_to_plot.values, color=c_bar, alpha=0.8)
        
        ax.set_title('Weekly Workout Frequency (Last 12 Weeks)', color='white')
        ax.set_ylabel('Workouts', color='white')
        
        # Style ticks
        ax.tick_params(axis='x', colors='white', rotation=45)
        ax.tick_params(axis='y', colors='white')
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.spines['left'].set_color('#888')
        
        ax.bar_label(bars, color='white')
        plt.tight_layout()
        plots['consistency'] = plot_to_base64(fig)
    
    # 2. Progress Plot
    if progress_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(progress_data)))
        
        for i, item in enumerate(progress_data[:5]): 
            hist = item['history']
            ax.plot(hist.index, hist.values, marker='o', label=item['exercise'], color=colors[i], linewidth=2)
            
        ax.set_title('Estimated 1 Rep Max (E1RM) Trends (6 Months)', color='white')
        ax.set_ylabel('Weight (lbs)', color='white')
        
        legend = ax.legend(facecolor='#333', edgecolor='none', labelcolor='white')
        
        ax.tick_params(colors='white')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.2)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.spines['left'].set_color('#888')
        
        plt.tight_layout()
        plots['progress'] = plot_to_base64(fig)
        
    # 3. Volume Plot
    if not volume_data.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        data_to_plot = volume_data.tail(12) # Last 12 weeks
        dates = [d.strftime('%Y-%m-%d') for d in data_to_plot.index]
        
        ax.fill_between(dates, data_to_plot.values, color=c_fill, alpha=0.3)
        ax.plot(dates, data_to_plot.values, color=c_line, marker='o')
        
        ax.set_title('Weekly Volume Load (lbs * sets * reps)', color='white')
        ax.set_ylabel('Volume (lbs)', color='white')
        ax.tick_params(colors='white')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle=':', alpha=0.2)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.spines['left'].set_color('#888')
        
        plt.tight_layout()
        plots['volume'] = plot_to_base64(fig)
        
    # 4. Muscle Balance Radar Plot
    if not muscle_dist.empty:
        categories = ['Push', 'Pull', 'Legs', 'Core', 'Cardio']
        values = []
        total_vol = muscle_dist.sum()
        
        for cat in categories:
            val = muscle_dist.get(cat, 0)
            values.append((val / total_vol) * 100 if total_vol > 0 else 0)
            
        values += values[:1] # Close the loop
        angles = [n / float(len(categories)) * 2 * math.pi for n in range(len(categories))]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        # Style Polar Axis
        ax.patch.set_alpha(0.0)
        ax.grid(color='#555')
        ax.spines['polar'].set_color('#555')
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=c_radar)
        ax.fill(angles, values, c_radar, alpha=0.4)
        
        plt.xticks(angles[:-1], categories, color='white', size=11)
        ax.tick_params(axis='y', colors='#aaa')
        
        try:
           ax.set_rlabel_position(0)
           ax.set_rticks([20, 40, 60, 80])
        except:
           pass
           
        ax.set_title('Muscle Group Balance (%)', y=1.1, color='white')
        plt.tight_layout()
        plots['radar'] = plot_to_base64(fig)
        
    # 5. Volume Heatmap
    if not heatmap_data.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('none')
        fig.patch.set_alpha(0.0)
        
        data_vals = heatmap_data.values.T 
        
        # Use a dark-friendly colormap (e.g., magma, inferno, or just Reds with alpha)
        im = ax.imshow(data_vals, cmap='inferno', aspect='auto')
        
        # Labels
        ax.set_yticks(np.arange(len(heatmap_data.columns)))
        ax.set_yticklabels(heatmap_data.columns, color='white')
        
        weeks = [d.strftime('%b %d') for d in heatmap_data.index]
        ax.set_xticks(np.arange(len(weeks)))
        ax.set_xticklabels(weeks, rotation=45, ha='right', color='white')
        
        ax.set_title('Volume Heatmap (Intensity per Muscle Group)', color='white')
        
        # Style colorbar
        cbar = plt.colorbar(im, label='Volume')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        cbar.set_label('Volume', color='white')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#888')
        ax.spines['left'].set_color('#888')
        
        plt.tight_layout()
        plots['heatmap'] = plot_to_base64(fig)
        
    return plots

def generate_html_report(consistency, progress, splits, volume, prs, plots, filename="workout_report.html"):
    
    score_color = "#2ecc71" if consistency['consistency_score'] > 80 else "#f1c40f" if consistency['consistency_score'] > 50 else "#e74c3c"
    
    # CSS Styling - Dark Mode
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #1a1a1a; color: #e0e0e0; }
        .container { max-width: 1100px; margin: 0 auto; background: #2d2d2d; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h1, h2, h3 { color: #ffffff; }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 2px solid #404040; padding-bottom: 20px; position: relative;}
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .card { background: #363636; border: 1px solid #404040; border-radius: 6px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .metric-big { font-size: 2.5em; font-weight: bold; color: #3498db; }
        .metric-label { color: #b0b0b0; text-transform: uppercase; font-size: 0.85em; letter-spacing: 1px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #4d4d4d; color: #e0e0e0; }
        th { background-color: #404040; color: #ffffff; }
        .growth { color: #2ecc71; font-weight: bold; }
        .stagnant { color: #e74c3c; }
        .alert { background-color: #2c3e50; padding: 15px; border-left: 5px solid #f39c12; margin-bottom: 20px; border-radius: 4px; }
        .alert h3 { margin-top: 0; color: #f39c12; }
        img { max-width: 100%; height: auto; border-radius: 4px; filter: brightness(0.9); }
        li { margin-bottom: 5px; color: #e0e0e0; }
        .tag { padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold; color: white;}
        .tag-e1rm { background-color: #d35400; }
        .tag-weight { background-color: #c0392b; }
        .score-circle { 
            width: 100px; height: 100px; border-radius: 50%; display: flex; align-items: center; justify-content: center; 
            font-size: 2em; font-weight: bold; color: white; margin: 0 auto; box-shadow: 0 0 15px rgba(0,0,0,0.3);
        }
        p { color: #cccccc; }
        
        .print-btn {
            position: absolute; top: 0; right: 0;
            background-color: #4a69bd; color: white; border: none; padding: 10px 20px;
            border-radius: 5px; cursor: pointer; font-size: 1rem; transition: background 0.2s;
        }
        .print-btn:hover { background-color: #6a89cc; }
        @media print {
            .print-btn { display: none; }
            body { background-color: white; color: black; }
            .container { box-shadow: none; max-width: 100%; padding: 0; background: white; }
            .card { break-inside: avoid; border: 1px solid #ddd; background: white; box-shadow: none; }
            th { background-color: #f8f9fa; color: black; }
            h1, h2, h3 { color: black; }
            p, li, td { color: black; }
            img { filter: none; }
        }
    </style>
    """
    
    # Progress Rows
    progress_rows = ""
    for p in progress:
        status_class = "growth" if p['change'] > 0 else "stagnant"
        sign = "+" if p['change'] > 0 else ""
        progress_rows += f"""
        <tr>
            <td>{p['exercise']}</td>
            <td>{p['current_e1rm']:.1f} lbs</td>
            <td class="{status_class}">{sign}{p['change']:.1f} lbs</td>
        </tr>
        """
        
    # Split Rows
    split_rows = ""
    for title, count in splits.head(5).items():
        split_rows += f"<tr><td>{title}</td><td>{count}</td></tr>"

    # PR Alerts HTML
    pr_html = ""
    if prs:
        pr_items = ""
        prs = sorted(prs, key=lambda x: x['date'], reverse=True)
        for pr in prs:
            improvement_str = f"(+{pr['improvement']:.1f} lbs)" if 'improvement' in pr and pr['improvement'] > 0 else ""
            if pr['type'] == 'E1RM':
                tag = '<span class="tag tag-e1rm">🔥 Strength</span>'
                val_str = f"{pr['value']:.1f} lbs (est)"
            else:
                tag = '<span class="tag tag-weight">🏆 Heavy</span>'
                val_str = f"{pr['value']:.1f} lbs"
            pr_items += f"<li>{tag} <strong>{pr['exercise']}</strong>: {val_str} {improvement_str} at {pr['date'].strftime('%d %b')}</li>"
        
        pr_html = f"""
        <div class="alert">
            <h3>🚀 Recent PRs (Last 30 Days)</h3>
            <ul style="list-style: none; padding-left: 0;">
                {pr_items}
            </ul>
        </div>
        """
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Workout Analysis Report</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <button class="print-btn" onclick="window.print()">🖨️ Print to PDF</button>
                <h1>🏋️ Workout Analysis Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%d %b %Y')}</p>
            </div>
            
            {pr_html}
            
            <div class="grid">
                <div class="card" style="text-align: center;">
                    <div class="metric-label">Consistency Score</div>
                    <div class="score-circle" style="background-color: {score_color}">
                        {consistency['consistency_score']}
                    </div>
                    <p style="margin-top:10px; font-size:0.9em; color:#7f8c8d;">Based on regularity</p>
                </div>
                <div class="card">
                    <div class="metric-label">Avg Weekly Frequency</div>
                    <div class="metric-big">{consistency['avg_weekly_freq']:.1f}</div>
                    <p>Last 4 Weeks: {consistency['last_4_weeks_count']} workouts</p>
                </div>
                 <div class="card">
                    <div class="metric-label">Avg Weekly Volume</div>
                    <div class="metric-big">{volume.tail(4).mean():.0f} lbs</div>
                    <p>Last 4 weeks Avg</p>
                </div>
            </div>

            <!-- Full Width Charts -->
            <div class="card">
                <h2>📊 Weekly Volume Trend</h2>
                {f'<img src="data:image/png;base64,{plots.get("volume")}" />' if 'volume' in plots else '<p>No data</p>'}
            </div>
            
            <br>

            <div class="card">
                <h2>� Consistency Trend</h2>
                {f'<img src="data:image/png;base64,{plots.get("consistency")}" />' if 'consistency' in plots else '<p>No data</p>'}
            </div>

            <br>
            
            <div class="grid">
                 <div class="card">
                    <h2>�️ Muscle Balance</h2>
                     {f'<img src="data:image/png;base64,{plots.get("radar")}" />' if 'radar' in plots else '<p>No data</p>'}
                </div>
                <div class="card">
                    <h3>Split Distribution</h3>
                    <table>
                        <thead>
                            <tr><th>Workout</th><th>Count</th></tr>
                        </thead>
                        <tbody>
                            {split_rows}
                        </tbody>
                    </table>
                </div>
            </div>
            
             <br>
            
            <div class="card">
                <h2>🌡️ Volume Heatmap</h2>
                {f'<img src="data:image/png;base64,{plots.get("heatmap")}" />' if 'heatmap' in plots else '<p>No data</p>'}
            </div>
            
            <br>
            
            <div class="card">
                <h2>💪 Strength Progress (6 Months)</h2>
                 {f'<img src="data:image/png;base64,{plots.get("progress")}" />' if 'progress' in plots else '<p>No data</p>'}
            </div>

            <br>

            <div class="card">
                <h3>Top Movers (6 Months)</h3>
                <table>
                    <thead>
                        <tr><th>Exercise</th><th>Current Max</th><th>Change</th></tr>
                    </thead>
                    <tbody>
                        {progress_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report generated: {os.path.abspath(filename)}")
    
    if sys.platform == 'win32':
        os.startfile(filename)


def main():
    parser = argparse.ArgumentParser(description='Analyze workout data CSV')
    parser.add_argument('--input', type=str, required=True, help='Path to workout_data.csv')
    args = parser.parse_args()
    
    print(f"Analyzing: {args.input}...")
    df = load_data(args.input)
    df = preprocess_data(df)
    
    consistency = analyze_consistency(df)
    progress = analyze_progress(df, top_n=7, months=6) 
    splits = analyze_splits(df)
    volume = analyze_volume(df)
    prs = detect_prs(df, days_lookback=30) 
    muscle_dist = analyze_muscle_distribution(df)
    heatmap_data = analyze_volume_heatmap(df)
    
    print("Generating visualizations...")
    plots = generate_plots(consistency, progress, volume, muscle_dist, heatmap_data)
    
    generate_html_report(consistency, progress, splits, volume, prs, plots)

if __name__ == "__main__":
    main()

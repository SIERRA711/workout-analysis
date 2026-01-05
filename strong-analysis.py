import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import re
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from sklearn.cluster import KMeans
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import matplotlib.cm as cm
from matplotlib.colors import Normalize


from typing import Dict, Optional, Tuple
import logging
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# Set global style for plots with modern, professional look
plt.style.use('fivethirtyeight')
sns.set_context("talk")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    # Update units and column mappings
    BODYWEIGHT = 185  # in lbs
    WEIGHT_COLUMN = "Weight (kg)"
    REPS_COLUMN = "Reps"
    DATE_COLUMN = "Date"
    EXERCISE_COLUMN = "Exercise Name"
    DURATION_COLUMN = "Duration (sec)"
    KG_TO_LBS = 2.20462
    DEFAULT_DELIMITER = ';'
    REPS_COLUMN = "Reps"
    VOLUME_COLUMN = "Volume"
    # Added missing constants
    MIN_CLUSTER_SIZE = 3
    MAX_CLUSTERS = 5
    CLUSTER_RANDOM_STATE = 42
    COLOR_PALETTE = "tab10"

# Add missing RECENCY_THRESHOLD constant
RECENCY_THRESHOLD = 90

def expand_muscle_groups(data):
    """Automatically categorize exercises with improved muscle group classification"""
    # Get unique exercises and their metrics
    exercise_stats = data.groupby('Exercise Name').agg(
        avg_weight=('Weight', 'mean'),
        avg_reps=('Reps', 'mean'),
        total_volume=('Volume', 'sum')
    ).reset_index()

    # Enhanced muscle groups with expanded keywords for better categorization
    base_groups = {
        'Legs': r'squat|leg press|leg curl|calf|leg extension|hamstring|quad|glute|lunge',
        'Pull': r'deadlift|row|pulldown|pull.?up|chin.?up|lat|posterior|back|curl|bicep',
        'Push': r'bench|press|push.?up|dip|chest|shoulder|tricep|fly|extension|ohp|pec|delt',
        'Core': r'plank|crunch|sit.?up|ab|core|russian twist|hollow|l.sit|hanging',
        'Cardio': r'run|jog|sprint|cycle|bike|rowing|elliptical|cardio|hiit|interval'
    }

    # First pass: Keyword matching
    categorized = {}
    for ex in exercise_stats['Exercise Name']:
        ex_lower = ex.lower()
        for group, pattern in base_groups.items():  # Fix: use base_groups, not muscle_patterns
            if re.search(pattern, ex_lower):
                categorized[ex] = group
                break
        else:
            categorized[ex] = None

    # [Improved Clustering]
    uncategorized = [ex for ex, grp in categorized.items() if grp is None]
    if uncategorized:
        features = exercise_stats[exercise_stats['Exercise Name'].isin(uncategorized)]
        features = features[['avg_weight', 'avg_reps', 'total_volume']].fillna(0)

        # Better feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine optimal clusters
        if len(uncategorized) >= Config.MIN_CLUSTER_SIZE:
            silhouette_scores = []
            max_clusters = min(Config.MAX_CLUSTERS, len(uncategorized)-1)

            for n in range(2, max_clusters+1):
                kmeans = KMeans(n_clusters=n, random_state=Config.CLUSTER_RANDOM_STATE)
                labels = kmeans.fit_predict(features_scaled)
                if len(set(labels)) > 1:
                    silhouette_scores.append(silhouette_score(features_scaled, labels))
                else:
                    silhouette_scores.append(-1)

            best_n = np.argmax(silhouette_scores) + 2  # Offset for starting at 2 clusters
            kmeans = KMeans(n_clusters=best_n, random_state=Config.CLUSTER_RANDOM_STATE)
            labels = kmeans.fit_predict(features_scaled)

            # Map clusters to existing groups
            cluster_map = {i: list(base_groups.keys())[i % len(base_groups)]
                         for i in range(best_n)}

            for idx, ex in enumerate(uncategorized):
                categorized[ex] = cluster_map[labels[idx]]

    # [Add error handling for variation factors]
    variation_patterns = {
        r'single|one.(arm|leg)': 0.5,
        r'(arm|leg).alternat': 0.8,
        r'dual|both.(arm|leg)': 1.0,
        r'decline|incline': 1.0,
        r'wide|narrow|close': 1.0
    }

    # Create base exercise names and variation factors
    exercise_variations = {}
    for ex in exercise_stats['Exercise Name']:
        base_name = ex
        variation_factor = 1.0  # Default to bilateral version

        for pattern, factor in variation_patterns.items():
            if re.search(pattern, ex, re.IGNORECASE):
                base_name = re.sub(pattern, '', ex, flags=re.IGNORECASE).strip()
                variation_factor = max(factor, 0.1)  # Prevent division by zero
                break

        exercise_variations[ex] = {
            'base_exercise': base_name,
            'variation_factor': variation_factor
        }

    return {
        ex: {
            'muscle_group': grp,
            **exercise_variations[ex]
        } for ex, grp in categorized.items()
    }

def normalize_exercise_name(exercise_name):
    """Normalize exercise names by removing variations with improved patterns"""
    patterns = [
        r'\(.*?\)',             # Remove parentheses content
        r'\b(single|one|both|arm|leg|dumbbell|barbell|machine|cable|kettle ?bell)\b',
        r'\b(incline|decline|flat)\b',  # Remove position qualifiers
        r'\b(wide|narrow|close)\b',     # Remove grip qualifiers
        r'\d+\s*(kg|lb|pounds|lbs)',    # Remove weights
        r'\s+',                  # Replace multiple spaces
    ]
    normalized = exercise_name.lower()
    for pattern in patterns:
        normalized = re.sub(pattern, ' ', normalized, flags=re.IGNORECASE)
    return ' '.join(normalized.split()).strip()  # Remove extra spaces

def enhance_analysis(data):
    """Enhance dataset with normalized exercise names and additional metrics"""
    mapping = expand_muscle_groups(data)

    additional_patterns = [
        r'\(.*?\)',  # Remove parentheses content
        r'\b(barbell|dumbbell|machine|cable|kettlebell|smith)\b',
        r'\d+\s*(kg|lb)',
        r'\s-\s.*'  # Remove exercise variations after dash
    ]
    
    data['Normalized Exercise'] = (
        data[Config.EXERCISE_COLUMN]
        .str.lower()
        .replace(additional_patterns, '', regex=True)
        .str.strip()
    )
    
    data['Muscle Group'] = data['Exercise Name'].map(lambda x: mapping.get(x, {}).get('muscle_group', 'Other'))
    data['Base Exercise'] = data['Exercise Name'].map(lambda x: mapping.get(x, {}).get('base_exercise', x))
    data['Variation Factor'] = data['Exercise Name'].map(lambda x: mapping.get(x, {}).get('variation_factor', 1.0))

    # Adjust weights for variations
    data['Adjusted Weight'] = data['Weight'] / data['Variation Factor']
    data['Adjusted Volume'] = data['Adjusted Weight'] * data['Reps']
    
    # Calculate relative intensity (as % of max weight recorded for that exercise)
    exercise_max_weights = data.groupby('Normalized Exercise')['Adjusted Weight'].transform('max')
    data['Relative Intensity'] = (data['Adjusted Weight'] / exercise_max_weights * 100).fillna(0)
    
    # Add recency flag for recent workouts
    now = pd.to_datetime('today')
    data['Days Ago'] = (now - pd.to_datetime(data['Workout Date'])).dt.days
    data['Is Recent'] = data['Days Ago'] <= RECENCY_THRESHOLD

    # Add 1RM calculation (was missing)
    def _calc_1rm(row):
        if row[Config.EXERCISE_COLUMN].lower() in ['pull up', 'chin up']:
            return Config.BODYWEIGHT * (1 + row[Config.REPS_COLUMN]/30)
        if row['Weight'] > 0:
            weight = row['Weight']
            reps = row[Config.REPS_COLUMN]
            return weight * (1 + reps/30)
        return np.nan
    data['1RM'] = data.apply(_calc_1rm, axis=1)

    return data

def convert_to_minutes(duration):
    """Convert workout duration string to total minutes with improved parsing"""
    if pd.isnull(duration): 
        return 0
        
    # Handle "1h 30m" format
    h_match = re.search(r'(\d+)\s*[hH]', duration)
    m_match = re.search(r'(\d+)\s*[mM]', duration)
    
    hours = int(h_match.group(1)) if h_match else 0
    minutes = int(m_match.group(1)) if m_match else 0
    
    if hours == 0 and minutes == 0 and re.search(r'\d+', duration):
        # If no h/m indicator but has numbers, assume minutes
        digits_only = re.search(r'(\d+)', duration)
        minutes = int(digits_only.group(1)) if digits_only else 0
        
    return hours * 60 + minutes

def calculate_1rm(row):
    """Handle mixed units and bodyweight exercises"""
    if row[Config.EXERCISE_COLUMN] in ['Pull Up', 'Chin Up']:
        return Config.BODYWEIGHT * (1 + row[Config.REPS_COLUMN]/30)
    
    if row['Weight'] > 0:
        weight = row['Weight']  # Already converted to lbs
        reps = row[Config.REPS_COLUMN]
        return weight * (1 + reps/30)
    
    return np.nan

def load_and_preprocess_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess data with volume calculation"""
    try:
        # Load data
        data = pd.read_csv(
            file_path, 
            delimiter=Config.DEFAULT_DELIMITER,
            quotechar='"',
            dtype={Config.WEIGHT_COLUMN: str}
        )
        
        # Clean and convert weight data
        data[Config.WEIGHT_COLUMN] = (
            data[Config.WEIGHT_COLUMN]
            .str.replace('"', '')
            .replace('', np.nan)
            .astype(float)
        )
        # Convert kg to lbs
        data['Weight'] = data[Config.WEIGHT_COLUMN] * Config.KG_TO_LBS
        
        # Handle bodyweight exercises
        bodyweight_exercises = ['Pull Up', 'Chin Up', 'Push Up']
        bw_mask = data[Config.EXERCISE_COLUMN].isin(bodyweight_exercises)
        data.loc[bw_mask, 'Weight'] = Config.BODYWEIGHT
        
        # Filter valid workout sets
        valid_data = data[
            (data[Config.REPS_COLUMN] > 0) &
            (data['Set Order'].apply(lambda x: str(x).isdigit()))
        ].copy()
        
        # Calculate volume (ADD THIS)
        valid_data['Volume'] = valid_data['Weight'] * valid_data[Config.REPS_COLUMN]
        
        # Convert duration
        valid_data['Workout Duration (minutes)'] = valid_data[Config.DURATION_COLUMN] / 60
        
        # Parse dates
        valid_data['Workout Date'] = pd.to_datetime(
            valid_data[Config.DATE_COLUMN], 
            format='%Y-%m-%d %H:%M:%S'
        )
        
        return enhance_analysis(valid_data)
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        return None
        
    """# Check for required columns
    required_columns = ['Date', 'Exercise Name', 'Weight', 'Reps']
    missing = [col for col in required_columns if col not in data.columns]
    if missing:
        logging.error(f"Missing required columns: {missing}")
        return None

    # [Enhanced datetime handling]
    try:
        data['Workout Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.date
        if data['Workout Date'].isna().any():
            logging.warning("Invalid dates found in dataset")
    except Exception as e:
        logging.error(f"Date parsing failed: {str(e)}")
        return None
    # Convert and clean duration
    data['Workout Duration (minutes)'] = data['Workout Duration'].apply(convert_to_minutes)

    # Convert Weight to numeric and handle bodyweight exercises
    data['Weight'] = pd.to_numeric(data['Weight'], errors='coerce')
    
    # Expanded list of bodyweight exercises
    bodyweight_exercises = ['pull up', 'chin up', 'push up', 'dip', 'plank', 'crunch', 'sit up']
    
    # Handle case-insensitive matching
    mask = data['Exercise Name'].str.lower().isin([ex.lower() for ex in bodyweight_exercises])
    data.loc[mask, 'Weight'] = BODYWEIGHT  # Set bodyweight exercises
    
    # For exercises with 'bodyweight' in the name
    bw_mask = data['Exercise Name'].str.lower().str.contains('bodyweight')
    data.loc[bw_mask, 'Weight'] = BODYWEIGHT

    # Handle basic calculations
    data['Reps'] = pd.to_numeric(data['Reps'], errors='coerce').fillna(0)
    data['Volume'] = data['Weight'] * data['Reps']
    data['1RM'] = data.apply(calculate_1rm, axis=1)
    data['Workout Date'] = pd.to_datetime(data['Date']).dt.date
    
    # Enhance with additional analytics
    data = enhance_analysis(data.sort_values('Workout Date'))
    
    # Add workout frequency metrics
    data['Workout Week'] = pd.to_datetime(data['Workout Date']).dt.isocalendar().week
    data['Workout Year'] = pd.to_datetime(data['Workout Date']).dt.isocalendar().year
    
    return data"""

def analyze_progressive_overload(data: pd.DataFrame) -> pd.DataFrame:
    """Optimized progressive overload analysis using vectorization."""
    # Use pandas vectorized operations instead of loops
    data['Normalized Base Exercise'] = data['Base Exercise'].apply(normalize_exercise_name)
    
    progression = data.groupby(['Normalized Base Exercise', 'Workout Date']).agg(
        Max_Weight=('Adjusted Weight', 'max'),
        Avg_Weight=('Adjusted Weight', 'mean'),
        Total_Volume=('Adjusted Volume', 'sum'),
        Max_1RM=('1RM', 'max')
    ).reset_index()
    
    # Vectorized rolling averages
    progression['Rolling_Max_Weight'] = (
        progression.groupby('Normalized Base Exercise')['Max_Weight']
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    
    return progression.pivot_table(
        index='Workout Date',
        columns='Normalized Base Exercise',
        values=['Max_Weight', 'Avg_Weight', 'Total_Volume', 'Max_1RM', 'Rolling_Max_Weight']
    ).reset_index() # Reset index to include 'Workout Date' as a column

def calculate_growth_rates(data):
    """Calculate growth rates for key metrics over different time periods"""
    # Get top exercises by frequency
    top_exercises = data['Normalized Exercise'].value_counts().nlargest(10).index.tolist()
    
    # Time periods to analyze
    periods = {
        'last_month': 30,
        'last_quarter': 90,
        'last_year': 365,
        'all_time': float('inf')
    }
    
    growth_rates = {}
    
    for exercise in top_exercises:
        exercise_data = data[data['Normalized Exercise'] == exercise]
        
        for period_name, days in periods.items():
            if period_name == 'all_time':
                period_data = exercise_data
            else:
                cutoff_date = pd.to_datetime('today') - pd.Timedelta(days=days)
                period_data = exercise_data[pd.to_datetime(exercise_data['Workout Date']) >= cutoff_date]
            
            if len(period_data) >= 2:  # Need at least 2 data points
                # Sort by date
                period_data = period_data.sort_values('Workout Date')
                
                # Get first and last records
                first = period_data.iloc[0]
                last = period_data.iloc[-1]
                
                # Calculate days between
                days_between = (pd.to_datetime(last['Workout Date']) - pd.to_datetime(first['Workout Date'])).days
                if days_between > 0:
                    # Calculate 1RM growth
                    first_1rm = first['1RM']
                    last_1rm = last['1RM']
                    
                    if pd.notnull(first_1rm) and pd.notnull(last_1rm) and first_1rm > 0:
                        growth_pct = ((last_1rm - first_1rm) / first_1rm) * 100
                        annualized_growth = growth_pct * (365 / days_between)
                        
                        if exercise not in growth_rates:
                            growth_rates[exercise] = {}
                        
                        growth_rates[exercise][period_name] = {
                            'growth_pct': growth_pct,
                            'annualized_growth': annualized_growth,
                            'days': days_between
                        }
    
    return growth_rates

def calculate_workout_consistency(data):
    """Calculate workout consistency metrics"""
    # Get workout dates
    workout_dates = pd.to_datetime(data['Workout Date'].unique())
    workout_dates = sorted(workout_dates)
    
    # Calculate intervals between workouts
    intervals = []
    for i in range(1, len(workout_dates)):
        interval = (workout_dates[i] - workout_dates[i-1]).days
        intervals.append(interval)
    
    # Calculate metrics
    metrics = {
        'total_workouts': len(workout_dates),
        'first_date': min(workout_dates).strftime('%Y-%m-%d'),
        'last_date': max(workout_dates).strftime('%Y-%m-%d'),
        'date_range_days': (max(workout_dates) - min(workout_dates)).days,
        'avg_interval': np.mean(intervals) if intervals else 0,
        'median_interval': np.median(intervals) if intervals else 0,
        'max_interval': max(intervals) if intervals else 0,
        'workouts_per_week': len(workout_dates) * 7 / (max(workout_dates) - min(workout_dates)).days if len(workout_dates) > 1 else 0
    }
    
    # Calculate consistency score (0-100)
    if metrics['avg_interval'] > 0:
        # More consistent = lower score
        interval_score = max(0, 100 - (metrics['avg_interval'] * 10))
        regularity_score = max(0, 100 - (np.std(intervals) * 5)) if intervals else 100
        metrics['consistency_score'] = int((interval_score + regularity_score) / 2)
    else:
        metrics['consistency_score'] = 0
    
    # Calculate recent trends (last 90 days)
    recent_cutoff = max(workout_dates) - timedelta(days=90)
    recent_workouts = [d for d in workout_dates if d >= recent_cutoff]
    
    metrics['recent_workouts'] = len(recent_workouts)
    
    if len(recent_workouts) > 1:
        recent_intervals = []
        for i in range(1, len(recent_workouts)):
            interval = (recent_workouts[i] - recent_workouts[i-1]).days
            recent_intervals.append(interval)
            
        metrics['recent_avg_interval'] = np.mean(recent_intervals)
        metrics['recent_workouts_per_week'] = len(recent_workouts) * 7 / 90  # Based on 90 day period
    else:
        metrics['recent_avg_interval'] = 0
        metrics['recent_workouts_per_week'] = 0
    
    return metrics

def analyze_workout_splits(data):
    """Analyze workout splits based on muscle groups trained per day"""
    # Group by workout date and get muscle groups trained
    workout_splits = data.groupby('Workout Date')['Muscle Group'].unique().reset_index()
    
    # Map muscle groups to abbreviated form
    muscle_group_map = {
        'Push': 'P',
        'Pull': 'P',
        'Legs': 'L',
        'Core': 'C',
        'Cardio': 'Ca'
    }
    
    # Create split patterns
    splits = []
    for _, row in workout_splits.iterrows():
        groups = row['Muscle Group']
        # Count number of unique groups
        split_type = []
        
        # Check for common combinations
        has_push = 'Push' in groups
        has_pull = 'Pull' in groups
        has_legs = 'Legs' in groups
        has_core = 'Core' in groups
        has_cardio = 'Cardio' in groups
        
        if has_push and has_pull and has_legs:
            split_type.append('Full Body')
        elif has_push and has_pull:
            split_type.append('Upper Body')
        elif has_legs:
            if has_push or has_pull:
                split_type.append('Lower+Upper')
            else:
                split_type.append('Lower Body')
        elif has_push:
            split_type.append('Push')
        elif has_pull:
            split_type.append('Pull')
        
        if has_core and len(split_type) > 0:
            split_type[-1] += '+Core'
        if has_cardio:
            split_type.append('Cardio')
            
        if not split_type:
            if len(groups) > 0:
                split_type = ['+'.join(groups)]
            else:
                split_type = ['Unknown']
        
        splits.append({
            'date': row['Workout Date'],
            'muscle_groups': groups,
            'split_type': '/'.join(split_type)
        })
    
    # Convert to dataframe
    split_df = pd.DataFrame(splits)
    
    # Analyze which split is most common
    split_counts = split_df['split_type'].value_counts()
    
    # Analyze patterns over time
    split_df['date'] = pd.to_datetime(split_df['date'])
    split_df = split_df.sort_values('date')
    
    # Add weekday
    split_df['weekday'] = split_df['date'].dt.day_name()
    
    # Analyze common weekday patterns
    weekday_splits = split_df.groupby('weekday')['split_type'].value_counts().unstack().fillna(0)
    
    return {
        'splits': split_df,
        'common_splits': split_counts,
        'weekday_patterns': weekday_splits
    }

def plot_progression(
    data: pd.DataFrame,
    exercise_list: list,
    metric: str = 'Max_Weight',
    title_prefix: str = '',
    recent_focus: bool = True
) -> str:
    """Enhanced progression plots with dynamic trend detection."""
    plt.figure(figsize=(15, 8))
    plt.style.use('ggplot')
    cmap = cm.get_cmap(Config.COLOR_PALETTE, len(exercise_list))
    
    for i, exercise in enumerate(exercise_list):
        # Use normalized exercise name
        base_exercise = normalize_exercise_name(exercise)
        
        # Find matching columns
        col_name = f'{metric}_{base_exercise}'
        
        if col_name in data.columns:
            subset = data[['Workout Date', col_name]].dropna()
            
            if len(subset) > 0:
                # Convert dates to datetime for proper plotting
                subset['Workout Date'] = pd.to_datetime(subset['Workout Date'])
                subset = subset.sort_values('Workout Date')
                
                # Plot data points
                plt.plot(subset['Workout Date'], subset[col_name], 
                        marker='o', 
                        linestyle='-',
                        color=cmap(i),
                        label=exercise)
                
                # Add trend line if we have enough data
                if len(subset) >= 3:
                    # Get x values as days since start
                    x_days = [(date - subset['Workout Date'].min()).days for date in subset['Workout Date']]
                    
                    # Calculate trend line using linear regression
                    slope, intercept, r_value, p_value, std_err = linregress(x_days, subset[col_name])
                    
                    # Generate line points
                    x_trend = np.array([min(x_days), max(x_days)])
                    y_trend = intercept + slope * x_trend
                    
                    # Convert x back to dates
                    x_dates = [subset['Workout Date'].min() + timedelta(days=int(x)) for x in x_trend]
                    
                    # Plot trend line
                    plt.plot(x_dates, y_trend, 
                            linestyle='--', 
                            color=cmap(i), 
                            alpha=0.7)
                    
                    # Add growth rate to legend
                    if slope > 0:
                        if 'Weight' in metric or '1RM' in metric:
                            # Calculate monthly growth
                            monthly_growth = slope * 30  # Approx 30 days in month
                            exercise_label = f"{exercise} (+{monthly_growth:.1f} lbs/month)"
                        else:
                            # For volume and other metrics
                            pct_change = (subset[col_name].iloc[-1] - subset[col_name].iloc[0]) / subset[col_name].iloc[0] * 100
                            exercise_label = f"{exercise} ({pct_change:.1f}% total)"
                    else:
                        exercise_label = exercise
                    # Actually update legend label
                    plt.gca().get_lines()[-2].set_label(exercise_label)
                
                # Highlight recent data if requested
                if recent_focus:
                    recent_cutoff = pd.to_datetime('today') - pd.Timedelta(days=RECENCY_THRESHOLD)
                    recent_subset = subset[subset['Workout Date'] >= recent_cutoff]
                    
                    if len(recent_subset) > 0:
                        plt.plot(recent_subset['Workout Date'], recent_subset[col_name], 
                                marker='o', 
                                markersize=10,
                                linestyle='none',
                                color=cmap(i),
                                alpha=0.8)
    
    plt.title(f'{title_prefix} {metric.replace("_", " ")} Progression', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    
    # Set appropriate y-axis label
    if 'Weight' in metric:
        plt.ylabel('Weight (lbs)', fontsize=14)
    elif 'Volume' in metric:
        plt.ylabel('Volume (weight × reps)', fontsize=14)
    elif 'Reps' in metric:
        plt.ylabel('Repetitions', fontsize=14)
    elif '1RM' in metric:
        plt.ylabel('Estimated 1RM (lbs)', fontsize=14)
    else:
        plt.ylabel(metric, fontsize=14)
    
    # Format date axis nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Add legend with growth rates
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save with consistent naming based on metric and title prefix
    prefix = title_prefix.lower().replace(' ', '_') + '_' if title_prefix else ''
    metric_name = metric.lower().replace(' ', '_')
    plt.savefig(f'{prefix}{metric_name}_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return f'{prefix}{metric_name}_progression.png'

def plot_muscle_group_balance(data, recent_only=False):
    """Plot radar chart showing muscle group balance with recent vs all-time comparison"""
    # Filter data if needed
    if recent_only:
        recent_data = data[data['Is Recent']]
        plot_data = recent_data
        title_suffix = "Recent (90 Days)"
    else:
        plot_data = data
        title_suffix = "All-Time"
    
    mg_volume = plot_data.groupby('Muscle Group')['Volume'].sum().reset_index()
    
    # Get total volume for percentage calculation
    total_volume = mg_volume['Volume'].sum()
    mg_volume['Percentage'] = (mg_volume['Volume'] / total_volume * 100).round(1)
    
    # Sort categories for consistent order
    categories = ['Push', 'Pull', 'Legs', 'Core', 'Cardio']
    values = []
    
    for cat in categories:
        if cat in mg_volume['Muscle Group'].values:
            val = mg_volume.loc[mg_volume['Muscle Group'] == cat, 'Percentage'].values[0]
        else:
            val = 0
        values.append(val)
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon
    
    values += values[:1]  # Close the polygon
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.4)
    
    # Add percentage labels
    for i, value in enumerate(values[:-1]):  # Skip last repeated value
        angle_rad = angles[i]
        # Adjust radius for label placement
        radius = value + 5
        ha = 'center'
        
        # Adjust horizontal alignment based on angle
        if angle_rad == 0 or np.isclose(angle_rad, 2*np.pi):
            ha = 'left'
        elif np.isclose(angle_rad, np.pi):
            ha = 'right'
            
        ax.text(angle_rad, radius, f"{value:.1f}%", 
                ha=ha, va='center', fontsize=12, fontweight='bold')
    
    # Make the plot look nice
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=14)
    ax.set_yticklabels([])  # Hide radial labels
    
    # Add concentric circles as grid
    grid_values = [20, 40, 60, 80]
    for gv in grid_values:
        ax.plot(angles, [gv] * len(angles), '--', color='grey', alpha=0.7, linewidth=0.5)
        ax.text(0, gv, f"{gv}%", ha='left', va='center', alpha=0.7)
    
    ax.set_ylim(0, 100)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    plt.title(f'Muscle Group Volume Balance - {title_suffix}', y=1.1, fontsize=16, fontweight='bold')
    
    filename = f"muscle_balance_radar{'_recent' if recent_only else ''}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_muscle_balance_comparison(data):
    """Plot comparison of recent vs all-time muscle group balance"""
    # Calculate recent balance
    recent_data = data[data['Is Recent']]
    recent_mg = recent_data.groupby('Muscle Group')['Volume'].sum()
    recent_total = recent_mg.sum()
    recent_pct = (recent_mg / recent_total * 100).round(1)
    
    # Calculate all-time balance
    all_mg = data.groupby('Muscle Group')['Volume'].sum()
    all_total = all_mg.sum()
    all_pct = (all_mg / all_total * 100).round(1)
    
    # Merge into one dataframe
    categories = ['Push', 'Pull', 'Legs', 'Core', 'Cardio']
    comparison = pd.DataFrame(index=categories)
    
    for cat in categories:
        comparison.loc[cat, 'Recent'] = recent_pct.get(cat, 0)
        comparison.loc[cat, 'All Time'] = all_pct.get(cat, 0)
    
    comparison = comparison.fillna(0)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(12,8))
    
    # Plot data
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, comparison['Recent'], width, label='Recent (90 Days)', color='#3498db')
    rects2 = ax.bar(x + width/2, comparison['All Time'], width, label='All Time', color='#2c3e50')
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Format plot
    ax.set_title('Muscle Group Volume Distribution: Recent vs All-Time', fontsize=16, fontweight='bold')
    ax.set_ylabel('Percentage of Total Volume', fontsize=14)
    ax.set_xlabel('Muscle Group', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add grid for readability
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add a horizontal line at y=20% as a reference for balanced training
    ax.axhline(y=20, color='red', linestyle='--', alpha=0.5)
    ax.text(len(categories)-1, 21, 'Balanced (20%)', color='red', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('muscle_balance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'muscle_balance_comparison.png'

def plot_workout_frequency(data):
    """Plot workout frequency over time with trend analysis"""
    # Extract workout dates and create a series
    workout_dates = pd.to_datetime(data['Workout Date'].unique())
    date_series = pd.Series(workout_dates).sort_values()
    
    # Create a date range from first to last workout
    full_range = pd.date_range(start=date_series.min(), end=date_series.max(), freq='D')
    
    # Count workouts by week
    workout_counts = pd.Series(index=full_range, data=0)
    workout_counts[date_series] = 1
    
    # Resample by week
    weekly_workouts = workout_counts.resample('W').sum()
    
    # Calculate 4-week moving average
    rolling_avg = weekly_workouts.rolling(window=4, min_periods=1).mean()
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot weekly workout counts
    ax.bar(weekly_workouts.index, weekly_workouts.values, width=5, alpha=0.7, color='#3498db', label='Weekly Workouts')
    
    # Plot moving average trend line
    ax.plot(rolling_avg.index, rolling_avg.values, 'r-', linewidth=2, label='4-Week Moving Avg')
    
    # Highlight recent period
    recent_cutoff = pd.to_datetime('today') - pd.Timedelta(days=RECENCY_THRESHOLD)
    ax.axvline(x=recent_cutoff, linestyle='--', color='green', alpha=0.7, label=f'Last {RECENCY_THRESHOLD} Days')
    
    # Format plot
    ax.set_title('Workout Frequency Over Time', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Number of Workouts per Week', fontsize=14)
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    # Integer y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('workout_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'workout_frequency.png'

def plot_volume_heatmap(data):
    """Generate a heatmap showing workout volume by muscle group over time"""
    # Prepare data: aggregate by week and muscle group
    data['Week'] = pd.to_datetime(data['Workout Date']).dt.to_period('W')
    heatmap_data = data.groupby(['Week', 'Muscle Group'])['Volume'].sum().unstack().fillna(0)
    
    # Convert Period index to datetime
    heatmap_data.index = heatmap_data.index.to_timestamp()
    
    # Ensure all muscle groups are present
    for group in ['Push', 'Pull', 'Legs', 'Core', 'Cardio']:
        if group not in heatmap_data.columns:
            heatmap_data[group] = 0
    
    # Sort columns in consistent order
    heatmap_data = heatmap_data[['Push', 'Pull', 'Legs', 'Core', 'Cardio']]
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    
    # Normalize data for better visualization (log scale works well for volume)
    norm_data = np.log1p(heatmap_data)  # log(1+x) to handle zeros
    
    # Plot heatmap with improved colormap
    ax = sns.heatmap(norm_data.T, cmap='YlOrRd', cbar_kws={'label': 'Log(Volume)'})
    
    # Format plot
    plt.title('Weekly Training Volume by Muscle Group', fontsize=16, fontweight='bold')
    plt.ylabel('Muscle Group', fontsize=14)
    plt.xlabel('Week', fontsize=14)
    
    # Format x-axis labels
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('volume_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'volume_heatmap.png'

def plot_workout_duration_vs_volume(data):
    """Plot relationship between workout duration and total volume"""
    # Group by workout date
    workout_metrics = data.groupby('Workout Date').agg(
        Duration=('Workout Duration (minutes)', 'first'),
        Total_Volume=('Volume', 'sum'),
        Exercise_Count=('Exercise Name', 'nunique')
    ).reset_index()
    
    # Create scatterplot
    plt.figure(figsize=(12, 8))
    
    # Create a colormap based on exercise count
    scatter = plt.scatter(
        workout_metrics['Duration'], 
        workout_metrics['Total_Volume'],
        c=workout_metrics['Exercise_Count'], 
        cmap='viridis',
        alpha=0.7,
        s=80,
        edgecolor='k'
    )
    
    # Add colorbar for exercise count
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Different Exercises', fontsize=12)
    
    # Add trendline
    z = np.polyfit(workout_metrics['Duration'], workout_metrics['Total_Volume'], 1)
    p = np.poly1d(z)
    plt.plot(
        workout_metrics['Duration'],
        p(workout_metrics['Duration']),
        "r--", 
        alpha=0.8,
        linewidth=2
    )
    
    # Calculate correlation coefficient
    corr = np.corrcoef(workout_metrics['Duration'], workout_metrics['Total_Volume'])[0,1]
    
    # Format plot
    plt.title('Workout Duration vs Total Volume', fontsize=16, fontweight='bold')
    plt.xlabel('Workout Duration (minutes)', fontsize=14)
    plt.ylabel('Total Volume (weight × reps)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add correlation annotation
    plt.annotate(
        f"Correlation: {corr:.2f}",
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
        fontsize=12
    )
    
    plt.tight_layout()
    plt.savefig('duration_vs_volume.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'duration_vs_volume.png'

def plot_1rm_improvement_summary(data, top_n=5):
    """Plot horizontal bar chart of 1RM improvements for top exercises"""
    # Get top exercises by frequency
    top_exercises = data['Normalized Exercise'].value_counts().nlargest(top_n).index.tolist()
    
    # Calculate 1RM improvements
    improvements = []
    
    for exercise in top_exercises:
        exercise_data = data[data['Normalized Exercise'] == exercise].sort_values('Workout Date')
        
        if len(exercise_data) >= 2:
            first_1rm = exercise_data.iloc[0]['1RM']
            last_1rm = exercise_data.iloc[-1]['1RM']
            
            if pd.notnull(first_1rm) and pd.notnull(last_1rm) and first_1rm > 0:
                absolute_change = last_1rm - first_1rm
                percent_change = (absolute_change / first_1rm) * 100
                days_between = (pd.to_datetime(exercise_data.iloc[-1]['Workout Date']) - 
                               pd.to_datetime(exercise_data.iloc[0]['Workout Date'])).days
                
                improvements.append({
                    'Exercise': exercise,
                    'Initial_1RM': first_1rm,
                    'Final_1RM': last_1rm,
                    'Absolute_Change': absolute_change,
                    'Percent_Change': percent_change,
                    'Days': days_between,
                    'Monthly_Rate': (absolute_change / days_between) * 30 if days_between > 0 else 0,
                    'Date_Range': f"{exercise_data.iloc[0]['Workout Date']} to {exercise_data.iloc[-1]['Workout Date']}"
                })
    
    if not improvements:
        return None
    
    # Convert to DataFrame and sort
    improvement_df = pd.DataFrame(improvements)
    improvement_df = improvement_df.sort_values('Percent_Change', ascending=False)
    
    # Plot horizontal bar chart
    plt.figure(figsize=(12, 8))
    
    # Create a colormap for positive/negative changes
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvement_df['Percent_Change']]
    
    # Plot bars
    bars = plt.barh(
        improvement_df['Exercise'], 
        improvement_df['Percent_Change'],
        color=colors,
        alpha=0.7,
        edgecolor='k'
    )
    
    # Add data labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + (1 if width >= 0 else -1)
        
        plt.text(
            label_x,
            bar.get_y() + bar.get_height()/2,
            f"{improvement_df.iloc[i]['Absolute_Change']:.1f} lbs ({improvement_df.iloc[i]['Percent_Change']:.1f}%)",
            va='center',
            fontsize=10,
            fontweight='bold'
        )
    
    # Format plot
    plt.title('1RM Improvement by Exercise', fontsize=16, fontweight='bold')
    plt.xlabel('Percent Change (%)', fontsize=14)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1rm_improvements.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return '1rm_improvements.png', improvement_df

def analyze_exercise_clusters(data):
    """Analyze and visualize exercise clusters based on metrics"""
    # Prepare exercise-level metrics
    exercise_metrics = data.groupby('Normalized Exercise').agg(
        Avg_Weight=('Weight', 'mean'),
        Max_Weight=('Weight', 'max'),
        Avg_Reps=('Reps', 'mean'),
        Max_Reps=('Reps', 'max'),
        Total_Volume=('Volume', 'sum'),
        Frequency=('Workout Date', 'nunique'),
        Avg_1RM=('1RM', 'mean')
    ).reset_index()
    
    # Filter exercises with at least 3 occurrences and handle NaNs
    exercise_metrics = exercise_metrics[exercise_metrics['Frequency'] >= 3]
    exercise_metrics = exercise_metrics.fillna(0)  # Fill missing values
    
    if len(exercise_metrics) < 5:
        return None  # Not enough exercises for meaningful clustering
    
    # Select features for clustering
    features = exercise_metrics[['Avg_Weight', 'Avg_Reps', 'Total_Volume']]
    
    # Normalize features with NaN protection
    features_scaled = (features - features.mean()) / (features.std() + 1e-8) 
    
    # Determine optimal number of clusters (2-5 clusters)
    max_clusters = min(5, len(exercise_metrics) - 1)
    
    if max_clusters < 2:
        return None
        
    # Calculate silhouette scores
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        score = 0  # Default if only one sample per cluster
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters with samples
            from sklearn.metrics import silhouette_score
            score = silhouette_score(features_scaled, cluster_labels)
        silhouette_scores.append(score)
    
    # Select best number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from 2
    
    # Perform clustering with optimal number
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    exercise_metrics['Cluster'] = kmeans.fit_predict(features_scaled)
    
    # Add cluster centers to original data
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=['Avg_Weight', 'Avg_Reps', 'Total_Volume']
    )
    
    # Visualize clusters
    plt.figure(figsize=(14, 10))
    
    # Create gridspec for layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Main scatter plot
    ax1 = plt.subplot(gs[0, :])
    
    # Create a colormap for clusters
    cmap = plt.get_cmap('tab10', optimal_clusters)
    norm = Normalize(vmin=0, vmax=optimal_clusters-1)
    
    # Plot each cluster
    for cluster in range(optimal_clusters):
        cluster_data = exercise_metrics[exercise_metrics['Cluster'] == cluster]
        ax1.scatter(
            cluster_data['Avg_Weight'], 
            cluster_data['Avg_Reps'],
            s=cluster_data['Total_Volume'] / 1000,  # Size by volume
            c=[cmap(norm(cluster))] * len(cluster_data),
            alpha=0.7,
            edgecolor='k',
            label=f'Cluster {cluster+1}'
        )
        
        # Annotate points with exercise names
        for i, row in cluster_data.iterrows():
            ax1.annotate(
                row['Normalized Exercise'],
                (row['Avg_Weight'], row['Avg_Reps']),
                fontsize=9,
                alpha=0.8,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # Format main plot
    ax1.set_title('Exercise Clusters by Weight, Reps, and Volume', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Average Weight (lbs)', fontsize=14)
    ax1.set_ylabel('Average Reps', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title='Cluster', fontsize=10)
    
    # Add size legend
    sizes = [1000, 5000, 20000, 50000]
    for size in sizes:
        ax1.scatter([], [], s=size/1000, c='gray', alpha=0.4, edgecolor='k', label=f'{size:,} Volume')
    
    leg = ax1.legend(title="Total Volume", loc="upper left", bbox_to_anchor=(1, 1))
    
    # Add cluster summary
    ax2 = plt.subplot(gs[1, :])
    ax2.axis('off')
    
    # Create cluster summary text
    summary_text = "Cluster Summary:\n\n"
    
    for i in range(optimal_clusters):
        cluster_exercises = exercise_metrics[exercise_metrics['Cluster'] == i]['Normalized Exercise'].tolist()
        cluster_avg_weight = exercise_metrics[exercise_metrics['Cluster'] == i]['Avg_Weight'].mean()
        cluster_avg_reps = exercise_metrics[exercise_metrics['Cluster'] == i]['Avg_Reps'].mean()
        
        summary_text += f"Cluster {i+1}: Avg Weight: {cluster_avg_weight:.1f} lbs, Avg Reps: {cluster_avg_reps:.1f}\n"
        summary_text += f"  Exercises: {', '.join(cluster_exercises[:5])}"
        if len(cluster_exercises) > 5:
            summary_text += f" and {len(cluster_exercises) - 5} more"
        summary_text += "\n\n"
    
    ax2.text(0, 1, summary_text, fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('exercise_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return 'exercise_clusters.png', exercise_metrics

def generate_pdf_report(
    data: pd.DataFrame,
    output_file: str = "workout_analysis_report.pdf",
    include_sections: list = ['progress', 'balance', 'frequency']
) -> str:
    """Generate comprehensive PDF report with all analyses"""
    # Get most common exercises
    top_exercises = data['Normalized Exercise'].value_counts().nlargest(5).index.tolist()
    
    # Get important exercise progressions
    progression_data = analyze_progressive_overload(data)
    
    # Calculate workout consistency metrics
    consistency_metrics = calculate_workout_consistency(data)
    
    # Calculate growth rates
    growth_rates = calculate_growth_rates(data)
    
    # Analyze workout splits
    workout_splits = analyze_workout_splits(data)
    
    # Generate all the plots and save them
    progression_data = analyze_progressive_overload(data)
    top_exercises = data['Normalized Exercise'].value_counts().nlargest(5).index.tolist()
    plot_files = {
        'weight_progression': plot_progression(progression_data, top_exercises, 'Max_Weight', 'Top Exercises'),
        'volume_progression': plot_progression(progression_data, top_exercises, 'Total_Volume', 'Top Exercises'),
        # Fix: Only plot Max_Reps if present in progression_data
        'rep_progression': plot_progression(progression_data, top_exercises, 'Max_Reps', 'Top Exercises') if any(col.startswith('Max_Reps') for col in progression_data.columns) else None,
        'muscle_balance': plot_muscle_group_balance(data, False),
        'recent_muscle_balance': plot_muscle_group_balance(data, True),
        'muscle_comparison': plot_muscle_balance_comparison(data),
        'workout_frequency': plot_workout_frequency(data),
        'volume_heatmap': plot_volume_heatmap(data),
        'duration_volume': plot_workout_duration_vs_volume(data)
    }
    
    # Attempt to generate 1RM improvement summary
    rm_improvement_result = plot_1rm_improvement_summary(data)
    if rm_improvement_result:
        plot_files['1rm_improvements'] = rm_improvement_result[0]
        improvement_df = rm_improvement_result[1]
    else:
        improvement_df = None
    
    # Attempt to generate exercise clusters
    clusters_result = analyze_exercise_clusters(data)
    if clusters_result:
        plot_files['exercise_clusters'] = clusters_result[0]
        clusters_df = clusters_result[1]
    else:
        clusters_df = None
    
    # Create PDF document
    doc = SimpleDocTemplate(output_file, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'SubheadingStyle',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceBefore=5,
        spaceAfter=5
    )
    
    # Start building the document
    story = []
    
    # Title page
    story.append(Paragraph("Workout Analysis Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d')}", normal_style))
    story.append(Spacer(1, 20))
    
    # Summary section
    story.append(Paragraph("Executive Summary", heading_style))
    
    # Workout consistency summary
    workout_summary = [
        f"Total Workouts: {consistency_metrics['total_workouts']}",
        f"Date Range: {consistency_metrics['first_date']} to {consistency_metrics['last_date']} ({consistency_metrics['date_range_days']} days)",
        f"Average Workouts Per Week: {consistency_metrics['workouts_per_week']:.2f}",
        f"Recent Workouts (Last 90 Days): {consistency_metrics['recent_workouts']}",
        f"Recent Workouts Per Week: {consistency_metrics['recent_workouts_per_week']:.2f}",
        f"Consistency Score: {consistency_metrics['consistency_score']}/100"
    ]
    
    for line in workout_summary:
        story.append(Paragraph(line, normal_style))
    
    story.append(Spacer(1, 10))
    
    # Top exercises summary
    story.append(Paragraph("Most Frequent Exercises", subheading_style))
    for i, exercise in enumerate(top_exercises[:5], 1):
        count = data[data['Normalized Exercise'] == exercise]['Workout Date'].nunique()
        story.append(Paragraph(f"{i}. {exercise} ({count} workouts)", normal_style))
    
    story.append(Spacer(1, 10))
    
    # Add workout frequency chart
    story.append(Paragraph("Workout Frequency", heading_style))
    story.append(Paragraph("The chart below shows your workout frequency over time with a 4-week moving average trend line.", normal_style))
    story.append(Image(plot_files['workout_frequency'], width=7*inch, height=3.5*inch))
    
    story.append(PageBreak())
    
    # Muscle balance section
    story.append(Paragraph("Muscle Group Analysis", heading_style))
    story.append(Paragraph("This analysis shows how your training volume is distributed across major muscle groups.", normal_style))
    
    # Add both charts side by side in a table
    img_table = Table([
        [Image(plot_files['muscle_balance'], width=3*inch, height=3*inch), 
         Image(plot_files['recent_muscle_balance'], width=3*inch, height=3*inch)]
    ])
    story.append(img_table)
    story.append(Spacer(1, 10))
    
    # Add comparison chart
    story.append(Paragraph("Recent vs. All-Time Muscle Group Distribution", subheading_style))
    story.append(Image(plot_files['muscle_comparison'], width=7*inch, height=3.5*inch))
    story.append(Spacer(1, 10))
    
    # Add volume heatmap
    story.append(Paragraph("Weekly Training Volume Heatmap", subheading_style))
    story.append(Paragraph("This heatmap shows your training volume distribution over time by muscle group.", normal_style))
    story.append(Image(plot_files['volume_heatmap'], width=7*inch, height=3.5*inch))
    
    story.append(PageBreak())
    
    # Exercise progression section
    story.append(Paragraph("Exercise Progression Analysis", heading_style))
    story.append(Paragraph("These charts show how your performance has progressed over time for your most frequent exercises.", normal_style))
    
    # Add weight progression chart
    story.append(Paragraph("Weight Progression", subheading_style))
    story.append(Image(plot_files['weight_progression'], width=7*inch, height=3.5*inch))
    
    # Add volume progression chart
    story.append(Paragraph("Volume Progression", subheading_style))
    story.append(Image(plot_files['volume_progression'], width=7*inch, height=3.5*inch))
    
    story.append(PageBreak())
    
    # Add 1RM improvements if available
    if plot_files.get('1rm_improvements'):
        story.append(Paragraph("Strength Improvements (1RM)", heading_style))
        story.append(Paragraph("This chart shows your estimated one-rep max (1RM) improvements for your main exercises.", normal_style))
        story.append(Image(plot_files['1rm_improvements'], width=7*inch, height=3.5*inch))
        
        # Add improvement details in a table
        if improvement_df is not None and len(improvement_df) > 0:
            improvement_data = [['Exercise', 'Initial 1RM', 'Current 1RM', 'Change', 'Percent']]
            
            for _, row in improvement_df.iterrows():
                improvement_data.append([
                    row['Exercise'],
                    f"{row['Initial_1RM']:.1f}",
                    f"{row['Final_1RM']:.1f}",
                    f"{row['Absolute_Change']:.1f}",
                    f"{row['Percent_Change']:.1f}%"
                ])
                
            improvement_table = Table(improvement_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
            improvement_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(improvement_table)
            
        story.append(Spacer(1, 20))
    
    # Add workout duration vs volume analysis
    story.append(Paragraph("Workout Duration vs. Volume", heading_style))
    story.append(Paragraph("This chart shows the relationship between workout duration and total volume, with color indicating the number of different exercises performed.", normal_style))
    story.append(Image(plot_files['duration_volume'], width=7*inch, height=3.5*inch))
    
    # Add exercise clusters if available
    if 'exercise_clusters' in plot_files:
        story.append(PageBreak())
        story.append(Paragraph("Exercise Clustering Analysis", heading_style))
        story.append(Paragraph("This analysis groups your exercises based on similar characteristics (weight, reps, and volume).", normal_style))
        story.append(Image(plot_files['exercise_clusters'], width=7*inch, height=5*inch))
    
    # Add workout splits analysis
    story.append(PageBreak())
    story.append(Paragraph("Workout Split Analysis", heading_style))
    
    # Create workout splits summary
    splits_summary = workout_splits['common_splits'].nlargest(5)
    splits_data = [['Split Type', 'Frequency']]
    
    for split, count in splits_summary.items():
        splits_data.append([split, str(count)])
        
    splits_table = Table(splits_data, colWidths=[4*inch, 1*inch])
    splits_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(splits_table)
    
    # Description of split types
    split_descriptions = [
        "Your most common split types are listed above. For full-body workouts, all major muscle groups are trained in a single session.",
        "For push/pull/legs splits, each session focuses on either push muscles (chest, shoulders, triceps), pull muscles (back, biceps), or legs.",
        "For upper/lower splits, workouts alternate between upper body and lower body focus.",
        "Specialized splits like 'Chest & Back', 'Arms', etc., indicate days with specific muscle group focus."
    ]
    
    for desc in split_descriptions:
        story.append(Paragraph(desc, normal_style))
    
    # Add growth rates
    story.append(Paragraph("Progress Summary", heading_style))
    progress_data = [['Metric', 'Monthly Growth Rate', 'Projected Annual Growth']]
    
    metrics = [
        ('Max Weight', 'Weight_Growth_Rate', '%'),
        ('Volume', 'Volume_Growth_Rate', '%'),
        ('Workout Duration', 'Duration_Growth_Rate', 'min'),
        ('Exercise Variety', 'Variety_Growth_Rate', 'exercises')
    ]
    
    for label, key, unit in metrics:
        if key in growth_rates:
            monthly = growth_rates[key]
            annual = (1 + monthly/100)**12 - 1 if unit == '%' else monthly * 12
            annual_unit = '%' if unit == '%' else unit
            
            progress_data.append([
                label,
                f"{monthly:.2f}{unit}/month",
                f"{annual:.2f}{annual_unit}/year"
            ])
    
    progress_table = Table(progress_data, colWidths=[2*inch, 2*inch, 2*inch])
    progress_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(progress_table)
    story.append(Spacer(1, 10))
    
    # Add recommendations
    story.append(Paragraph("Recommendations", heading_style))
    
    # Based on consistency score
    if consistency_metrics['consistency_score'] < 50:
        story.append(Paragraph("Consistency: Consider creating a more regular workout schedule. Aim for at least 3-4 workouts per week for optimal results.", normal_style))
    elif consistency_metrics['consistency_score'] < 80:
        story.append(Paragraph("Consistency: Your workout consistency is good but could be improved. Try to maintain a more regular schedule.", normal_style))
    else:
        story.append(Paragraph("Consistency: Excellent workout consistency! Keep up the regular training schedule.", normal_style))
    
    # Based on muscle balance
    muscle_balance = workout_splits.get('muscle_balance', {})
    if muscle_balance:
        imbalanced = [muscle for muscle, ratio in muscle_balance.items() if ratio < 0.7 or ratio > 1.3]
        if imbalanced:
            story.append(Paragraph(f"Muscle Balance: Consider adding more focus to these muscle groups: {', '.join(imbalanced)}", normal_style))
        else:
            story.append(Paragraph("Muscle Balance: Your training shows good balance across muscle groups.", normal_style))
    
    # Based on progression rates
    if 'Weight_Growth_Rate' in growth_rates and growth_rates['Weight_Growth_Rate'] < 1:
        story.append(Paragraph("Progression: Your strength gains have slowed. Consider implementing periodization or varying your rep ranges to continue progressing.", normal_style))
    
    # Based on volume trends
    if 'Volume_Growth_Rate' in growth_rates:
        if growth_rates['Volume_Growth_Rate'] < 0:
            story.append(Paragraph("Volume: Training volume has decreased over time. Consider gradually increasing volume to continue making progress.", normal_style))
        elif growth_rates['Volume_Growth_Rate'] > 5:
            story.append(Paragraph("Volume: Training volume is increasing rapidly. Ensure you're allowing adequate recovery between sessions.", normal_style))
    
    # Footer
    story.append(PageBreak())
    story.append(Paragraph("Analysis Notes", heading_style))
    story.append(Paragraph("This report was generated based on your workout data and provides an objective analysis of your training patterns, progress, and potential areas for improvement.", normal_style))
    story.append(Paragraph("The recommendations are based on general fitness principles and may need to be adjusted based on your specific goals, limitations, and preferences.", normal_style))
    story.append(Paragraph("For best results, consider reviewing this analysis with a qualified fitness professional who can provide personalized guidance.", normal_style))
    
    # Build and save the PDF
    doc.build(story)
    
    return output_file

def main() -> None:
    """Main function with enhanced user interaction."""
    try:
        file_path = input("Enter CSV file path: ").strip()
        if not file_path.endswith('.csv'):
            raise ValueError("Invalid file format. Please provide a CSV file.")
            
        data = load_and_preprocess_data(file_path)
        if data is not None:
            report_path = generate_pdf_report(data)
            logging.info(f"Report generated successfully: {report_path}")
        else:
            logging.error("Failed to generate report due to data issues")
    except Exception as e:
        logging.error(f"Critical error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
# 🏋️ Workout Analysis Project

## Overview
A Python-based tool to analyze `workout_data.csv` (exported from Strong/Hevy apps) and generate a rich, visual HTML report. 
It helps you track consistency, strength progress (E1RM), volume, and muscle balance.

## project Structure
-   `analyze_workouts.py`: Main script. Reads data, processes metrics, generates plots, and creates the HTML report.
-   `workout_data.csv`: Input data file (must be in the same directory).
-   `workout_report.html`: The generated dashboard (automatically opens after running).
-   `strong-analysis.py`: Legacy script used for reference (contains advanced math/sklearn logic).

## Version Control
This project is git-initialized.
-   **Remote**: [SIERRA711/workout-analysis](https://github.com/SIERRA711/workout-analysis)
-   **One-time Setup**: Add git to PATH and configure identity (`git config --global user.email "you@example.com"`).
-   **Ignore**: The `.gitignore` excludes `workout_data.csv` (privacy) and `workout_report.html` (generated).

## Features
### 📊 Visual Dashboard
-   **Dark Mode** HTML Report.
-   **Print-to-PDF** ready.

### 📈 key Metrics
1.  **Consistency Score (0-100)**: Gamified score based on workout regularity and gaps.
2.  **Muscle Balance Radar**: Spider chart showing Push / Pull / Legs / Core / Cardio distribution.
3.  **Volume Heatmap**: Intensity visualization by muscle group over time.
4.  **Strength Progress**: 6-month E1RM history for top exercises.
5.  **PR Alerts**: 
    -   **🔥 Strength**: New Estimated 1RM.
    -   **🏆 Heavy**: New Absolute Max Weight.

## Usage
Run the script using the Python launcher:
```powershell
py analyze_workouts.py --input workout_data.csv
```

## Roadmap / Todo
- [x] Basic Analysis (Frequency, Volume, E1RM)
- [x] HTML Report Generation
- [x] Visualizations (Matplotlib)
- [x] Advanced Metrics (PRs, Consistency Score)
- [x] Legacy Feature Integration (Radar, Heatmap)
- [x] Dark Mode
- [ ] User Goals (Target weights/reps)
- [ ] Bodyweight tracking integration
- [ ] Machine Learning clustering (from legacy script)

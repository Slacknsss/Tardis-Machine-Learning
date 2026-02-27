
# TARDIS - SNCF Delay Analysis & Prediction

## Project Overview

TARDIS is a data analysis and machine learning project focused on analyzing and predicting train delays in France.

The objective is to:
- Explore historical delay data
- Identify patterns and trends
- Build a predictive model
- Deploy an interactive dashboard for visualization and delay estimation

This project combines data preprocessing, exploratory data analysis (EDA), and supervised machine learning.

---

## Dataset Description

The dataset contains information about train traffic including:

- Date
- Service type
- Departure station
- Arrival station
- Average trip duration
- Number of scheduled circulations
- Number of delayed trains
- Average delay at departure and arrival
- Causes of delays (infrastructure, traffic management, rolling stock, etc.)
- Year and month

Target variable:
- **Retard moyen de tous les trains à l'arrivée**

---

## Data Processing

Data preprocessing includes:

- Cleaning column names
- Handling missing values
- Converting dates
- Encoding categorical variables using LabelEncoder
- Feature selection

Encoders are saved as:
- `le_dep.joblib`
- `le_arr.joblib`

The trained model is saved as:
- `model.joblib`

---

## Exploratory Data Analysis (EDA)

The dashboard includes:

### Summary Statistics Panel
- Global average delay
- Total number of scheduled circulations
- Punctuality rate (< 5 minutes delay)

### Visualizations
- Monthly delay trends
- Top 10 most delayed arrival stations
- Departure / arrival delay heatmap
- Correlation between traffic volume and delays

These visualizations help identify structural patterns and congestion effects.

---

## Machine Learning Model

The model predicts the average arrival delay based on:

- Encoded departure station
- Encoded arrival station
- Month
- Average trip duration

Model type:
- Random Forest Regressor

Encoders ensure consistency between training and prediction phases.

---

## Dashboard

The dashboard is built using Streamlit.

It includes two sections:

### 1. Graphiques
Interactive data exploration and KPI display.

### 2. Prédiction
Users select:
- Departure station
- Arrival station
- Month

The system:
- Automatically estimates average trip duration
- Encodes stations
- Predicts the expected delay

To launch the dashboard:

---

## Key Insights

- Delay patterns vary seasonally.
- Some stations consistently show higher average delays.
- Traffic density impacts delay magnitude.
- Predictive modeling provides reasonable estimation based on historical patterns.

---

## Limitations

- Model performance depends on historical patterns only.
- External real-time events are not included.
- Prediction accuracy may vary for rare routes.

---

## Future Improvements

- Add real-time data integration
- Improve feature engineering
- Add model evaluation metrics to dashboard
- Deploy the application online

---
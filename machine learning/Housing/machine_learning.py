import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Read CSV and filter data for Canada (index % 9 == 8)
file_path = 'machine learning\\Trimmed_Time_Series_Data.csv'
df = pd.read_csv(file_path)
df_subset = df[df.index % 9 == 8].iloc[:, :]
df_subset.reset_index(drop=True, inplace=True)
print(df_subset.head())
df_subset.to_csv('machine learning/Trimmed_Time_Series_Data_Canada.csv', index=False)

# Prepare DataFrame for Prophet by renaming "REF_DATE" to "ds" and converting to datetime
df_prophet = df_subset.rename(columns={"REF_DATE": "ds"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

# Use all columns except the first two as target variables
target_columns = list(df_prophet.columns[2:3])

def create_features(df, n_lags=3):
    """
    Create simple lag features and a month feature.
    Assumes df has columns 'ds' and 'y'.
    """
    df_features = df.copy()
    for lag in range(1, n_lags+1):
        df_features[f'lag_{lag}'] = df_features['y'].shift(lag)
    df_features['month'] = df_features['ds'].dt.month
    df_features.dropna(inplace=True)
    return df_features

holdout_period = 30  # e.g., predicting 30 months ahead
n_lags = 3         # number of lag features for ML models

for target in target_columns:
    print(f"\nEvaluating target: {target}")
    
    # --- Prophet Forecast with MAE ---
    df_target = df_prophet[["ds", target]].rename(columns={target: "y"})
    # Split into training and testing sets for Prophet
    train_prophet = df_target.iloc[:-holdout_period]
    test_prophet  = df_target.iloc[-holdout_period:]
    prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=False)
    prophet_model.fit(train_prophet)
    future = prophet_model.make_future_dataframe(periods=holdout_period, freq='MS')
    forecast = prophet_model.predict(future).tail(holdout_period)
    prophet_mae = mean_absolute_error(test_prophet['y'], forecast['yhat'])
    print("Prophet MAE:", prophet_mae)
    # Optionally, display forecast details:
    print("Prophet Forecast (first few rows):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
    
    # --- Prepare Data for Random Forest / XGBoost ---
    # Create lag features for ML models; we'll only use known (historical) data.
    df_ml = create_features(df_target, n_lags=n_lags)
    
    # Use the last 'holdout_period' records as the test set (hold-out evaluation)
    train_ml = df_ml.iloc[:-holdout_period]
    test_ml = df_ml.iloc[-holdout_period:]
    
    X_train = train_ml.drop(['ds', 'y'], axis=1)
    y_train = train_ml['y']
    X_test  = test_ml.drop(['ds', 'y'], axis=1)
    y_test  = test_ml['y']
    
    # --- Random Forest ---
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    print("Random Forest MAE:", rf_mae)
    
    # --- XGBoost ---
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    print("XGBoost MAE:", xgb_mae)
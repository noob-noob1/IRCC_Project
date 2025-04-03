import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Set global plotting style
plt.rcParams['figure.figsize'] = (14, 6)
plt.style.use('seaborn-v0_8-whitegrid') # Use updated style name

def load_and_prepare_data(file_path):
    """Loads and prepares the housing data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_csv(file_path)
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df.set_index('REF_DATE', inplace=True)
    df.sort_index(inplace=True)
    print("Data Information:")
    print(df.info())
    print("\nData Head:")
    print(df.head())
    return df

def train_evaluate_prophet(train_prophet, test_prophet, exog_cols):
    """Trains and evaluates the Prophet model."""
    try:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        for col in exog_cols:
            m.add_regressor(col)
        m.fit(train_prophet)

        future_test = m.make_future_dataframe(periods=len(test_prophet), freq='MS')
        # Fill future exogenous regressors with the last known value from training
        for col in exog_cols:
             # Ensure the column exists in future_test before assigning
            if col in future_test.columns:
                 # Use the last value from the training set for the future periods
                last_val = train_prophet[col].iloc[-1]
                future_test[col] = last_val
            else:
                # If the column wasn't automatically added (e.g., if it was the target 'y'), handle appropriately
                # In this setup, exog_cols should not contain 'y' or 'ds', so this might indicate an issue
                print(f"Warning: Exogenous regressor '{col}' not found in future_test DataFrame during prediction setup.")
                # As a fallback, try assigning based on the last training value directly if needed,
                # but Prophet's make_future_dataframe should handle known regressors.
                # This might require creating the column if it's missing.
                last_val = train_prophet[col].iloc[-1]
                future_test[col] = last_val # Add the column if missing and fill


        forecast_prophet = m.predict(future_test)
        prophet_pred = forecast_prophet.iloc[-len(test_prophet):]['yhat'].values

        rmse = np.sqrt(mean_squared_error(test_prophet['y'], prophet_pred))
        mae = mean_absolute_error(test_prophet['y'], prophet_pred)
        r2 = r2_score(test_prophet['y'], prophet_pred)
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"Prophet RMSE: {rmse:.2f}")

        # Plotting
        plt.figure()
        plt.plot(train_prophet['ds'], train_prophet['y'], label='Train')
        plt.plot(test_prophet['ds'], test_prophet['y'], label='Test')
        plt.plot(future_test['ds'].iloc[-len(test_prophet):], prophet_pred, label='Prophet Prediction')
        plt.title(f"Prophet Forecast on Test Data")
        plt.legend()
        # plt.show() # Avoid showing plots in script mode unless specified

        return metrics, m # Return the trained model as well
    except Exception as e:
        print(f"Prophet error: {e}")
        return {'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf}, None

def train_evaluate_rf(X_train, y_train, X_test, y_test):
    """Trains and evaluates the Random Forest model."""
    try:
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        mae = mean_absolute_error(y_test, rf_pred)
        r2 = r2_score(y_test, rf_pred)
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"Random Forest RMSE: {rmse:.2f}")

        # Plotting
        plt.figure()
        plt.plot(y_test.index, y_test, label='Actual')
        plt.plot(y_test.index, rf_pred, label='RF Prediction', color='orange')
        plt.title(f"Random Forest Forecast on Test Data")
        plt.legend()
        # plt.show()

        return metrics, rf
    except Exception as e:
        print(f"Random Forest error: {e}")
        return {'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf}, None

def train_evaluate_xgb(X_train, y_train, X_test, y_test):
    """Trains and evaluates the XGBoost model."""
    try:
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        mae = mean_absolute_error(y_test, xgb_pred)
        r2 = r2_score(y_test, xgb_pred)
        metrics = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"XGBoost RMSE: {rmse:.2f}")

        # Plotting
        plt.figure()
        plt.plot(y_test.index, y_test, label='Actual')
        plt.plot(y_test.index, xgb_pred, label='XGBoost Prediction', color='green')
        plt.title(f"XGBoost Forecast on Test Data")
        plt.legend()
        # plt.show()

        return metrics, xgb_model
    except Exception as e:
        print(f"XGBoost error: {e}")
        return {'RMSE': np.inf, 'MAE': np.inf, 'R2': -np.inf}, None

def forecast_future(model, model_name, prov_df, exog_cols, future_periods):
    """Retrains the best model on full data and forecasts the future."""
    y_full = prov_df['Number_of_Households']
    X_full = prov_df.drop(['Number_of_Households', 'GEO'], axis=1)
    last_exog_full = X_full.iloc[-1]

    # Define future date range
    future_start_date = prov_df.index.max() + pd.DateOffset(months=1)
    future_index = pd.date_range(start=future_start_date, periods=future_periods, freq='MS')

    # Create future exogenous data: assume constant (last observed) values.
    future_exog = pd.DataFrame(np.tile(last_exog_full.values, (len(future_index), 1)),
                               columns=last_exog_full.index, index=future_index)

    future_forecast = None
    best_model_used = model_name

    if model_name == 'Prophet':
        full_prophet = prov_df.reset_index().rename(columns={'REF_DATE': 'ds', 'Number_of_Households': 'y'})
        m_full = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        for col in exog_cols:
            m_full.add_regressor(col)
        m_full.fit(full_prophet)

        future_full_df = m_full.make_future_dataframe(periods=len(future_index), freq='MS')
         # Fill future exogenous regressors for Prophet
        for col in exog_cols:
            if col in future_full_df.columns:
                last_val = full_prophet[col].iloc[-1]
                future_full_df[col] = last_val
            else:
                 # Add column if missing and fill
                last_val = full_prophet[col].iloc[-1]
                future_full_df[col] = last_val


        forecast_full = m_full.predict(future_full_df)
        # Ensure we only take the forecast for the actual future periods requested
        future_forecast_values = forecast_full.iloc[-len(future_index):]['yhat'].values
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    elif model_name == 'RandomForest':
        rf_full = RandomForestRegressor(random_state=42)
        rf_full.fit(X_full, y_full)
        future_forecast_values = rf_full.predict(future_exog)
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    elif model_name == 'XGBoost':
        xgb_full = xgb.XGBRegressor(random_state=42)
        xgb_full.fit(X_full, y_full)
        future_forecast_values = xgb_full.predict(future_exog)
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    # Plotting future forecast
    plt.figure()
    plt.plot(y_full.index, y_full, label="Historical Data", color="blue")
    if future_forecast is not None:
        plt.plot(future_forecast.index, future_forecast, label=f"Forecast ({best_model_used})", color="green")
    plt.title(f"Future Forecast using {best_model_used}")
    plt.legend()
    # plt.show()

    return future_forecast, best_model_used


def run_housing_forecast(input_csv, metrics_output_csv, forecast_output_csv, future_years=12):
    """Main function to run the housing forecast pipeline."""
    df = load_and_prepare_data(input_csv)

    future_periods = future_years * 12 # Monthly data

    provincial_future_forecasts = []
    province_metrics_summary = {}
    provinces = df['GEO'].unique()

    for prov in provinces:
        print("\n===================================")
        print(f"Processing province: {prov}")

        prov_df = df[df['GEO'] == prov].copy().sort_index()

        if len(prov_df) < 50: # Minimum data threshold
            print(f"Not enough data for province {prov}. Skipping.")
            continue

        # Split data
        n = len(prov_df)
        split_idx = int(n * 0.8)
        train_df = prov_df.iloc[:split_idx]
        test_df = prov_df.iloc[split_idx:]

        # Prepare data for ML models
        y_train = train_df['Number_of_Households']
        y_test = test_df['Number_of_Households']
        X_train = train_df.drop(['Number_of_Households', 'GEO'], axis=1)
        X_test = test_df.drop(['Number_of_Households', 'GEO'], axis=1)

        # Prepare data for Prophet
        prophet_df = prov_df.reset_index().rename(columns={'REF_DATE': 'ds', 'Number_of_Households': 'y'})
        exog_cols = [col for col in prophet_df.columns if col not in ['ds', 'y', 'GEO']]
        train_prophet = prophet_df.iloc[:split_idx].copy()
        test_prophet = prophet_df.iloc[split_idx:].copy()

        # Evaluate models
        metrics = {}
        models = {}

        print("\n--- Evaluating Prophet ---")
        metrics['Prophet'], models['Prophet'] = train_evaluate_prophet(train_prophet, test_prophet, exog_cols)

        print("\n--- Evaluating Random Forest ---")
        metrics['RandomForest'], models['RandomForest'] = train_evaluate_rf(X_train, y_train, X_test, y_test)

        print("\n--- Evaluating XGBoost ---")
        metrics['XGBoost'], models['XGBoost'] = train_evaluate_xgb(X_train, y_train, X_test, y_test)

        province_metrics_summary[prov] = metrics

        # Select best model based on RMSE
        valid_metrics = {k: v['RMSE'] for k, v in metrics.items() if v['RMSE'] != np.inf}
        if not valid_metrics:
            print(f"No valid models found for {prov}. Skipping forecast.")
            continue
        best_model_name = min(valid_metrics, key=valid_metrics.get)
        best_model_instance = models[best_model_name] # Get the trained model instance if needed, though forecast_future retrains
        print(f"\nBest model for {prov} based on RMSE: {best_model_name}")

        # Retrain best model and forecast future
        print(f"\n--- Forecasting Future for {prov} using {best_model_name} ---")
        future_forecast, best_model_used = forecast_future(
            best_model_instance, # Pass the instance (though retrained inside)
            best_model_name,
            prov_df,
            exog_cols, # Needed for Prophet retraining
            future_periods
        )

        if future_forecast is not None:
            forecast_df = pd.DataFrame({
                "Date": future_forecast.index,
                "GEO": prov,
                "Number_of_Houses_Forecast": future_forecast.values,
                "Model_Used": best_model_used
            })
            provincial_future_forecasts.append(forecast_df)
        else:
            print(f"Future forecast failed for {prov}.")


    # Save evaluation metrics
    metrics_list = []
    for prov, models_metrics in province_metrics_summary.items():
        for model_name, m in models_metrics.items():
            metrics_list.append({
                "Province": prov,
                "Model": model_name,
                "RMSE": m['RMSE'],
                "MAE": m['MAE'],
                "R2": m['R2']
            })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_output_dir = os.path.dirname(metrics_output_csv)
    if metrics_output_dir and not os.path.exists(metrics_output_dir):
        os.makedirs(metrics_output_dir)
    metrics_df.to_csv(metrics_output_csv, index=False)
    print(f"\nEvaluation metrics saved to '{metrics_output_csv}'.")

    # Save future forecasts
    if provincial_future_forecasts:
        final_forecast = pd.concat(provincial_future_forecasts, ignore_index=True)
        forecast_output_dir = os.path.dirname(forecast_output_csv)
        if forecast_output_dir and not os.path.exists(forecast_output_dir):
            os.makedirs(forecast_output_dir)
        final_forecast.to_csv(forecast_output_csv, index=False)
        print(f"Provincial forecasts saved to '{forecast_output_csv}'.")
    else:
        print("No future forecasts were generated or saved.")

    plt.close('all') # Close all plot figures

if __name__ == '__main__':
    # Example usage when run directly
    INPUT_CSV = "data/processed/Housing_Features_Engineered.csv"
    METRICS_OUTPUT = "models/housing_model_evaluation_metrics.csv"
    FORECAST_OUTPUT = "models/housing_forecasts_next_12_years.csv"

    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    run_housing_forecast(INPUT_CSV, METRICS_OUTPUT, FORECAST_OUTPUT)

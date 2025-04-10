import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["figure.figsize"] = (14, 6)
plt.style.use("seaborn-v0_8-whitegrid")


def load_and_prepare_data(file_path):
    """Loads and prepares the housing data."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    df = pd.read_csv(file_path)
    df["REF_DATE"] = pd.to_datetime(df["REF_DATE"])
    df.set_index("REF_DATE", inplace=True)
    df.sort_index(inplace=True)
    # print("Data Information:") # Removed logging
    # print(df.info()) # Removed logging
    # print("\nData Head:") # Removed logging
    # print(df.head()) # Removed logging
    return df


def train_evaluate_prophet(train_prophet, test_prophet, exog_cols):
    """Trains and evaluates the Prophet model."""
    try:
        m = Prophet(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        for col in exog_cols:
            m.add_regressor(col)
        m.fit(train_prophet)

        future_test = m.make_future_dataframe(periods=len(test_prophet), freq="MS")
        for col in exog_cols:
            last_train_val = train_prophet[col].iloc[-1]
            future_test[col] = last_train_val

        forecast_prophet = m.predict(future_test)
        prophet_pred = forecast_prophet.iloc[-len(test_prophet) :]["yhat"].values

        rmse = np.sqrt(mean_squared_error(test_prophet["y"], prophet_pred))
        mae = mean_absolute_error(test_prophet["y"], prophet_pred)
        r2 = r2_score(test_prophet["y"], prophet_pred)
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
        # print(f"Prophet RMSE: {rmse:.2f}") # Removed logging

        # Removed commented-out plotting code

        return metrics, m
    except Exception as e:
        print(f"Prophet error: {e}")  # Keep error logging
        return {"RMSE": np.inf, "MAE": np.inf, "R2": -np.inf}, None


def train_evaluate_rf(X_train, y_train, X_test, y_test):
    """Trains and evaluates the Random Forest model."""
    try:
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        mae = mean_absolute_error(y_test, rf_pred)
        r2 = r2_score(y_test, rf_pred)
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
        # print(f"Random Forest RMSE: {rmse:.2f}") # Removed logging

        # Removed commented-out plotting code

        return metrics, rf
    except Exception as e:
        print(f"Random Forest error: {e}")  # Keep error logging
        return {"RMSE": np.inf, "MAE": np.inf, "R2": -np.inf}, None


def train_evaluate_xgb(X_train, y_train, X_test, y_test):
    """Trains and evaluates the XGBoost model."""
    try:
        xgb_model = xgb.XGBRegressor(random_state=42)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        mae = mean_absolute_error(y_test, xgb_pred)
        r2 = r2_score(y_test, xgb_pred)
        metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
        # print(f"XGBoost RMSE: {rmse:.2f}") # Removed logging

        # Removed commented-out plotting code

        return metrics, xgb_model
    except Exception as e:
        print(f"XGBoost error: {e}")  # Keep error logging
        return {"RMSE": np.inf, "MAE": np.inf, "R2": -np.inf}, None


# Modified forecast_future to accept fixed future_index
def forecast_future(model_name, prov_df, exog_cols, future_index):
    """Retrains the best model on full data and forecasts the future using a fixed index."""
    y_full = prov_df["Number_of_Households"]
    X_full = prov_df.drop(["Number_of_Households", "GEO"], axis=1)
    last_exog_full = X_full.iloc[-1]

    future_exog = pd.DataFrame(
        np.tile(last_exog_full.values, (len(future_index), 1)),
        columns=last_exog_full.index,
        index=future_index,
    )

    future_forecast = None
    best_model_used = model_name

    if model_name == "Prophet":
        full_prophet = prov_df.reset_index().rename(
            columns={"REF_DATE": "ds", "Number_of_Households": "y"}
        )
        m_full = Prophet(
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False
        )
        for col in exog_cols:
            m_full.add_regressor(col)
        m_full.fit(full_prophet)

        future_full_df = m_full.make_future_dataframe(
            periods=len(future_index), freq="MS"
        )
        for col in exog_cols:
            last_full_val = full_prophet[col].iloc[-1]
            future_full_df[col] = last_full_val

        forecast_full = m_full.predict(future_full_df)
        future_forecast_values = forecast_full.iloc[-len(future_index) :]["yhat"].values
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    elif model_name == "RandomForest":
        rf_full = RandomForestRegressor(random_state=42)
        rf_full.fit(X_full, y_full)
        future_forecast_values = rf_full.predict(future_exog)
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    elif model_name == "XGBoost":
        xgb_full = xgb.XGBRegressor(random_state=42)
        xgb_full.fit(X_full, y_full)
        future_forecast_values = xgb_full.predict(future_exog)
        future_forecast = pd.Series(future_forecast_values, index=future_index)

    # Removed commented-out plotting code

    return future_forecast, best_model_used


# Modified run_housing_forecast to use fixed future index
def run_housing_forecast(input_csv, metrics_output_csv, forecast_output_csv):
    """Main function to run the housing forecast pipeline."""
    df = load_and_prepare_data(input_csv)

    future_index = pd.date_range(start="2024-01-01", end="2035-12-01", freq="MS")

    provincial_future_forecasts = []
    province_metrics_summary = {}
    provinces = df["GEO"].unique()

    for prov in provinces:
        # print("\n===================================") # Removed logging
        # print(f"Processing province: {prov}") # Removed logging

        prov_df = df[df["GEO"] == prov].copy().sort_index()

        if len(prov_df) < 50:
            print(
                f"Not enough data for province {prov}. Skipping."
            )  # Keep this info message
            continue

        n = len(prov_df)
        split_idx = int(n * 0.8)
        train_df = prov_df.iloc[:split_idx]
        test_df = prov_df.iloc[split_idx:]

        y_train = train_df["Number_of_Households"]
        y_test = test_df["Number_of_Households"]
        X_train = train_df.drop(["Number_of_Households", "GEO"], axis=1)
        X_test = test_df.drop(["Number_of_Households", "GEO"], axis=1)

        prophet_df = prov_df.reset_index().rename(
            columns={"REF_DATE": "ds", "Number_of_Households": "y"}
        )
        exog_cols = [col for col in prophet_df.columns if col not in ["ds", "y", "GEO"]]
        train_prophet = prophet_df.iloc[:split_idx].copy()
        test_prophet = prophet_df.iloc[split_idx:].copy()

        metrics = {}

        # print("\n--- Evaluating Prophet ---") # Removed logging
        metrics["Prophet"], _ = train_evaluate_prophet(
            train_prophet, test_prophet, exog_cols
        )

        # print("\n--- Evaluating Random Forest ---") # Removed logging
        metrics["RandomForest"], _ = train_evaluate_rf(X_train, y_train, X_test, y_test)

        # print("\n--- Evaluating XGBoost ---") # Removed logging
        metrics["XGBoost"], _ = train_evaluate_xgb(X_train, y_train, X_test, y_test)

        province_metrics_summary[prov] = metrics

        valid_metrics = {
            k: v["RMSE"] for k, v in metrics.items() if v["RMSE"] != np.inf
        }
        if not valid_metrics:
            print(
                f"No valid models found for {prov}. Skipping forecast."
            )  # Keep this info message
            continue
        best_model_name = min(valid_metrics, key=valid_metrics.get)
        # print(f"\nBest model for {prov} based on RMSE: {best_model_name}") # Removed logging

        # print(f"\n--- Forecasting Future for {prov} using {best_model_name} ---") # Removed logging
        future_forecast, best_model_used = forecast_future(
            best_model_name, prov_df, exog_cols, future_index
        )

        if future_forecast is not None:
            forecast_df = pd.DataFrame(
                {
                    "Date": future_forecast.index,
                    "GEO": prov,
                    "Number_of_Houses": future_forecast.values,
                }
            )
            provincial_future_forecasts.append(forecast_df)
        else:
            print(f"Future forecast failed for {prov}.")  # Keep this info message

    metrics_list = []
    for prov, models_metrics in province_metrics_summary.items():
        for model_name, m in models_metrics.items():
            metrics_list.append(
                {
                    "Province": prov,
                    "Model": model_name,
                    "RMSE": m["RMSE"],
                    "MAE": m["MAE"],
                    "R2": m["R2"],
                }
            )
    metrics_df = pd.DataFrame(metrics_list)
    metrics_output_dir = os.path.dirname(metrics_output_csv)
    if metrics_output_dir and not os.path.exists(metrics_output_dir):
        os.makedirs(metrics_output_dir)
    metrics_df.to_csv(metrics_output_csv, index=False)
    # print(f"\nEvaluation metrics saved to '{metrics_output_csv}'.") # Removed logging

    if provincial_future_forecasts:
        final_forecast = pd.concat(provincial_future_forecasts, ignore_index=True)
        forecast_output_dir = os.path.dirname(forecast_output_csv)
        if forecast_output_dir and not os.path.exists(forecast_output_dir):
            os.makedirs(forecast_output_dir)
        final_forecast.to_csv(forecast_output_csv, index=False)
        # print(f"Provincial forecasts saved to '{forecast_output_csv}'.") # Removed logging
    else:
        print("No future forecasts were generated or saved.")  # Keep this info message

    plt.close("all")


# Removed __main__ block and trailing # %%

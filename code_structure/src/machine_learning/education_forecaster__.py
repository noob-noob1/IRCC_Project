import pandas as pd
import numpy as np
from prophet import Prophet
from functools import reduce
import os
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def forecast_education_data(input_path, output_path):
    """
    Loads enhanced education data from input_path, forecasts educator numbers using Prophet,
    and saves the forecast to output_path.

    Args:
        input_path (str): Path to the input CSV file (e.g., Education_Features_Enhanced.csv).
        output_path (str): Full path to save the forecasted CSV file (e.g., data/forecasted/Education_Forecast_2024_2035.csv).
    """
    logging.info(f"Starting education forecasting process...")
    logging.info(f"Loading data from: {input_path}")
    # Load dataset
    try:
        edu_df = pd.read_csv(input_path)
        logging.info("Data loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

    # Prepare REF_DATE and GEO
    logging.info("Preparing REF_DATE and GEO columns...")
    try:
        # Check if REF_DATE is already datetime
        if not pd.api.types.is_datetime64_any_dtype(edu_df['REF_DATE']):
             # Try parsing with specific format first
            try:
                edu_df['REF_DATE'] = pd.to_datetime(edu_df['REF_DATE'], format='%d-%m-%Y')
            except ValueError:
                 # Attempt standard parsing if the specific format fails
                edu_df['REF_DATE'] = pd.to_datetime(edu_df['REF_DATE'])
        edu_df['Year'] = edu_df['REF_DATE'].dt.year
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing REF_DATE: {e}. Please ensure the date format is consistent (e.g., DD-MM-YYYY or YYYY-MM-DD).")
        return None

    # Rename Province to GEO if necessary
    if 'Province' in edu_df.columns and 'GEO' not in edu_df.columns:
        edu_df = edu_df.rename(columns={'Province': 'GEO'})
        logging.info("Renamed 'Province' column to 'GEO'.")
    elif 'GEO' not in edu_df.columns:
        logging.error("Neither 'Province' nor 'GEO' column found in the input data.")
        return None

    # Filter from 1991 onward
    logging.info("Filtering data from 1991 onwards.")
    edu_df = edu_df[edu_df['Year'] >= 1991]

    # Define educator columns to forecast
    required_columns = ['Full-time educators', 'Part-time educators', 'Total, work status']
    if not all(col in edu_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in edu_df.columns]
        logging.error(f"Missing required educator columns: {missing_cols}")
        return None

    educator_types = {
        'Full-time educators': 'Full-time',
        'Part-time educators': 'Part-time',
        'Total, work status': 'Total'
    }
    logging.info(f"Forecasting for educator types: {list(educator_types.values())}")

    # Prepare results container
    forecast_by_type = {label: [] for label in educator_types.values()}

    # Forecast for each educator type
    for column, label in educator_types.items():
        logging.info(f"Processing forecast for: {label} ({column})")
        temp_df = edu_df[['REF_DATE', 'GEO', column]].copy()
        # Handle potential duplicate entries before pivoting by averaging
        temp_df = temp_df.groupby(['REF_DATE', 'GEO'])[column].mean().reset_index()
        try:
            pivot_df = temp_df.pivot(index='REF_DATE', columns='GEO', values=column)
        except Exception as e:
            logging.error(f"Error pivoting data for {label}: {e}")
            continue # Skip to next educator type

        pivot_df.index = pd.to_datetime(pivot_df.index)
        # Resample to monthly start frequency and interpolate
        # Using 'time' interpolation, then ffill/bfill for robustness
        monthly_df = pivot_df.resample('MS').interpolate(method='time').ffill().bfill()

        for geo in monthly_df.columns:
            df_geo = monthly_df[[geo]].reset_index()
            df_geo.columns = ['ds', 'y']
            df_geo = df_geo.dropna() # Drop rows with NaN values which Prophet can't handle

            # Remove outliers using IQR
            q1, q3 = df_geo['y'].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            initial_rows = len(df_geo)
            df_geo = df_geo[(df_geo['y'] >= lower_bound) & (df_geo['y'] <= upper_bound)]
            outliers_removed = initial_rows - len(df_geo)
            if outliers_removed > 0:
                logging.debug(f"Removed {outliers_removed} outliers for {label} in {geo}.")


            if len(df_geo) < 10: # Need sufficient data points for Prophet
                logging.warning(f"Skipping {geo} for {label} due to insufficient data points after cleaning ({len(df_geo)}).")
                continue

            try:
                # Initialize Prophet model (match original settings where sensible)
                model = Prophet(
                    yearly_seasonality=True, # Default Prophet yearly seasonality often sufficient
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.5 # From original script
                )
                # Removed custom seasonalities from original script as default yearly might be better
                # and monthly was potentially problematic (period=30.5)
                model.fit(df_geo)

                # Create future dataframe for prediction
                future = pd.date_range(start='2024-01-01', end='2035-12-01', freq='MS')
                future_df = pd.DataFrame({'ds': future})
                forecast = model.predict(future_df)

                # Prepare result structure
                result = forecast[['ds', 'yhat']].copy()
                result.columns = ['REF_DATE', label]
                result['GEO'] = geo
                # Keep REF_DATE as datetime objects for merging, format later

                forecast_by_type[label].append(result)
                logging.debug(f"Successfully forecasted {label} for {geo}.")

            except Exception as e:
                logging.error(f"Error during Prophet forecasting for {label} in {geo}: {e}")

    # Check if any forecasts were generated
    if not any(forecast_by_type.values()):
        logging.error("No forecasts were generated for any educator type. Check data quality and parameters.")
        return None

    # Merge forecasts side-by-side
    logging.info("Merging forecasts for different educator types...")
    merged_forecasts = []
    for label in educator_types.values():
        if forecast_by_type[label]: # Only merge if forecasts exist for this type
            df = pd.concat(forecast_by_type[label], ignore_index=True)
            # Ensure columns are correct before merge attempt
            if label not in df.columns:
                 logging.warning(f"Column '{label}' missing after concat for {label}. Skipping merge for this type.")
                 continue
            df = df[['REF_DATE', 'GEO', label]]
            merged_forecasts.append(df)
        else:
             logging.warning(f"No forecasts generated for {label}, it will be excluded from the final output.")

    if not merged_forecasts:
        logging.error("No valid forecasts available to merge.")
        return None
    elif len(merged_forecasts) == 1:
        final_df = merged_forecasts[0]
        logging.info("Only one educator type was successfully forecasted.")
    else:
        # Combine all forecast types into one DataFrame using outer merge
        try:
            final_df = reduce(lambda left, right: pd.merge(left, right, on=['REF_DATE', 'GEO'], how='outer'), merged_forecasts)
            logging.info("Successfully merged forecasts.")
        except Exception as e:
            logging.error(f"Error merging forecast dataframes: {e}")
            return None

    # Ensure final columns and sort
    final_columns = ['REF_DATE', 'GEO'] + list(educator_types.values())
    # Reorder columns and fill missing ones if any type failed
    final_df = final_df.reindex(columns=final_columns) # Ensure all expected columns are present
    final_df['REF_DATE'] = pd.to_datetime(final_df['REF_DATE']) # Ensure datetime for sorting
    final_df = final_df.sort_values(by=['GEO', 'REF_DATE']).reset_index(drop=True)

    # Format REF_DATE back to string DD-MM-YYYY for output consistency with original script
    final_df['REF_DATE'] = final_df['REF_DATE'].dt.strftime('%d-%m-%Y')

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating output directory {output_dir}: {e}")
        return None

    # Export to CSV
    logging.info(f"Saving forecast to: {output_path}")
    try:
        final_df.to_csv(output_path, index=False)
        logging.info(f"✅ Forecast saved successfully to '{output_path}'")
        # Preview
        logging.info("\nForecast Preview (first 5 rows):\n" + final_df.head().to_string())
        return output_path # Return the path to the saved file
    except Exception as e:
        logging.error(f"Error saving forecast to {output_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage for direct script execution (requires manual path setting)
    logging.info("Running education forecaster script directly (example usage).")
    # Define example paths for direct execution testing
    example_input = os.path.join('data', 'processed', 'Education_Features_Enhanced.csv')
    example_output = os.path.join('data', 'forecasted', 'Education_Forecast_Direct_Run.csv')

    if os.path.exists(example_input):
        logging.info(f"Using example input: {example_input}")
        logging.info(f"Using example output: {example_output}")
        forecast_result_path = forecast_education_data(example_input, example_output)
        if forecast_result_path:
            logging.info(f"Direct run finished. Forecast saved at: {forecast_result_path}")
        else:
            logging.error("Direct run finished with errors. Forecast not saved.")
    else:
        logging.error(f"Example input file not found for direct run: {example_input}")
        logging.error("Cannot run forecast directly without the input file.")

import pandas as pd
import numpy as np
from prophet import Prophet
from functools import reduce
import os
import logging

# Configure logging with a DEBUG level for more granular output
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def forecast_education_data(input_path, output_path):
    """
    Loads enhanced education data from input_path, forecasts educator numbers using Prophet,
    and saves the forecast to output_path.

    Args:
        input_path (str): Path to the input CSV file (e.g., Education_Features_Enhanced.csv).
        output_path (str): Full path to save the forecasted CSV file (e.g., data/forecasted/Education_Forecast_2024_2035.csv).
    """
    print("DEBUG: Starting the education forecasting process...")
    logging.info(f"Starting education forecasting process...")
    print(f"DEBUG: Attempting to load data from: {input_path}")
    logging.info(f"Loading data from: {input_path}")

    # Load dataset
    try:
        edu_df = pd.read_csv(input_path)
        print("DEBUG: Data loaded. Verifying structure...")
        logging.info("Data loaded successfully.")
        print(f"DEBUG: DataFrame shape: {edu_df.shape}, Columns: {list(edu_df.columns)}")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}")
        print(f"ERROR: Input file not found at {input_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print(f"ERROR: Exception while loading data: {e}")
        return None

    # Prepare REF_DATE and GEO
    print("DEBUG: Preparing REF_DATE and GEO columns...")
    logging.info("Preparing REF_DATE and GEO columns...")
    try:
        # Check if REF_DATE column exists
        if 'REF_DATE' not in edu_df.columns:
            raise KeyError("Column 'REF_DATE' not found in the input data.")

        # Attempt parsing with the expected format '%d-%m-%Y', coercing errors
        print(f"DEBUG: Attempting to parse REF_DATE with format '%d-%m-%Y', coercing errors...")
        initial_date_dtype = edu_df['REF_DATE'].dtype # Keep track of original type if needed
        edu_df['REF_DATE_parsed'] = pd.to_datetime(edu_df['REF_DATE'], format='%d-%m-%Y', errors='coerce')
        parsed_count = edu_df['REF_DATE_parsed'].notna().sum()
        total_count = len(edu_df)
        failed_count = total_count - parsed_count

        print(f"DEBUG: Parsed {parsed_count}/{total_count} dates using format '%d-%m-%Y'. {failed_count} failed.")
        logging.info(f"Attempted parsing REF_DATE with format '%d-%m-%Y'. Success: {parsed_count}, Failed: {failed_count}.")

        if failed_count > 0:
            # Log some examples of failed values
            failed_examples = edu_df[edu_df['REF_DATE_parsed'].isna()]['REF_DATE'].unique()
            logging.warning(f"Could not parse {failed_count} REF_DATE values with format '%d-%m-%Y'. Examples: {list(failed_examples)[:5]}")
            print(f"WARNING: Could not parse {failed_count} REF_DATE values. Examples: {list(failed_examples)[:5]}")

            # Drop rows where date parsing failed
            print(f"DEBUG: Dropping {failed_count} rows with unparseable REF_DATE.")
            logging.info(f"Dropping {failed_count} rows with unparseable REF_DATE.")
            edu_df = edu_df.dropna(subset=['REF_DATE_parsed'])

            # Check if any data remains after dropping invalid dates
            if edu_df.empty:
                 print("ERROR: All rows dropped after attempting to parse REF_DATE. No valid dates found in the expected format.")
                 logging.error("All rows dropped after attempting to parse REF_DATE. No valid dates found in format '%d-%m-%Y'.")
                 return None
            print(f"DEBUG: Proceeding with {len(edu_df)} rows after dropping parse failures.")

        # Use the successfully parsed column
        edu_df['REF_DATE'] = edu_df['REF_DATE_parsed']
        edu_df = edu_df.drop(columns=['REF_DATE_parsed']) # Clean up temporary column

        # Extract Year from the correctly parsed dates
        edu_df['Year'] = edu_df['REF_DATE'].dt.year
        unique_years = sorted(edu_df['Year'].unique())
        print(f"DEBUG: REF_DATE converted successfully. Unique years found: {unique_years}")
        logging.info(f"REF_DATE processing complete. Unique years found: {unique_years}")

    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        print(f"ERROR: Missing expected column: {e}")
        return None
    except Exception as e:
        # Catch other potential errors during date processing
        logging.error(f"Error processing REF_DATE: {e}")
        print(f"ERROR: Exception during REF_DATE processing: {e}")
        return None


    # Rename Province to GEO if necessary
    if 'Province' in edu_df.columns and 'GEO' not in edu_df.columns:
        edu_df = edu_df.rename(columns={'Province': 'GEO'})
        logging.info("Renamed 'Province' column to 'GEO'.")
        print("DEBUG: Renamed 'Province' column to 'GEO'.")
    elif 'GEO' not in edu_df.columns:
        logging.error("Neither 'Province' nor 'GEO' column found in the input data.")
        print("ERROR: No 'Province' or 'GEO' column found.")
        return None

    # Filter from 1991 onward
    print("DEBUG: Filtering data from year >= 1991...")
    logging.info("Filtering data from 1991 onwards.")
    initial_count = len(edu_df)
    edu_df = edu_df[edu_df['Year'] >= 1991].copy() # Use .copy() to avoid SettingWithCopyWarning
    filtered_count = len(edu_df)
    print(f"DEBUG: Filtered dataset for Year >= 1991. Rows before: {initial_count}, after: {filtered_count}.")

    # Check if data exists after filtering
    if filtered_count == 0:
        logging.error("No data remaining after filtering for years >= 1991. Check date range in input file.")
        print("ERROR: No data found for years 1991 onwards. Cannot proceed with forecasting.")
        return None

    # Define educator columns to forecast
    required_columns = ['Full-time educators', 'Part-time educators', 'Total, work status']
    if not all(col in edu_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in edu_df.columns]
        logging.error(f"Missing required educator columns: {missing_cols}")
        print(f"ERROR: Missing these educator columns: {missing_cols}")
        return None

    educator_types = {
        'Full-time educators': 'Full-time',
        'Part-time educators': 'Part-time',
        'Total, work status': 'Total'
    }
    logging.info(f"Forecasting for educator types: {list(educator_types.values())}")
    print(f"DEBUG: Forecasting columns: {educator_types}")

    # Prepare results container
    forecast_by_type = {label: [] for label in educator_types.values()}

    # Forecast for each educator type
    for column, label in educator_types.items():
        logging.info(f"Processing forecast for: {label} ({column})")
        print(f"DEBUG: Reading data for column '{column}', storing as '{label}'.")

        # Ensure the column actually exists before selecting (belt-and-suspenders check)
        if column not in edu_df.columns:
            logging.warning(f"Column '{column}' for label '{label}' not found in the filtered data. Skipping.")
            print(f"WARNING: Column '{column}' not found after filtering. Skipping forecast for '{label}'.")
            continue

        # Select only necessary columns and make a copy to avoid modifying the main df slice
        temp_df = edu_df[['REF_DATE', 'GEO', column]].copy()

        # Handle potential duplicate entries before pivoting by averaging
        print(f"DEBUG: Checking for duplicates for {label} based on REF_DATE and GEO...")
        pre_agg_shape = temp_df.shape
        # Only group and aggregate if duplicates actually exist for performance
        if temp_df.duplicated(subset=['REF_DATE', 'GEO']).any():
            print(f"DEBUG: Duplicates found for {label}. Aggregating by mean...")
            temp_df = temp_df.groupby(['REF_DATE', 'GEO'], as_index=False)[column].mean()
            post_agg_shape = temp_df.shape
            print(f"DEBUG: Aggregated duplicates for {label}. Shape before: {pre_agg_shape}, after: {post_agg_shape}")
        else:
            print(f"DEBUG: No duplicates found for {label}. Skipping aggregation.")
            post_agg_shape = pre_agg_shape

        # Pivot the data
        try:
            print(f"DEBUG: Pivoting data for {label}...")
            pivot_df = temp_df.pivot(index='REF_DATE', columns='GEO', values=column)
        except Exception as e:
            # More detailed error for pivoting issues
            logging.error(f"Error pivoting data for {label}: {e}")
            print(f"ERROR: Pivot failed for label '{label}': {e}")
            # Check for duplicate index/column pairs if pivot fails
            duplicates = temp_df[temp_df.duplicated(subset=['REF_DATE', 'GEO'], keep=False)]
            if not duplicates.empty:
                 logging.error(f"Pivot failed likely due to remaining duplicates after aggregation attempt: \n{duplicates.head()}")
                 print(f"ERROR: Pivot failure might be due to duplicate REF_DATE/GEO pairs. Check aggregation logic. Example duplicates:\n{duplicates.head()}")
            continue # Skip to next educator type if pivot fails

        # Ensure index is datetime after pivoting
        pivot_df.index = pd.to_datetime(pivot_df.index)
        print(f"DEBUG: Pivot successful for {label}. Pivot_df shape: {pivot_df.shape}")
        logging.debug(f"Pivot columns for {label}: {list(pivot_df.columns)}")

        # Resample to monthly start frequency and interpolate
        # Use 'linear' interpolation first, as 'time' might require a more strictly monotonic index sometimes
        print(f"DEBUG: Resampling monthly and interpolating for {label}...")
        try:
            monthly_df = pivot_df.resample('MS').interpolate(method='linear').ffill().bfill()
            print(f"DEBUG: Resampled monthly and interpolated for {label}. monthly_df shape: {monthly_df.shape}")
        except Exception as e:
             logging.error(f"Error resampling/interpolating for {label}: {e}")
             print(f"ERROR: Resampling/interpolation failed for {label}: {e}")
             continue # Skip to next educator type

        # Forecast for each GEO within this educator type
        for geo in monthly_df.columns:
            print(f"DEBUG: Processing GEO: {geo} for '{label}' forecast.")
            df_geo = monthly_df[[geo]].reset_index()
            df_geo.columns = ['ds', 'y']
            df_geo = df_geo.dropna() # Drop rows with NaN values (e.g., if interpolation failed at edges)
            logging.debug(f"Data shape after dropping NaN for {label}, {geo}: {df_geo.shape}")

            # Check if enough data points remain after dropping NaNs
            if len(df_geo) < 10: # Prophet generally needs at least a few points
                logging.warning(f"Skipping {geo} for {label} due to insufficient data points after processing ({len(df_geo)}).")
                print(f"DEBUG: Skipped {geo} for {label}, insufficient data after NaN drop: {len(df_geo)} rows.")
                continue

            # Remove outliers using IQR (only if enough data points exist)
            if len(df_geo) >= 20: # Arbitrary threshold to make IQR meaningful
                q1, q3 = df_geo['y'].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                initial_rows = len(df_geo)
                df_geo = df_geo[(df_geo['y'] >= lower_bound) & (df_geo['y'] <= upper_bound)]
                outliers_removed = initial_rows - len(df_geo)
                if outliers_removed > 0:
                    logging.debug(f"Removed {outliers_removed} potential outliers for {label} in {geo}.")
                    print(f"DEBUG: {outliers_removed} potential outliers removed in {geo} for {label}.")
            else:
                print(f"DEBUG: Skipping outlier removal for {geo}, {label} due to low data count ({len(df_geo)}).")


            # Final check on data points before Prophet
            if len(df_geo) < 10: # Re-check after outlier removal
                logging.warning(f"Skipping {geo} for {label} due to insufficient data points after outlier removal ({len(df_geo)}).")
                print(f"DEBUG: Skipped {geo} for {label}, insufficient data after outlier removal: {len(df_geo)} rows.")
                continue

            # Run Prophet forecasting
            try:
                print(f"DEBUG: Initializing Prophet for {geo}, {label} with {len(df_geo)} data points...")
                # Disable stderr logging from CmdStanPy used by Prophet if it gets too noisy
                # logging.getLogger('cmdstanpy').setLevel(logging.WARNING) # Optional: Reduces Stan output
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False, # Data is monthly/yearly originally
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1 # Slightly increased flexibility
                )
                model.fit(df_geo)
                print(f"DEBUG: Prophet model fit complete for {geo}, {label}.")

                # Create future dataframe
                # Ensure future dates align with the monthly start frequency used in resampling
                future = pd.date_range(start='2024-01-01', end='2035-12-01', freq='MS')
                future_df = pd.DataFrame({'ds': future})
                forecast = model.predict(future_df)
                print(f"DEBUG: Prophet forecast generated for {geo}, {label}. Rows: {forecast.shape[0]}")

                # Prepare result structure
                result = forecast[['ds', 'yhat']].copy() # Select forecast value
                result.columns = ['REF_DATE', label] # Rename columns appropriately
                result['GEO'] = geo

                forecast_by_type[label].append(result)
                logging.debug(f"Successfully forecasted {label} for {geo}.")
                print(f"DEBUG: Forecast appended to results container for label '{label}', geo '{geo}'.")

            except Exception as e:
                logging.error(f"Error during Prophet forecasting for {label} in {geo}: {e}", exc_info=True) # Log traceback
                print(f"ERROR: Prophet forecasting for {label} in {geo} failed: {e}")
                # Consider printing df_geo.info() or df_geo.describe() here for debugging specific failures

    # Check if any forecasts were generated
    print("DEBUG: Checking if any forecasts were generated across all educator types...")
    successful_labels = [label for label, forecasts in forecast_by_type.items() if forecasts]

    if not successful_labels:
        logging.error("No forecasts were generated for ANY educator type. Check data quality, date parsing, filtering, and Prophet parameters.")
        print("ERROR: No forecasts generated. Possible causes: All data filtered out, insufficient data per GEO, consistent Prophet errors.")
        return None
    else:
        print(f"DEBUG: Forecasts were generated for: {successful_labels}")
        logging.info(f"Successfully generated forecasts for: {successful_labels}")

    # Merge forecasts side-by-side
    print("DEBUG: Merging forecasts for different educator types...")
    logging.info("Merging forecasts for different educator types...")
    merged_forecasts_dfs = []
    for label in educator_types.values(): # Iterate in defined order
        if label in successful_labels:
            # Concatenate all results for this specific label (across different GEOs)
            df_label = pd.concat(forecast_by_type[label], ignore_index=True)
            # Ensure columns are correct before merge attempt
            if label not in df_label.columns or 'REF_DATE' not in df_label.columns or 'GEO' not in df_label.columns:
                logging.warning(f"Column mismatch after concat for {label}. Skipping merge. Columns: {df_label.columns}")
                print(f"WARNING: Missing expected columns ('REF_DATE', 'GEO', '{label}') after concatenation for {label}. Skipping.")
                continue
            # Keep only the essential columns for merging
            df_label = df_label[['REF_DATE', 'GEO', label]]
            merged_forecasts_dfs.append(df_label)
            print(f"DEBUG: Prepared forecast DataFrame for '{label}' to merge list. Shape: {df_label.shape}")
        else:
            logging.warning(f"No forecasts generated for {label}, it will be excluded from the final output.")
            print(f"WARNING: No forecasts available for {label}, excluding from final merge.")

    # Check again if we have anything to merge after potential skips
    if not merged_forecasts_dfs:
        logging.error("No valid forecast dataframes available to merge after concatenation/validation step.")
        print("ERROR: No valid DataFrames to merge into final output.")
        return None
    elif len(merged_forecasts_dfs) == 1:
        final_df = merged_forecasts_dfs[0]
        logging.info("Only one educator type was successfully forecasted and prepared for output.")
        print("DEBUG: Only one educator type available. Using its DataFrame as final.")
    else:
        # Combine all forecast types into one DataFrame using outer merge
        try:
            print(f"DEBUG: Attempting outer merge on {len(merged_forecasts_dfs)} forecast dataframes...")
            # Start with the first dataframe and iteratively merge others onto it
            final_df = merged_forecasts_dfs[0]
            for i in range(1, len(merged_forecasts_dfs)):
                final_df = pd.merge(final_df, merged_forecasts_dfs[i], on=['REF_DATE', 'GEO'], how='outer')
            logging.info("Successfully merged forecasts using outer join.")
            print(f"DEBUG: Successfully merged all forecast DataFrames via outer join. Final shape pre-sort: {final_df.shape}")
        except Exception as e:
            logging.error(f"Error merging forecast dataframes: {e}", exc_info=True)
            print(f"ERROR: Failed to merge final forecasts: {e}")
            return None

    # Ensure final columns and sort
    # Define expected final columns based on what was successfully forecasted
    final_column_order = ['REF_DATE', 'GEO'] + [label for label in educator_types.values() if label in final_df.columns]
    final_df = final_df[final_column_order] # Reorder and select columns
    final_df['REF_DATE'] = pd.to_datetime(final_df['REF_DATE']) # Ensure it's datetime for sorting
    final_df = final_df.sort_values(by=['GEO', 'REF_DATE']).reset_index(drop=True)
    print(f"DEBUG: Final DataFrame sorted and columns reindexed. Shape: {final_df.shape}")

    # Format REF_DATE back to string DD-MM-YYYY for output consistency
    try:
        final_df['REF_DATE'] = final_df['REF_DATE'].dt.strftime('%d-%m-%Y')
        print("DEBUG: Converted REF_DATE back to string format (DD-MM-YYYY).")
    except Exception as e:
        logging.error(f"Error formatting final REF_DATE to string: {e}")
        print(f"ERROR: Failed to format REF_DATE column to string: {e}")
        # Decide whether to proceed with datetime objects or fail
        # return None # Option: fail if formatting fails

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not output_dir: # Handle case where output_path is just a filename in the current dir
        output_dir = '.'
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"DEBUG: Ensured that the output directory '{output_dir}' exists.")
    except OSError as e:
        logging.error(f"Error creating output directory {output_dir}: {e}")
        print(f"ERROR: Failed to create or access directory: {output_dir}")
        return None

    # Export to CSV
    print(f"DEBUG: Attempting to save forecast to: {output_path}")
    logging.info(f"Saving forecast to: {output_path}")
    try:
        final_df.to_csv(output_path, index=False)
        logging.info(f"✅ Forecast saved successfully to '{output_path}'")
        print(f"DEBUG: Forecast successfully saved to '{output_path}'.")
        # Preview
        print("\nForecast Preview (first 5 rows):")
        preview_str = final_df.head().to_string()
        logging.info("\nForecast Preview (first 5 rows):\n" + preview_str)
        print(preview_str)
        return output_path
    except Exception as e:
        logging.error(f"Error saving forecast to {output_path}: {e}", exc_info=True)
        print(f"ERROR: Failed to save forecast to {output_path}: {e}")
        return None

if __name__ == "__main__":
    logging.info("Running education forecaster script directly (example usage).")
    print("DEBUG: Running script directly with default example paths.")

    # Define example paths relative to the script location or using absolute paths
    # Assuming 'data' folder is in the same directory as the script or project root
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Get script's directory
    project_root = os.path.dirname(script_dir) # Example: if script is in 'src/ml', root is 'src' - adjust as needed!
    # Correctly join paths for cross-platform compatibility
    # Assuming 'data' is one level up from 'src/machine_learning'
    project_root_for_data = os.path.dirname(os.path.dirname(script_dir)) # Go up two levels if script is in src/ml
    example_input = os.path.join(project_root_for_data, 'data', 'processed', 'Education_Features_Enhanced.csv')
    example_output = os.path.join(project_root_for_data, 'data', 'forecasted', 'Education_Forecast_Direct_Run_v2.csv')


    print(f"DEBUG: Using resolved input path: {example_input}")
    print(f"DEBUG: Using resolved output path: {example_output}")

    if os.path.exists(example_input):
        logging.info(f"Found example input: {example_input}")
        logging.info(f"Using example output: {example_output}")
        print(f"DEBUG: Found example input at '{example_input}'. Output will be '{example_output}'")
        forecast_result_path = forecast_education_data(example_input, example_output)
        if forecast_result_path:
            logging.info(f"Direct run finished. Forecast saved at: {forecast_result_path}")
            print(f"DEBUG: Direct run complete. Forecast saved at '{forecast_result_path}'.")
        else:
            logging.error("Direct run finished with errors. Forecast not saved.")
            print("ERROR: Direct run ended with an error. No forecast saved.")
    else:
        logging.error(f"Example input file not found for direct run: {example_input}")
        print(f"ERROR: Example input file not found: {example_input}")
        logging.error("Cannot run forecast directly without the input file.")
import pandas as pd
import numpy as np
from prophet import Prophet
from functools import reduce
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def forecast_education_data(input_path, output_path):
    """
    Loads enhanced education data, forecasts all numeric columns using Prophet,
    handles outliers and negative values, and saves the forecast.

    Args:
        input_path (str): Path to the input CSV file (e.g., Enhanced_Education_Dataset.csv).
        output_path (str): Full path to save the forecasted CSV file (e.g., Education_Forecast_2020_2035.csv).
    """
    logging.info(f"Starting education forecasting process...")
    logging.info(f"Loading data from: {input_path}")

    # Load dataset
    try:
        edu_df = pd.read_csv(input_path)
        logging.info(f"Data loaded successfully. Shape: {edu_df.shape}")
        logging.debug(f"Columns: {list(edu_df.columns)}")
    except FileNotFoundError:
        logging.error(f"Input file not found at {input_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        return None

    # Prepare REF_DATE and GEO
    logging.info("Preparing REF_DATE and GEO columns...")
    try:
        if "REF_DATE" not in edu_df.columns:
            raise KeyError("Column 'REF_DATE' not found in the input data.")

        # Assume 'DD-MM-YYYY' format directly.
        edu_df["REF_DATE_parsed"] = pd.to_datetime(
            edu_df["REF_DATE"], format="%d-%m-%Y", errors="coerce"
        )
        parsed_count = edu_df["REF_DATE_parsed"].notna().sum()
        total_count = len(edu_df)
        failed_count = total_count - parsed_count

        if failed_count > 0:
            failed_examples = edu_df[edu_df["REF_DATE_parsed"].isna()][
                "REF_DATE"
            ].unique()
            logging.warning(
                f"Could not parse {failed_count} REF_DATE values with format '%d-%m-%Y'. Examples: {list(failed_examples)[:5]}"
            )
            logging.info(f"Dropping {failed_count} rows with unparseable REF_DATE.")
            edu_df = edu_df.dropna(subset=["REF_DATE_parsed"])
            if edu_df.empty:
                logging.error(
                    "All rows dropped after attempting to parse REF_DATE. No valid dates found in format '%d-%m-%Y'."
                )
                return None

        edu_df["REF_DATE"] = edu_df["REF_DATE_parsed"]
        edu_df = edu_df.drop(columns=["REF_DATE_parsed"])
        edu_df["Year"] = edu_df["REF_DATE"].dt.year
        unique_years = sorted(edu_df["Year"].unique())
        logging.info(
            f"REF_DATE processing complete. Unique years found: {unique_years}"
        )

    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing REF_DATE: {e}", exc_info=True)
        return None

    # Rename Province to GEO if necessary
    if "Province" in edu_df.columns and "GEO" not in edu_df.columns:
        edu_df = edu_df.rename(columns={"Province": "GEO"})
        logging.info("Renamed 'Province' column to 'GEO'.")
    elif "GEO" not in edu_df.columns:
        logging.error("Neither 'Province' nor 'GEO' column found.")
        return None

    # Filter from 1991 onward
    logging.info("Filtering data from 1991 onwards.")
    initial_count = len(edu_df)
    edu_df = edu_df[
        edu_df["Year"] >= 1991
    ].copy()  # Use copy to avoid SettingWithCopyWarning
    filtered_count = len(edu_df)
    logging.info(
        f"Filtered dataset for Year >= 1991. Rows before: {initial_count}, after: {filtered_count}."
    )

    if filtered_count == 0:
        logging.error("No data remaining after filtering for years >= 1991.")
        return None

    # Identify all numeric columns to forecast
    exclude_cols = {"REF_DATE", "Year", "GEO"}
    forecast_columns = [
        col
        for col in edu_df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(edu_df[col])
    ]
    if not forecast_columns:
        logging.error(
            "No numeric columns found to forecast after excluding REF_DATE, Year, GEO."
        )
        return None
    logging.info(
        f"Identified {len(forecast_columns)} numeric columns for forecasting: {forecast_columns}"
    )

    # Results container
    forecast_by_column = {col: [] for col in forecast_columns}

    # Forecast each numeric column by province
    for column in forecast_columns:
        logging.info(f"--- Processing forecast for column: {column} ---")
        temp_df = edu_df[["REF_DATE", "GEO", column]].copy()

        # Pivot the data
        try:
            logging.debug(f"Pivoting data for {column}...")
            pivot_df = temp_df.pivot(index="REF_DATE", columns="GEO", values=column)
        except ValueError as e:
            # Pivot fails if duplicates exist on ['REF_DATE', 'GEO'] - this mimics script1's potential failure
            logging.error(
                f"Error pivoting data for {column}, likely due to duplicate index/column pairs: {e}",
                exc_info=True,
            )
            duplicates = temp_df[
                temp_df.duplicated(subset=["REF_DATE", "GEO"], keep=False)
            ]
            if not duplicates.empty:
                logging.error(
                    f"Duplicate entries that likely caused pivot failure: \n{duplicates.head()}"
                )
            continue  # Skip to next column if pivot fails

        except Exception as e:
            logging.error(
                f"Unexpected error pivoting data for {column}: {e}", exc_info=True
            )
            continue  # Skip to next column

        pivot_df.index = pd.to_datetime(pivot_df.index)
        logging.debug(f"Pivot successful for {column}. Shape: {pivot_df.shape}")

        # Resample to monthly start frequency and interpolate linearly
        logging.debug(
            f"Resampling monthly and interpolating (linear only) for {column}..."
        )
        try:
            # Only linear interpolation, no ffill/bfill
            monthly_df = pivot_df.resample("MS").interpolate(method="linear")
            logging.debug(
                f"Resampled monthly and interpolated for {column}. Shape: {monthly_df.shape}"
            )
        except Exception as e:
            logging.error(
                f"Error resampling/interpolating for {column}: {e}", exc_info=True
            )
            continue  # Skip to next column

        # Forecast for each GEO within this column
        for geo in monthly_df.columns:
            logging.debug(f"Processing GEO: {geo} for '{column}' forecast.")
            df_geo = monthly_df[[geo]].reset_index()
            df_geo.columns = ["ds", "y"]
            # Prophet likely handles internal NaNs. Interpolation should have filled most NaNs.

            # Remove outliers using IQR
            q1, q3 = df_geo["y"].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            initial_rows = len(df_geo)
            # Apply outlier filter - important to handle NaNs correctly in comparison
            df_geo = df_geo[
                df_geo["y"].between(lower_bound, upper_bound, inclusive="both")
            ]
            outliers_removed = initial_rows - len(df_geo)
            if outliers_removed > 0:
                logging.debug(
                    f"Removed {outliers_removed} potential outliers for {column} in {geo} (IQR method)."
                )

            # Check if enough data points remain
            if len(df_geo) < 10:
                logging.warning(
                    f"Skipping {geo} for {column} due to insufficient data points after outlier removal ({len(df_geo)})."
                )
                continue

            # Run Prophet forecasting
            try:
                logging.debug(
                    f"Initializing Prophet for {geo}, {column} with {len(df_geo)} data points..."
                )
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.5,
                )
                model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
                model.add_seasonality(name="yearly", period=365.25, fourier_order=10)

                model.fit(df_geo)
                logging.debug(f"Prophet model fit complete for {geo}, {column}.")

                # Forecast for 01-01-YYYY from 2020 to 2035
                custom_dates = [f"01-01-{year}" for year in range(2020, 2036)]
                future_df = pd.DataFrame(
                    {"ds": pd.to_datetime(custom_dates, format="%d-%m-%Y")}
                )

                forecast = model.predict(future_df)
                logging.debug(
                    f"Prophet forecast generated for {geo}, {column}. Rows: {forecast.shape[0]}"
                )

                # Prepare result structure
                result = forecast[["ds", "yhat"]].copy()
                result.columns = [
                    "REF_DATE",
                    column,
                ]  # Rename column to the original name
                result["GEO"] = geo

                # Format REF_DATE to string before post-processing numeric column and merge
                result["REF_DATE"] = result["REF_DATE"].dt.strftime("%d-%m-%Y")

                # --- Post-processing ---
                # Replace negative/zero with NaN first
                result[column] = result[column].apply(lambda x: x if x > 0 else np.nan)

                # Fill NaN using linear interpolation
                result[column] = result[column].interpolate(
                    method="linear", limit_direction="both"
                )

                # If still NaN, fill with fallback (5th percentile of historical)
                if result[column].isnull().any():
                    fallback_value = df_geo["y"].quantile(0.05)
                    # Handle case where quantile itself might be NaN (if df_geo is all NaN after filtering)
                    if pd.isna(fallback_value):
                        fallback_value = 0  # Or some other default if needed
                        logging.warning(
                            f"Fallback quantile for {column} in {geo} was NaN. Using 0."
                        )
                    result[column] = result[column].fillna(fallback_value)
                    logging.debug(
                        f"Filled remaining NaNs for {column} in {geo} with fallback value: {fallback_value:.4f}"
                    )
                # --- End Post-processing ---

                forecast_by_column[column].append(result)
                logging.debug(
                    f"Successfully forecasted and post-processed {column} for {geo}."
                )

            except Exception as e:
                logging.error(
                    f"Error during Prophet forecasting for {column} in {geo}: {e}",
                    exc_info=True,
                )

    # Check if any forecasts were generated
    successful_columns = [
        col for col, forecasts in forecast_by_column.items() if forecasts
    ]
    if not successful_columns:
        logging.error("No forecasts were generated for ANY column.")
        return None
    logging.info(f"Successfully generated forecasts for columns: {successful_columns}")

    # Merge forecasts side-by-side
    logging.info("Merging forecasts for different columns...")
    merged_forecasts_dfs = []
    for column in successful_columns:  # Iterate in the order they were processed
        try:
            df_col = pd.concat(forecast_by_column[column], ignore_index=True)
            if (
                column not in df_col.columns
                or "REF_DATE" not in df_col.columns
                or "GEO" not in df_col.columns
            ):
                logging.warning(
                    f"Column mismatch after concat for {column}. Skipping merge. Columns: {df_col.columns}"
                )
                continue
            # Ensure required columns exist before selecting
            df_col = df_col[["REF_DATE", "GEO", column]]  # Keep only essential columns
            merged_forecasts_dfs.append(df_col)
            logging.debug(
                f"Prepared forecast DataFrame for '{column}' to merge. Shape: {df_col.shape}"
            )
        except Exception as e:
            logging.error(
                f"Error concatenating forecasts for column {column}: {e}", exc_info=True
            )

    if not merged_forecasts_dfs:
        logging.error("No valid forecast dataframes available to merge.")
        return None
    elif len(merged_forecasts_dfs) == 1:
        final_df = merged_forecasts_dfs[0]
        logging.info("Only one column was successfully forecasted.")
    else:
        # Combine all forecast columns into one DataFrame using reduce (inner merge)
        try:
            logging.debug(
                f"Attempting inner merge on {len(merged_forecasts_dfs)} forecast dataframes..."
            )
            # Use reduce with 'inner' merge
            final_df = reduce(
                lambda left, right: pd.merge(
                    left, right, on=["REF_DATE", "GEO"], how="inner"
                ),
                merged_forecasts_dfs,
            )
            logging.info("Successfully merged forecasts using inner join.")
        except Exception as e:
            logging.error(f"Error merging forecast dataframes: {e}", exc_info=True)
            return None

    # Ensure final columns and sort
    # Convert REF_DATE back to datetime for sorting, then back to string if needed.
    # Since REF_DATE was converted to string *before* merge, it should already be string here.
    # Let's parse it back for correct sorting, then format again for output.
    try:
        final_df["REF_DATE_dt"] = pd.to_datetime(
            final_df["REF_DATE"], format="%d-%m-%Y"
        )
    except Exception as e:
        logging.warning(
            f"Could not parse REF_DATE back to datetime for sorting: {e}. Sorting may be lexicographical."
        )
        final_df["REF_DATE_dt"] = final_df["REF_DATE"]  # Keep as string if parse fails

    # Reorder columns + GEO, REF_DATE sort
    final_column_order = ["REF_DATE", "GEO"] + [
        col for col in forecast_columns if col in final_df.columns
    ]
    final_df = final_df.sort_values(by=["GEO", "REF_DATE_dt"]).reset_index(drop=True)
    final_df = final_df[final_column_order]  # Apply column order after sorting
    logging.info(f"Final DataFrame sorted and columns ordered. Shape: {final_df.shape}")

    # REF_DATE is already in string format '%d-%m-%Y' from earlier step. No final formatting needed.

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = "."
    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.debug(f"Ensured output directory '{output_dir}' exists.")
    except OSError as e:
        logging.error(f"Error creating output directory {output_dir}: {e}")
        return None

    # Export to CSV
    logging.info(f"Saving forecast to: {output_path}")
    try:
        final_df.to_csv(output_path, index=False)
        logging.info(f"✅ Forecast saved successfully to '{output_path}'")
        # Preview
        logging.info(
            "\nForecast Preview (first 5 rows):\n" + final_df.head().to_string()
        )
        return output_path
    except Exception as e:
        logging.error(f"Error saving forecast to {output_path}: {e}", exc_info=True)
        return None

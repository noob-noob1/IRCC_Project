import pandas as pd
import numpy as np
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_merged_education_data(file_path):
    """Loads the merged education data."""
    if not os.path.exists(file_path):
        logging.error(f"Merged education file not found at {file_path}")
        raise FileNotFoundError(f"Merged education file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded merged education data from {file_path}")
        # Convert REF_DATE (Year) back to datetime for processing
        df["REF_DATE"] = pd.to_datetime(
            df["REF_DATE"].astype(str) + "-01-01", errors="coerce"
        )
        df.dropna(subset=["REF_DATE"], inplace=True)  # Drop if conversion failed
        return df
    except Exception as e:
        logging.error(f"Error loading merged education data from {file_path}: {e}")
        raise


def handle_missing_data(df):
    """Handles missing data by dropping sparse columns and interpolating/filling."""
    logging.info("Handling missing data...")

    # Drop columns with more than 90% missing values
    initial_cols = df.shape[1]
    missing_ratio = df.isnull().mean()
    columns_to_drop = missing_ratio[missing_ratio > 0.90].index.tolist()
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)
        logging.info(
            f"Dropped columns with >90% missing values: {', '.join(columns_to_drop)}"
        )
    else:
        logging.info("No columns found with >90% missing values.")

    # Ensure data is sorted for interpolation
    df.sort_values(by=["GEO", "REF_DATE"], inplace=True)

    # Interpolate numeric columns only within each GEO group
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        logging.info(f"Interpolating numeric columns: {', '.join(numeric_cols)}")
        df[numeric_cols] = df.groupby("GEO")[numeric_cols].transform(
            lambda group: group.interpolate(method="linear")
        )
        logging.info("Linear interpolation completed.")

        # Fill remaining missing values with group-wise (province-level) mean
        logging.info("Filling remaining NaNs with group means...")
        df[numeric_cols] = df.groupby("GEO")[numeric_cols].transform(
            lambda group: group.fillna(group.mean())
        )

        # Final fallback: fill any remaining NaNs with overall column means
        logging.info("Filling final NaNs with overall column means...")
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    else:
        logging.warning("No numeric columns found to interpolate or fill.")

    final_missing = df.isnull().sum().sum()
    logging.info(f"Missing data handling complete. Remaining NaNs: {final_missing}")
    if final_missing > 0:
        logging.warning(
            "Some NaNs might remain in non-numeric columns or if means couldn't be calculated."
        )

    return df


def perform_feature_engineering(df):
    """Creates new features based on existing columns."""
    logging.info("Performing feature engineering...")

    # Replace 0 with NaN temporarily to avoid division by zero, will be filled later if needed
    df_fe = df.replace(0, np.nan)

    # 1. Educator-to-operating spending ratio
    if (
        "Total, work status" in df_fe.columns
        and "Total operating expenditures" in df_fe.columns
    ):
        df_fe["Educator_to_OperatingSpending"] = (
            df_fe["Total, work status"] / df_fe["Total operating expenditures"]
        )
        logging.info("Created feature: Educator_to_OperatingSpending")
    else:
        logging.warning(
            "Skipping Educator_to_OperatingSpending: Required columns missing."
        )

    # 2. Salary-to-EPI ratio
    if (
        "Teachers salaries" in df_fe.columns
        and "Education price index (EPI)" in df_fe.columns
    ):
        df_fe["Salary_to_EPI"] = (
            df_fe["Teachers salaries"] / df_fe["Education price index (EPI)"]
        )
        logging.info("Created feature: Salary_to_EPI")
    else:
        logging.warning("Skipping Salary_to_EPI: Required columns missing.")

    # 3. Operational spend per educator
    if (
        "Total operating expenditures" in df_fe.columns
        and "Total, work status" in df_fe.columns
    ):
        df_fe["OpSpend_per_Educator"] = (
            df_fe["Total operating expenditures"] / df_fe["Total, work status"]
        )
        logging.info("Created feature: OpSpend_per_Educator")
    else:
        logging.warning("Skipping OpSpend_per_Educator: Required columns missing.")

    # 4. Education Access Index: average of participation rates
    participation_cols = ["College", "Elementary and/or High School", "University"]
    available_participation_cols = [
        col for col in participation_cols if col in df_fe.columns
    ]
    if len(available_participation_cols) > 0:
        df_fe["Education_Access_Index"] = df_fe[available_participation_cols].mean(
            axis=1, skipna=True
        )
        logging.info(
            f"Created feature: Education_Access_Index using columns: {', '.join(available_participation_cols)}"
        )
    else:
        logging.warning(
            "Skipping Education_Access_Index: No participation columns found."
        )

    # 5. Capital efficiency
    if (
        "Teachers' salaries sub-index" in df_fe.columns
        and "Total expenditures" in df_fe.columns
    ):
        df_fe["Capital_Efficiency"] = (
            df_fe["Total expenditures"] / df_fe["Teachers' salaries sub-index"]
        )
        logging.info("Created feature: Capital_Efficiency")
    else:
        logging.warning("Skipping Capital_Efficiency: Required columns missing.")

    # Fill NaNs created by division by zero or missing inputs in new features
    new_feature_cols = [
        "Educator_to_OperatingSpending",
        "Salary_to_EPI",
        "OpSpend_per_Educator",
        "Education_Access_Index",
        "Capital_Efficiency",
    ]
    for col in new_feature_cols:
        if col in df_fe.columns:
            # Fill with 0 or another appropriate value, or re-interpolate if makes sense
            df_fe[col].fillna(0, inplace=True)  # Filling with 0 for simplicity

    logging.info("Feature engineering complete.")
    return df_fe


def final_cleanup_and_save(df, output_path):
    """Sorts, formats REF_DATE, and saves the enhanced dataset."""
    logging.info("Performing final cleanup and saving...")
    # Sort by REF_DATE
    df.sort_values(by="REF_DATE", inplace=True)

    # Format REF_DATE to day/month/year (e.g., "01-01-2024")
    df["REF_DATE"] = df["REF_DATE"].dt.strftime("%d-%m-%Y")

    # Save the final dataset
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Enhanced education dataset saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving enhanced education data to {output_path}: {e}")
        raise


def enhance_education_data(input_path, output_path):
    """Main function to load, enhance, and save merged education data."""
    logging.info("Starting education data enhancement pipeline...")
    try:
        df_merged = load_merged_education_data(input_path)
        df_handled_missing = handle_missing_data(df_merged)
        df_enhanced = perform_feature_engineering(df_handled_missing)
        final_cleanup_and_save(df_enhanced, output_path)
        logging.info("Education data enhancement pipeline finished successfully.")
    except FileNotFoundError as e:
        logging.error(f"Aborting enhancement due to missing file: {e}")
    except (KeyError, ValueError) as e:
        logging.error(f"Aborting enhancement due to data error: {e}")
    except Exception as e:
        logging.error(
            f"An unexpected error occurred during the education enhancement pipeline: {e}"
        )

import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(household_path, housing_started_path, nhpi_path):
    """Loads the three processed housing datasets."""
    try:
        df_household = pd.read_csv(household_path)
        logging.info(f"Successfully loaded household data from {household_path}")
    except FileNotFoundError:
        logging.error(f"Household data file not found at {household_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading household data from {household_path}: {e}")
        raise

    try:
        df_housing_started = pd.read_csv(housing_started_path)
        logging.info(f"Successfully loaded housing started data from {housing_started_path}")
    except FileNotFoundError:
        logging.error(f"Housing started data file not found at {housing_started_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading housing started data from {housing_started_path}: {e}")
        raise

    try:
        df_nhpi = pd.read_csv(nhpi_path)
        logging.info(f"Successfully loaded NHPI data from {nhpi_path}")
    except FileNotFoundError:
        logging.error(f"NHPI data file not found at {nhpi_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading NHPI data from {nhpi_path}: {e}")
        raise

    # Convert date columns to datetime objects for reliable merging
    for df, path in [(df_household, household_path), (df_housing_started, housing_started_path), (df_nhpi, nhpi_path)]:
        if 'REF_DATE' in df.columns:
            try:
                df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
            except Exception as e:
                logging.warning(f"Could not convert REF_DATE to datetime in {path}: {e}. Merging might be affected.")
        else:
            logging.warning(f"REF_DATE column not found in {path}. Cannot merge on date.")
            # Depending on requirements, might need to raise an error here

    return df_household, df_housing_started, df_nhpi

def merge_housing_data(df_household, df_housing_started, df_nhpi):
    """Merges the processed housing datasets."""
    logging.info("Starting merge process...")

    # First merge: Household (base) with Housing Started
    logging.info("Merging Household data with Housing Started data...")
    df_merged = pd.merge(df_household, df_housing_started, on=['REF_DATE', 'GEO'], how='left')
    logging.info(f"Shape after first merge: {df_merged.shape}")

    # Second merge: Result with NHPI
    logging.info("Merging intermediate result with NHPI data...")
    df_merged_final = pd.merge(df_merged, df_nhpi, on=['REF_DATE', 'GEO'], how='left')
    logging.info(f"Shape after final merge: {df_merged_final.shape}")

    logging.info("Merge process completed.")
    return df_merged_final

def save_merged_data(df, output_path):
    """Saves the final merged DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Final merged housing data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving merged data to {output_path}: {e}")
        raise

def merge_all_housing_features(household_path, housing_started_path, nhpi_path, output_path):
    """Main function to load, merge, and save housing feature data."""
    logging.info("Starting housing feature merging pipeline...")
    try:
        df_household, df_housing_started, df_nhpi = load_processed_data(
            household_path, housing_started_path, nhpi_path
        )
        df_merged = merge_housing_data(df_household, df_housing_started, df_nhpi)
        save_merged_data(df_merged, output_path)
        logging.info("Housing feature merging pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("One or more input files for merging were not found. Aborting merge.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the merge pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root if run directly
    # This assumes the script is run from the project root directory (e.g., 'c:/ircc project')
    processed_dir = 'data/processed/housing'  # Updated to include housing subdirectory
    household_file = os.path.join(processed_dir, 'Household_Numbers_Processed.csv')
    housing_started_file = os.path.join(processed_dir, 'HousingStarted_Processed.csv')
    nhpi_file = os.path.join(processed_dir, 'NHPI_Processed.csv')
    output_file = os.path.join('data/processed', 'Housing_Features_Merged.csv')

    merge_all_housing_features(household_file, housing_started_file, nhpi_file, output_file)

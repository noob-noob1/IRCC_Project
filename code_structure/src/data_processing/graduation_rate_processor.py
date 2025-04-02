import pandas as pd
import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge_graduation_data(folder_path):
    """Loads and merges all CSV files from the specified folder."""
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logging.error(f"No CSV files found in {folder_path}")
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    dfs = []
    logging.info(f"Loading graduation rate files from {folder_path}...")
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Optional: df['Source_File'] = os.path.basename(file)
            dfs.append(df)
            logging.info(f"Loaded {os.path.basename(file)}")
        except Exception as e:
            logging.error(f"Error loading file {file}: {e}")
            # Decide whether to skip the file or raise the error
            # raise # Uncomment to stop processing if a file fails to load

    if not dfs:
        logging.error("No dataframes were loaded successfully.")
        raise ValueError("No dataframes were loaded successfully.")

    merged_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"Successfully merged {len(dfs)} files. Total rows: {len(merged_df)}")
    return merged_df

def clean_pivot_group_graduation(df):
    """Cleans, pivots, and groups the graduation rate data."""
    logging.info("Cleaning, pivoting, and grouping graduation rate data...")
    # Keep only the year part of REF_DATE
    df['REF_DATE'] = df['REF_DATE'].astype(str).str[:4]

    # Keep only required columns
    required_cols = ['REF_DATE', 'GEO', 'Graduation rate', 'VALUE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {', '.join(missing_cols)}")
        raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")
    df_selected = df[required_cols]

    # Add temporary index for pivoting
    df_selected.insert(0, 'Index', range(1, len(df_selected) + 1))

    # Pivot the table
    try:
        df_pivot = df_selected.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Graduation rate", values='VALUE').reset_index()
    except ValueError as e:
        logging.error(f"Pivot failed. Error: {e}")
        # Handle duplicates if necessary
        logging.info("Attempting pivot after grouping duplicates...")
        df_agg = df_selected.groupby(['REF_DATE', 'GEO', 'Graduation rate'], as_index=False)['VALUE'].mean() # or .first()
        df_agg.insert(0, 'Index', range(1, len(df_agg) + 1))
        df_pivot = df_agg.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Graduation rate", values='VALUE').reset_index()

    # Define aggregation dictionary
    agg_dict = {'Index': 'first'}
    expected_cols = ['Extended-time', 'On-time']
    for col in expected_cols:
         if col in df_pivot.columns:
            agg_dict[col] = 'first' # Assuming unique after pivot/grouping
         else:
             logging.warning(f"Expected column '{col}' not found after pivot. Skipping aggregation.")

    # Group the data by REF_DATE and GEO
    grouped_df = df_pivot.groupby(['REF_DATE', 'GEO'], as_index=False).agg(agg_dict)

    # Drop the temporary Index column
    if 'Index' in grouped_df.columns:
        grouped_df.drop(columns=['Index'], inplace=True)

    # Remove column index name
    grouped_df.columns.name = None
    logging.info("Finished cleaning, pivoting, and grouping.")
    return grouped_df

def filter_and_interpolate_graduation(df):
    """Filters by GEO and interpolates missing graduation rates."""
    logging.info("Filtering by GEO and interpolating graduation rates...")
    # Define the list of GEO values to keep
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    # Filter by GEO
    df_filtered = df[df['GEO'].isin(geo_list)].reset_index(drop=True)

    # Interpolate missing values - ensure REF_DATE is numeric or sortable
    df_filtered['REF_DATE'] = pd.to_numeric(df_filtered['REF_DATE'], errors='coerce')
    df_filtered.sort_values(by=['GEO', 'REF_DATE'], inplace=True)

    rate_cols = ['Extended-time', 'On-time']
    # Check if columns exist before interpolation
    cols_to_interpolate = [col for col in rate_cols if col in df_filtered.columns]
    if not cols_to_interpolate:
         logging.warning("No graduation rate columns found for interpolation.")
    else:
        df_filtered[cols_to_interpolate] = df_filtered.groupby('GEO')[cols_to_interpolate].transform(lambda x: x.interpolate(method='linear'))
        logging.info(f"Interpolation applied to columns: {', '.join(cols_to_interpolate)}")

    logging.info("Finished filtering and interpolating.")
    return df_filtered

def save_processed_graduation_rate(df, output_path):
    """Saves the processed graduation rate DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed graduation rate data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed graduation rate data to {output_path}: {e}")
        raise

def process_graduation_rate_data(input_folder_path, output_path):
    """Main function to process graduation rate data."""
    logging.info("Starting graduation rate data processing pipeline...")
    try:
        df_merged = load_and_merge_graduation_data(input_folder_path)
        df_grouped = clean_pivot_group_graduation(df_merged)
        df_processed = filter_and_interpolate_graduation(df_grouped)
        save_processed_graduation_rate(df_processed, output_path)
        logging.info("Graduation rate data processing pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("Input folder or files for graduation rate processing not found. Aborting.")
    except KeyError as e:
        logging.error(f"Missing expected column during graduation rate processing: {e}. Aborting.")
    except ValueError as e:
         logging.error(f"Value error during graduation rate processing: {e}. Aborting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the graduation rate processing pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root
    raw_dir = 'data/raw/education_datasets/graduation_rates'
    processed_dir = 'data/processed/education'
    output_file = os.path.join(processed_dir, 'GraduationRate_Processed.csv')

    process_graduation_rate_data(raw_dir, output_file)

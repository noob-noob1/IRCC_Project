import pandas as pd
import os
import glob
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_merge_participation_data(folder_path):
    """Loads and merges all CSV files from the specified folder."""
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        logging.error(f"No CSV files found in {folder_path}")
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    dfs = []
    logging.info(f"Loading participation rate files from {folder_path}...")
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

def clean_filter_pivot_group_participation(df):
    """Cleans, filters, pivots, and groups the participation rate data."""
    logging.info("Cleaning, filtering, pivoting, and grouping participation rate data...")

    # Keep only required columns
    rate_col = 'Participation rate by type of institution attended'
    required_cols = ['REF_DATE', 'GEO', rate_col, 'VALUE']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing required columns: {', '.join(missing_cols)}")
        raise KeyError(f"Missing required columns: {', '.join(missing_cols)}")
    df_selected = df[required_cols]

    # Define the list of GEO values to keep
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    # Filter by GEO
    df_filtered_geo = df_selected[df_selected['GEO'].isin(geo_list)].reset_index(drop=True)

    # Keep only the year part of REF_DATE
    df_filtered_geo['REF_DATE'] = df_filtered_geo['REF_DATE'].astype(str).str[:4]

    # Filter out "Total participation rate"
    df_filtered_type = df_filtered_geo[df_filtered_geo[rate_col] != "Total participation rate"]

    # Add temporary index for pivoting
    df_filtered_type.insert(0, 'Index', range(1, len(df_filtered_type) + 1))

    # Pivot the table
    try:
        df_pivot = df_filtered_type.pivot(index=['REF_DATE', 'GEO', 'Index'], columns=rate_col, values='VALUE').reset_index()
    except ValueError as e:
        logging.error(f"Pivot failed. Error: {e}")
        # Handle duplicates if necessary
        logging.info("Attempting pivot after grouping duplicates...")
        df_agg = df_filtered_type.groupby(['REF_DATE', 'GEO', rate_col], as_index=False)['VALUE'].mean() # or .first()
        df_agg.insert(0, 'Index', range(1, len(df_agg) + 1))
        df_pivot = df_agg.pivot(index=['REF_DATE', 'GEO', 'Index'], columns=rate_col, values='VALUE').reset_index()

    # Define aggregation dictionary
    agg_dict = {'Index': 'first'}
    expected_cols = ['College', 'Elementary and/or High School', 'University']
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
    logging.info("Finished cleaning, filtering, pivoting, and grouping.")
    return grouped_df

def save_processed_participation_rate(df, output_path):
    """Saves the processed participation rate DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed participation rate data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed participation rate data to {output_path}: {e}")
        raise

def process_participation_rate_data(input_folder_path, output_path):
    """Main function to process participation rate data."""
    logging.info("Starting participation rate data processing pipeline...")
    try:
        df_merged = load_and_merge_participation_data(input_folder_path)
        df_processed = clean_filter_pivot_group_participation(df_merged)
        save_processed_participation_rate(df_processed, output_path)
        logging.info("Participation rate data processing pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("Input folder or files for participation rate processing not found. Aborting.")
    except KeyError as e:
        logging.error(f"Missing expected column during participation rate processing: {e}. Aborting.")
    except ValueError as e:
         logging.error(f"Value error during participation rate processing: {e}. Aborting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the participation rate processing pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root
    raw_dir = 'data/raw/education_datasets/participation_rates'
    processed_dir = 'data/processed/education'
    output_file = os.path.join(processed_dir, 'ParticipationRate_Processed.csv')

    process_participation_rate_data(raw_dir, output_file)

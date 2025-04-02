import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_epi_data(file_path):
    """Loads the raw EPI data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"Raw EPI file not found at {file_path}")
        raise FileNotFoundError(f"Raw EPI file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded EPI data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading EPI data from {file_path}: {e}")
        raise

def filter_epi_data(df):
    """Filters the EPI DataFrame by GEO and selects relevant columns."""
    logging.info("Filtering EPI data...")
    # Define the list of GEO values to keep
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    # Filter the DataFrame by GEO
    df_filtered = df[df['GEO'].isin(geo_list)].reset_index(drop=True)

    # Keep only required columns
    # Ensure 'Index categories' column exists before selecting
    if 'Index categories' not in df_filtered.columns:
        logging.error("'Index categories' column not found in the filtered data.")
        raise KeyError("'Index categories' column not found.")

    df_selected = df_filtered[['REF_DATE', 'GEO', 'Index categories', 'VALUE']]
    logging.info("Finished filtering EPI data.")
    return df_selected

def pivot_and_group_epi(df):
    """Pivots and groups the EPI data."""
    logging.info("Pivoting and grouping EPI data...")
    # Add temporary index for pivoting
    df.insert(0, 'Index', range(1, len(df) + 1))

    # Pivot the table
    try:
        df_pivot = df.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Index categories", values='VALUE').reset_index()
    except ValueError as e:
        logging.error(f"Pivot failed. This might be due to duplicate entries for the same REF_DATE, GEO, and Index category. Error: {e}")
        # Handle duplicates before pivoting if necessary
        logging.info("Attempting pivot after grouping duplicates...")
        df_agg = df.groupby(['REF_DATE', 'GEO', 'Index categories'], as_index=False)['VALUE'].mean() # or .first()
        df_agg.insert(0, 'Index', range(1, len(df_agg) + 1))
        df_pivot = df_agg.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Index categories", values='VALUE').reset_index()

    # Define aggregation dictionary dynamically based on columns present after pivot
    agg_dict = {'Index': 'first'}
    # Expected columns from the pivot based on 'Index categories' values
    expected_cols = [
        "Education price index (EPI)", "Fees and contractual services sub-index",
        "Instructional supplies sub-index", "Non-salary sub-index",
        "Non-teaching salaries sub-index", "Salaries and wages sub-index",
        "School facilities, supplies and services sub-index", "Teachers' salaries sub-index"
    ]
    for col in expected_cols:
        if col in df_pivot.columns:
            agg_dict[col] = 'first' # Use 'first' assuming unique values after pivot/grouping
        else:
            logging.warning(f"Expected column '{col}' not found after pivot. Skipping aggregation for this column.")


    # Group the data by REF_DATE and GEO
    grouped_df = df_pivot.groupby(['REF_DATE', 'GEO'], as_index=False).agg(agg_dict)

    # Drop the temporary Index column
    if 'Index' in grouped_df.columns:
        grouped_df.drop(columns=['Index'], inplace=True)

    # Remove column index name
    grouped_df.columns.name = None
    logging.info("Finished pivoting and grouping EPI data.")
    return grouped_df

def save_processed_epi(df, output_path):
    """Saves the processed EPI DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed EPI data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed EPI data to {output_path}: {e}")
        raise

def process_epi_data(input_path, output_path):
    """Main function to process EPI data."""
    logging.info("Starting EPI data processing pipeline...")
    try:
        df_raw = load_epi_data(input_path)
        df_filtered = filter_epi_data(df_raw)
        df_processed = pivot_and_group_epi(df_filtered)
        save_processed_epi(df_processed, output_path)
        logging.info("EPI data processing pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("Input file for EPI processing not found. Aborting.")
    except KeyError as e:
        logging.error(f"Missing expected column during EPI processing: {e}. Aborting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the EPI processing pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root
    raw_dir = 'data/raw/education_datasets'
    processed_dir = 'data/processed/education'
    input_file = os.path.join(raw_dir, 'Education price index (EPI), elementary and secondary.csv')
    output_file = os.path.join(processed_dir, 'EPI_Processed.csv')

    process_epi_data(input_file, output_file)

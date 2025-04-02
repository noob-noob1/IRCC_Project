import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_educators_data(file_path):
    """Loads the raw educators data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"Raw educators file not found at {file_path}")
        raise FileNotFoundError(f"Raw educators file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded educators data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading educators data from {file_path}: {e}")
        raise

def clean_and_filter_educators(df):
    """Cleans and filters the educators DataFrame."""
    logging.info("Cleaning and filtering educators data...")
    # Keep only the year part of REF_DATE
    df['REF_DATE'] = df['REF_DATE'].astype(str).str[:4]

    # Define the list of GEO values to keep
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    # Filter the DataFrame
    df_filtered = df[df['GEO'].isin(geo_list)].reset_index(drop=True)

    # Keep only required columns
    df_selected = df_filtered[['REF_DATE', 'GEO', 'Work status', 'VALUE']]
    logging.info("Finished cleaning and filtering educators data.")
    return df_selected

def pivot_and_group_educators(df):
    """Pivots and groups the educators data."""
    logging.info("Pivoting and grouping educators data...")
    # Add temporary index for pivoting if needed (though groupby might handle duplicates)
    df.insert(0, 'Index', range(1, len(df) + 1))

    # Pivot the table
    try:
        df_pivot = df.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Work status", values='VALUE').reset_index()
    except ValueError as e:
        logging.error(f"Pivot failed. This might be due to duplicate entries for the same REF_DATE, GEO, and Work status. Error: {e}")
        # Handle duplicates before pivoting if necessary, e.g., by averaging or taking the first
        logging.info("Attempting pivot after grouping duplicates...")
        df_agg = df.groupby(['REF_DATE', 'GEO', 'Work status'], as_index=False)['VALUE'].mean() # or .first()
        df_agg.insert(0, 'Index', range(1, len(df_agg) + 1))
        df_pivot = df_agg.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Work status", values='VALUE').reset_index()


    # Group the data by REF_DATE and GEO
    grouped_df = df_pivot.groupby(['REF_DATE', 'GEO'], as_index=False).agg({
        # 'Index': 'first', # Keep index if needed for debugging, otherwise remove
        'Full-time educators': 'first', # Use 'first' or 'sum'/'mean' depending on data structure
        'Part-time educators': 'first',
        'Total, work status': 'first'
    })

    # Remove column index name
    grouped_df.columns.name = None
    logging.info("Finished pivoting and grouping educators data.")
    return grouped_df

def save_processed_educators(df, output_path):
    """Saves the processed educators DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed educators data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed educators data to {output_path}: {e}")
        raise

def process_educators_data(input_path, output_path):
    """Main function to process educators data."""
    logging.info("Starting educators data processing pipeline...")
    try:
        df_raw = load_educators_data(input_path)
        df_cleaned = clean_and_filter_educators(df_raw)
        df_processed = pivot_and_group_educators(df_cleaned)
        save_processed_educators(df_processed, output_path)
        logging.info("Educators data processing pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("Input file for educators processing not found. Aborting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the educators processing pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root
    raw_dir = 'data/raw/education_datasets'
    processed_dir = 'data/processed/education'
    input_file = os.path.join(raw_dir, 'Educators in public elementary and secondary schools.csv')
    output_file = os.path.join(processed_dir, 'Educators_Processed.csv')

    process_educators_data(input_file, output_file)

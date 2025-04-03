import pandas as pd
import os

def load_nhpi_data(file_path):
    """Loads the raw NHPI data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw NHPI file not found at {file_path}")
    return pd.read_csv(file_path)

def transform_nhpi_data(df):
    """Transforms the raw NHPI data."""
    # Convert REF_DATE to datetime, handling potential errors
    try:
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], format="%b-%y")
    except ValueError as e:
        print(f"Warning: Could not parse all dates in REF_DATE column with format '%b-%y'. Error: {e}")
        # Attempt conversion with a more general parser, coercing errors
        df["REF_DATE"] = pd.to_datetime(df["REF_DATE"], errors='coerce')
        # Drop rows where date conversion failed if necessary
        df.dropna(subset=["REF_DATE"], inplace=True)


    # Select relevant columns
    df_selected = df[["REF_DATE", "GEO", "New housing price indexes", "VALUE"]]

    # Pivot the table
    df_pivoted = df_selected.pivot(index=['REF_DATE', 'GEO'], columns="New housing price indexes", values='VALUE')
    df_pivoted.reset_index(inplace=True)
    df_pivoted.columns.name = None # Remove the columns index name

    # Rename columns for clarity
    df_pivoted.rename(columns={
        'House only': 'House only NHPI',
        'Land only': 'Land only NHPI',
        'Total (house and land)': 'Total (house and land) NHPI'
    }, inplace=True)

    return df_pivoted

def filter_nhpi_by_geo(df):
    """Filters the NHPI data for specific GEO values."""
    geo_list = [
        "Alberta", "Canada", "British Columbia", "Manitoba", "New Brunswick",
        "Newfoundland and Labrador", "Nova Scotia", "Ontario",
        "Prince Edward Island", "Quebec", "Saskatchewan"
    ]
    filtered_df = df[df["GEO"].isin(geo_list)].reset_index(drop=True)
    return filtered_df

def save_processed_nhpi(df, output_path):
    """Saves the processed NHPI DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed NHPI data saved to {output_path}")

def process_nhpi_data(input_path, output_path):
    """Main function to process NHPI data."""
    df_raw = load_nhpi_data(input_path)
    df_transformed = transform_nhpi_data(df_raw)
    df_filtered = filter_nhpi_by_geo(df_transformed)
    save_processed_nhpi(df_filtered, output_path)

if __name__ == '__main__':
    # Example usage: Adjust paths as necessary if run directly
    # Assumes the script is run from within the src/data_processing directory
    raw_file = '../../data/raw/NHPI.csv'
    processed_file = '../../data/processed/housing/NHPI_Processed.csv'  # Updated path
    try:
        process_nhpi_data(raw_file, processed_file)
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred during NHPI processing: {e}")

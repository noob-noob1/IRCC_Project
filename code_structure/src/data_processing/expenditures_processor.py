import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_expenditures_data(file_path):
    """Loads the raw expenditures data from a CSV file."""
    if not os.path.exists(file_path):
        logging.error(f"Raw expenditures file not found at {file_path}")
        raise FileNotFoundError(f"Raw expenditures file not found at {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded expenditures data from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading expenditures data from {file_path}: {e}")
        raise

def clean_and_filter_expenditures(df):
    """Cleans and filters the expenditures DataFrame."""
    logging.info("Cleaning and filtering expenditures data...")
    # Keep only the year part of REF_DATE
    df['REF_DATE'] = df['REF_DATE'].astype(str).str[:4]

    # Keep only required columns
    if 'Type of expenditure' not in df.columns:
        logging.error("'Type of expenditure' column not found.")
        raise KeyError("'Type of expenditure' column not found.")
    df_selected = df[['REF_DATE', 'GEO', 'Type of expenditure', 'VALUE']]

    # Filter by expenditure type
    expenditure_types = [
        "Total expenditures",
        "Total operating expenditures",
        "Teachers salaries",
        "Capita outlay and debt charges"
    ]
    df_filtered_type = df_selected[df_selected["Type of expenditure"].isin(expenditure_types)]

    # Define the list of GEO values to keep
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    # Filter by GEO
    df_filtered_geo = df_filtered_type[df_filtered_type['GEO'].isin(geo_list)].reset_index(drop=True)

    logging.info("Finished cleaning and filtering expenditures data.")
    return df_filtered_geo

def pivot_group_interpolate_expenditures(df):
    """Pivots, groups, and interpolates the expenditures data."""
    logging.info("Pivoting, grouping, and interpolating expenditures data...")
    # Add temporary index for pivoting
    df.insert(0, 'Index', range(1, len(df) + 1))

    # Pivot the table
    try:
        df_pivot = df.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Type of expenditure", values='VALUE').reset_index()
    except ValueError as e:
        logging.error(f"Pivot failed. Error: {e}")
        # Handle duplicates if necessary
        logging.info("Attempting pivot after grouping duplicates...")
        df_agg = df.groupby(['REF_DATE', 'GEO', 'Type of expenditure'], as_index=False)['VALUE'].mean() # or .first()
        df_agg.insert(0, 'Index', range(1, len(df_agg) + 1))
        df_pivot = df_agg.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Type of expenditure", values='VALUE').reset_index()


    # Define aggregation dictionary
    agg_dict = {'Index': 'first'}
    expected_cols = [
        'Capita outlay and debt charges', 'Teachers salaries',
        'Total expenditures', 'Total operating expenditures'
    ]
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

    # Interpolate missing values - ensure REF_DATE is numeric or sortable for interpolation
    grouped_df['REF_DATE'] = pd.to_numeric(grouped_df['REF_DATE'], errors='coerce')
    grouped_df.sort_values(by=['GEO', 'REF_DATE'], inplace=True)
    # Interpolate only numeric columns, excluding GEO and REF_DATE if REF_DATE wasn't numeric originally
    cols_to_interpolate = grouped_df.select_dtypes(include='number').columns.difference(['REF_DATE']) # Adjust if REF_DATE needs interpolation
    grouped_df[cols_to_interpolate] = grouped_df.groupby('GEO')[cols_to_interpolate].transform(lambda x: x.interpolate(method='linear'))
    logging.info("Interpolation applied.")

    # Drop the 'Capita outlay and debt charges' column if it exists
    if 'Capita outlay and debt charges' in grouped_df.columns:
        grouped_df = grouped_df.drop(columns=["Capita outlay and debt charges"])
        logging.info("Dropped 'Capita outlay and debt charges' column.")
    else:
        logging.warning("'Capita outlay and debt charges' column not found for dropping.")


    logging.info("Finished pivoting, grouping, and interpolating.")
    return grouped_df

def save_processed_expenditures(df, output_path):
    """Saves the processed expenditures DataFrame to a CSV file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Processed expenditures data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving processed expenditures data to {output_path}: {e}")
        raise

def process_expenditures_data(input_path, output_path):
    """Main function to process expenditures data."""
    logging.info("Starting expenditures data processing pipeline...")
    try:
        df_raw = load_expenditures_data(input_path)
        df_cleaned = clean_and_filter_expenditures(df_raw)
        df_processed = pivot_group_interpolate_expenditures(df_cleaned)
        save_processed_expenditures(df_processed, output_path)
        logging.info("Expenditures data processing pipeline finished successfully.")
    except FileNotFoundError:
        logging.error("Input file for expenditures processing not found. Aborting.")
    except KeyError as e:
        logging.error(f"Missing expected column during expenditures processing: {e}. Aborting.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the expenditures processing pipeline: {e}")

import pandas as pd
import os
from itertools import product
import glob

def process_file(input_file):
    """Processes an input CSV file, interpolates missing values, and returns a cleaned DataFrame."""
    # Read CSV
    df = pd.read_csv(input_file, encoding='latin1', header=2)

    # Remove unnecessary rows
    df = pd.concat([df.iloc[:8], df.iloc[13:18]]).reset_index(drop=True)

    # Keep only relevant columns
    df = df.iloc[:, [1, 2]]
    df.columns = ['Year', 'Number_of_Households']

    # Convert Year column to numeric
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Set Year as index
    df.set_index('Year', inplace=True)

    # Reindex for missing years
    df = df.reindex(range(int(df.index.min()), 2037))

    # Convert data column to numeric
    df['Number_of_Households'] = pd.to_numeric(df['Number_of_Households'], errors='coerce')

    # Interpolate missing values
    df['Number_of_Households'] = df['Number_of_Households'].interpolate(method='linear').round()

    # Fill remaining NaN with 0 and convert to integer
    df['Number_of_Households'] = df['Number_of_Households'].fillna(0).astype(int)

    # Multiply the values by 1000
    df['Number_of_Households'] = df['Number_of_Households'] * 1000

    # Reset index
    df = df.reset_index()

    return df

def process_household_data(raw_data_dir, processed_output_path):
    """
    Processes raw household data files, merges them, aligns dates,
    standardizes names, and saves the final dataset.
    """
    # Find all CSV files in the raw data directory
    file_paths = glob.glob(os.path.join(raw_data_dir, "number-of-households-canada-provinces-*.csv"))

    if not file_paths:
        print(f"Error: No CSV files found in {raw_data_dir}")
        return

    # Process each file
    processed_data = {}
    for file in file_paths:
        # Extract province name robustly, handling potential variations
        base_name = os.path.basename(file)
        parts = base_name.replace("number-of-households-canada-provinces-", "").replace(".csv", "")
        province_name = parts # Assuming the remaining part is the province name
        try:
            processed_df = process_file(file)
            processed_data[province_name] = processed_df.rename(columns={"Number_of_Households": province_name})
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue # Skip this file and continue with others

    if not processed_data:
        print("Error: No files were successfully processed.")
        return

    # Merge all datasets
    merged_df = list(processed_data.values())[0]
    for province, df in processed_data.items():
        if province != list(processed_data.keys())[0]:
            # Ensure 'Year' column exists and is suitable for merging
            if 'Year' in df.columns:
                 merged_df = pd.merge(merged_df, df, on="Year", how="outer")
            else:
                print(f"Warning: 'Year' column missing in processed data for {province}. Skipping merge.")


    # Adding Canada data
    # Ensure we only sum numeric columns, excluding 'Year'
    numeric_cols = merged_df.select_dtypes(include='number').columns.difference(['Year'])
    merged_df["Canada"] = merged_df[numeric_cols].sum(axis=1)

    """
    # Load the reference dataset to align the date range
    try:
        reference_df = pd.read_csv(reference_data_path, parse_dates=["REF_DATE"])
        min_date = reference_df["REF_DATE"].min()
        max_date = reference_df["REF_DATE"].max()
        common_date_range = pd.date_range(start=min_date, end=max_date, freq='MS')  # Start of the month
    except FileNotFoundError:
        print(f"Error: Reference data file not found at {reference_data_path}")
        return
    except Exception as e:
        print(f"Error reading or processing reference data file {reference_data_path}: {e}")
        return
    """
    try:
        min_year = merged_df['Year'].min()
        # Use the known max year from interpolation or calculate from data
        max_year = 2036 # Or merged_df['Year'].max() if you don't want the fixed interpolation end
        start_date = pd.Timestamp(f'{min_year}-01-01')
        end_date = pd.Timestamp(f'{max_year}-12-01') # Last month start of the max year
        common_date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    except Exception as e:
        print(f"Error creating date range from processed data: {e}")
        return # Or handle error appropriately
    

    # Creating a new DataFrame with 'REF_DATE' and 'GEO' columns
    time_series_data = pd.DataFrame(list(product(common_date_range, merged_df.columns.difference(['Year']))), columns=["REF_DATE", "GEO"])

    # Mapping values from merged dataset to new format
    # Ensure 'Year' column exists before attempting the merge/lookup
    if 'Year' in merged_df.columns:
        time_series_data["Number_of_Households"] = time_series_data.apply(
            lambda row: merged_df.loc[merged_df["Year"] == row["REF_DATE"].year, row["GEO"]].iloc[0]
                        if row["GEO"] in merged_df.columns and not merged_df.loc[merged_df["Year"] == row["REF_DATE"].year, row["GEO"]].empty
                        else None,
            axis=1
        )
    else:
        print("Warning: 'Year' column missing in merged_df. Cannot map household numbers.")
        time_series_data["Number_of_Households"] = None # Assign default value


    # Convert REF_DATE to YYYY-MM-DD format
    time_series_data["REF_DATE"] = time_series_data["REF_DATE"].dt.strftime("%Y-%m-%d")

    # Standardizing province names
    province_mapping = {
        "Alberta": "Alberta",
        "BC": "British Columbia",
        "Manitoba": "Manitoba",
        "NewBrunswick": "New Brunswick",
        "NewFoundland": "Newfoundland and Labrador",
        "Ontario": "Ontario",
        "PEI": "Prince Edward Island",
        "quebec": "Quebec",
        "Canada": "Canada"
    }

    # Trim whitespace and standardize names
    time_series_data["GEO"] = time_series_data["GEO"].str.strip().map(province_mapping)

    # Remove rows where 'GEO' is empty or NaN after mapping
    time_series_data = time_series_data[time_series_data["GEO"].notna() & (time_series_data["GEO"] != "")]

    # Handle any remaining missing values in 'Number_of_Households'
    time_series_data["Number_of_Households"] = time_series_data["Number_of_Households"].fillna(0).astype(int)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(processed_output_path), exist_ok=True)

    # Save the final dataset
    try:
        time_series_data.to_csv(processed_output_path, index=False)
        print(f"Processing complete. Cleaned dataset saved as '{processed_output_path}'.")
    except Exception as e:
        print(f"Error saving processed data to {processed_output_path}: {e}")

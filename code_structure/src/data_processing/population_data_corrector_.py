import pandas as pd

def correct_population_dataset(input_file_path, canada_population_file):
    """
    Corrects the population dataset by removing duplicates, integrating official Canada population data,
    and recalculating Canada totals based on provincial data where necessary.

    Args:
        input_file_path (str): Path to the raw population timeseries CSV file.
        canada_population_file (str): Path to the CSV file containing official Canada population data.

    Returns:
        pandas.DataFrame: The corrected DataFrame with unique entries per Year-Province
                          and properly integrated Canada data.
    """
    # Load the dataset
    df = pd.read_csv(input_file_path)

    # Remove duplicate entries by keeping only the first occurrence per Year-Province pair
    df_unique = df.drop_duplicates(subset=["Year", "Province"])

    # Load the official Canada population dataset
    canada_population_df = pd.read_csv(canada_population_file)

    # Ensure Canada data aligns with the time series by keeping only relevant years
    relevant_years = df_unique["Year"].unique()
    canada_population_df = canada_population_df[canada_population_df["Year"].isin(relevant_years)]

    # Standardize column names for merging - ensure all expected columns exist
    # Define expected columns based on the notebook logic
    expected_columns = ["Year", "Province", "Total PRs", "Total TRs", "Total Births", "Total Deaths", "Population Estimate"]

    # Filter both dataframes to only include expected columns, handling potential missing ones gracefully
    df_unique = df_unique[[col for col in expected_columns if col in df_unique.columns]]
    canada_population_df = canada_population_df[[col for col in expected_columns if col in canada_population_df.columns]]

    # Remove any previously added "Canada" rows from the main dataset to avoid duplication before recalculation
    df_corrected = df_unique[df_unique["Province"] != "Canada"].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Calculate the total values for Canada from the other provinces
    # Ensure numeric columns are indeed numeric before summing
    numeric_cols_to_sum = ['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths', 'Population Estimate']
    for col in numeric_cols_to_sum:
        if col in df_corrected.columns:
            df_corrected[col] = pd.to_numeric(df_corrected[col], errors='coerce')

    # Group by Year and sum, handling potential NaNs introduced by coerce
    grouped_df = df_corrected.groupby('Year')[numeric_cols_to_sum].sum(min_count=1).reset_index() # min_count=1 ensures sum is NaN only if all values are NaN
    grouped_df['Province'] = 'Canada'

    # Merge the calculated Canada data with the official Canada population data
    # Use official 'Population Estimate' but calculated sums for other metrics if official data is missing
    canada_population_df = canada_population_df.set_index('Year')
    grouped_df = grouped_df.set_index('Year')

    # Prioritize official Canada data, fill missing values with calculated sums
    merged_canada_df = grouped_df.combine_first(canada_population_df)

    # Ensure 'Population Estimate' specifically uses the official data if available
    if 'Population Estimate' in canada_population_df.columns:
         merged_canada_df['Population Estimate'] = canada_population_df['Population Estimate'].combine_first(grouped_df['Population Estimate'])


    merged_canada_df = merged_canada_df.reset_index()
    # Re-order columns to match expected_columns order
    merged_canada_df = merged_canada_df[[col for col in expected_columns if col in merged_canada_df.columns]]


    # Append the corrected Canada data to the main dataset (provincial data)
    # Ensure columns match before concat
    df_final = pd.concat([df_corrected, merged_canada_df], ignore_index=True, sort=False)


    # Sort data to align Canada data with provinces within each year
    df_final = df_final.sort_values(by=["Year", "Province"]).reset_index(drop=True)

    # Ensure final dataframe only contains the expected columns in the correct order
    df_final = df_final[[col for col in expected_columns if col in df_final.columns]]


    return df_final

if __name__ == '__main__':
    # Example Usage (adjust paths for the new structure)
    # These paths assume the script is run from the root 'ircc project' directory
    # Or that the data files are appropriately placed relative to the script location.
    # For a production setup, paths should ideally be managed via config files or arguments.

    # Assuming raw data is in 'data/raw' and metadata might be alongside or structured
    # The original notebook seemed to read from the same directory it was in.
    # Let's assume the necessary CSVs are copied/placed into 'data/raw' for this example.
    raw_file_path = "data/raw/Population Timeseries.csv" # Adjust if needed
    canada_pop_file = "data/raw/canada_population_data.csv" # Adjust if needed
    output_file_path = "data/processed/Population_Demographics_Corrected.csv" # Save to processed

    print(f"Processing {raw_file_path} and {canada_pop_file}...")
    try:
        corrected_df = correct_population_dataset(raw_file_path, canada_pop_file)
        corrected_df.to_csv(output_file_path, index=False)
        print(f"Corrected dataset saved successfully to {output_file_path}!")

        # Display sample data
        print("\nSample of corrected data (Canada entries):")
        print(corrected_df[corrected_df["Province"] == "Canada"].head())
        print("\nSample of corrected data (First 20 rows):")
        print(corrected_df.head(20))

    except FileNotFoundError as e:
        print(f"Error: Input file not found. Please ensure '{e.filename}' exists.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
    Q
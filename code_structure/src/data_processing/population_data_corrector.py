import pandas as pd

def correct_population_dataset(file_path, canada_population_file):
    """
    Corrects the population dataset by removing duplicates, aligning with official
    Canada population data, and ensuring consistent structure.

    Args:
        file_path (str): Path to the main population timeseries CSV file, relative to 'data/raw/'.
                         Expected columns: Year, Province, Total PRs, Total TRs,
                         Total Births, Total Deaths, Population Estimate.
        canada_population_file (str): Path to the official Canada population data CSV file, relative to 'data/raw/'.
                                      Expected columns match file_path.

    Returns:
        pandas.DataFrame: A DataFrame with corrected and consolidated population data,
                          sorted by Year and Province.
    """

    # Load the datasets directly using the provided paths
    try:
        df = pd.read_csv(file_path)
        canada_population_df = pd.read_csv(canada_population_file)
    except FileNotFoundError as e:
        # Updated error message reflects that raw_data_path is no longer used
        print(f"Error loading file: {e}. Ensure '{file_path}' and '{canada_population_file}' specify correct paths.")
        raise

    # Remove duplicate entries by keeping only the first occurrence per Year-Province pair
    df_unique = df.drop_duplicates(subset=["Year", "Province"])

    # Ensure Canada data aligns with the time series by keeping only relevant years
    unique_years = df_unique["Year"].unique()
    canada_population_df = canada_population_df[canada_population_df["Year"].isin(unique_years)]

    # Standardize column names for merging (ensure these columns exist in both files)
    common_columns = ["Year", "Province", "Total PRs", "Total TRs", "Total Births", "Total Deaths", "Population Estimate"]
    # Select only common columns that actually exist in the dataframe to avoid errors
    df_unique_cols = [col for col in common_columns if col in df_unique.columns]
    canada_pop_cols = [col for col in common_columns if col in canada_population_df.columns]

    # Ensure both dataframes have the same set of common columns after filtering
    final_common_columns = list(set(df_unique_cols) & set(canada_pop_cols))
    if not final_common_columns:
        raise ValueError("No common columns found between the two dataframes after filtering for expected columns.")
    if 'Year' not in final_common_columns or 'Province' not in final_common_columns:
        raise ValueError("Required columns 'Year' and 'Province' are missing after column filtering.")


    df_unique = df_unique[final_common_columns]
    canada_population_df = canada_population_df[final_common_columns]


    # Remove any previously added "Canada" rows from the main dataset to avoid duplication
    df_corrected = df_unique[df_unique["Province"] != "Canada"].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Calculate the total values for Canada from the other provinces
    # Ensure only numeric columns (excluding Year) are summed
    numeric_cols = df_corrected.select_dtypes(include='number').columns.tolist()
    if 'Year' in numeric_cols:
        numeric_cols.remove('Year') # Don't sum the Year column

    if not numeric_cols:
        print("Warning: No numeric columns found to sum for Canada totals (excluding Year).")
        # Create an empty DataFrame with expected structure if no numeric cols
        grouped_df = pd.DataFrame(columns=['Year'] + numeric_cols)
        grouped_df['Province'] = 'Canada'

    else:
        grouped_df = df_corrected.groupby('Year')[numeric_cols].sum().reset_index()
        grouped_df['Province'] = 'Canada'

    # Merge the calculated Canada data with the official Canada population data
    # Use 'Year' and 'Province' as keys for merging/updating
    canada_population_df = canada_population_df.set_index(['Year', 'Province'])
    grouped_df = grouped_df.set_index(['Year', 'Province'])

    # Update official Canada data with calculated sums where official data might be missing
    # Only update columns present in grouped_df (calculated sums)
    update_cols = [col for col in ['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths'] if col in grouped_df.columns and col in canada_population_df.columns]
    if update_cols:
         # Use combine_first: fills NaNs in canada_population_df with values from grouped_df
        canada_population_df[update_cols] = canada_population_df[update_cols].combine_first(grouped_df[update_cols])


    # Ensure 'Population Estimate' from official data is preserved if it exists
    # combine_first already handles this: keeps existing non-NaN values in canada_population_df

    canada_population_df = canada_population_df.reset_index()

    # Append the corrected/updated Canada data to the main dataset
    # Ensure columns match before concat
    final_columns = df_corrected.columns.tolist() # Use columns from the provincial data as the standard
    canada_population_df = canada_population_df.reindex(columns=final_columns) # Align columns, fills missing with NaN

    df_final = pd.concat([df_corrected, canada_population_df], ignore_index=True)

    # Sort data to align Canada data with provinces within each year
    df_final = df_final.sort_values(by=["Year", "Province"]).reset_index(drop=True)

    return df_final

# Example of how this function might be called from another script (e.g., main.py)
# if __name__ == "__main__":
#     # These filenames are relative to the 'data/raw/' directory
#     input_file = "Population Timeseries.csv"
#     canada_file = "canada_population_data.csv"
#     # Output path relative to the project root
#     output_file = "data/processed/Population_Demographics_Corrected.csv"

#     try:
#         print(f"Processing {input_file} and {canada_file}...")
#         corrected_data = correct_population_dataset(input_file, canada_file)
#         # Ensure the output directory exists (optional, depends on setup)
#         # import os
#         # os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         corrected_data.to_csv(output_file, index=False)
#         print(f"Dataset corrected and saved successfully to {output_file}")
#         print("\nSample of corrected data (first 20 rows):")
#         print(corrected_data.head(20))
#         print("\nChecking Canada data in corrected dataset:")
#         print(corrected_data[corrected_data['Province'] == 'Canada'].head())

#     except FileNotFoundError as e:
#         print(f"Error: Input file not found - {e}")
#     except ValueError as e:
#         print(f"Data Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

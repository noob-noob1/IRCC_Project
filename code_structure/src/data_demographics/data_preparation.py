import pandas as pd
import logging
import colored_traceback

# Add colored traceback for better error visualization
colored_traceback.add_hook()

def prepare_demographics_data(raw_file: str) -> pd.DataFrame:
    """
    Reads the raw demographics data, performs cleaning/transformation, and returns the DataFrame.
    """
    logging.info(f"Loading demographics data from {raw_file}")
    try:
        # --- Configuration ---
        MIN_CENSUS_YEAR = 1991
        INITIAL_COL_COUNT = 9  # Number of columns to keep initially
        MAX_YEAR = 2024

        # Define column renaming map for clarity and maintainability
        COLUMN_RENAME_MAP = {
            "Geographic name": "Province",
            "Census year": "Year",
            "Sex": "Sex",  # Keep Sex column for pivoting
            "Total (counts)": "Population Estimate",
            # Ensure exact match for original column names, including trailing spaces
            "Age groups: 0 to 14 years (counts) ": "Age 0-14",
            "Age groups: 15 to 64 years (counts) ": "Age 15-64",
            "Age groups: 65 years and over (counts) ": "Age 65+",
            "Average age": "Average Age"
        }
        # Columns to carry over after pivoting (excluding 'Sex' and 'Population Estimate')
        MERGE_COLS = ['Province', 'Year', 'Age 0-14', 'Age 15-64', 'Age 65+', 'Average Age']

        # --- Data Loading and Initial Cleaning ---
        # Load data, keeping only the first specified columns
        df = pd.read_csv(raw_file, usecols=range(INITIAL_COL_COUNT))

        # Filter data to include only years from MIN_CENSUS_YEAR onwards
        df = df[df['Census year'] >= MIN_CENSUS_YEAR].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Rename columns using the defined map
        df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

        # --- Data Transformation ---
        # Pivot table: Transform 'Sex' values into separate columns for population estimates
        df_pivot = df.pivot_table(
            index=['Province', 'Year'],
            columns='Sex',
            values='Population Estimate',
            aggfunc='sum'  # Use sum to handle potential duplicates
        ).reset_index()
        df_pivot.columns.name = None  # Remove the index name 'Sex' created by pivot

        # Prepare age group and average age data for merging
        age_avg_data = df[MERGE_COLS].drop_duplicates(subset=['Province', 'Year'], keep='first')

        # Merge the pivoted population data with the age group and average age data
        df_merged = pd.merge(
            df_pivot,
            age_avg_data,
            on=['Province', 'Year'],
            how='left'  # Keep all pivoted rows, add age/avg data
        )

        # --- Interpolation per Province ---
        interpolated_province_dfs = []
        # Process each province independently to interpolate missing years
        for province in df_merged['Province'].unique():
            province_df = df_merged[df_merged['Province'] == province].copy()

            # Convert 'Year' to numeric and set as index for time-series operations
            province_df['Year'] = pd.to_numeric(province_df['Year'], errors='coerce')
            province_df = province_df.dropna(subset=['Year'])  # Drop rows where year conversion failed
            if province_df.empty:
                continue  # Skip province if no valid year data
            province_df['Year'] = province_df['Year'].astype(int)
            province_df = province_df.set_index('Year')

            # Define the full range of years for this province
            min_year = province_df.index.min()
            complete_year_index = range(min_year, MAX_YEAR + 1)

            # Reindex to include all years in the range, creating NaNs for missing years
            province_df = province_df.reindex(complete_year_index)

            # Fill the 'Province' column for the newly added rows
            province_df['Province'] = province_df['Province'].fillna(province)

            # Convert columns to best possible dtypes before interpolation
            province_df = province_df.convert_dtypes()
            
            # Interpolate missing numeric values linearly only for numeric columns
            numeric_cols = province_df.select_dtypes(include='number').columns
            province_df[numeric_cols] = province_df[numeric_cols].interpolate(method='linear', limit_direction='both')
            
            # Reset index to bring 'Year' back as a column
            province_df = province_df.reset_index().rename(columns={'index': 'Year'})
            interpolated_province_dfs.append(province_df)

        # --- Final Combination ---
        if interpolated_province_dfs:  # Proceed only if data was processed
            combined_df = pd.concat(interpolated_province_dfs, ignore_index=True)
            # Sort the final DataFrame by Year, then by Province
            combined_df = combined_df.sort_values(by=['Year', 'Province'])
            logging.info("Demographics data preparation successful.")
            return combined_df
        else:
            logging.warning("No data available after processing and interpolation.")
            return pd.DataFrame()  # Return empty DataFrame if no data
    except Exception as e:
        logging.error(f"Error preparing demographics data: {e}")
        raise

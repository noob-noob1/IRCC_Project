import pandas as pd

def load_data(file_path):
    """Loads the raw housing started data from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Removes unnecessary columns and filters for 'Total units'."""
    columns_to_remove = [
        "DGUID", "Seasonal adjustment", "UOM", "UOM_ID",
        "SCALAR_FACTOR", "SCALAR_ID", "VECTOR", "COORDINATE", "STATUS",
        "SYMBOL", "TERMINATED", "DECIMALS"
    ]
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    df_filtered = df[df["Type of unit"] == "Total units"].drop(columns=["Type of unit"])
    df_filtered.insert(0, 'Index', range(1, len(df_filtered) + 1))
    return df_filtered

def pivot_data(df):
    """Pivots the data based on 'Housing estimates'."""
    df_pivot = df.pivot(index=['REF_DATE', 'GEO', 'Index'], columns="Housing estimates", values='VALUE').reset_index()
    df_pivot.reset_index(drop=True, inplace=True)
    return df_pivot

def group_and_aggregate(df):
    """Groups data by REF_DATE and GEO and aggregates housing metrics."""
    grouped_df = df.groupby(['REF_DATE', 'GEO'], as_index=False).agg({
        'Index': 'first',
        'Housing completions': 'first',
        'Housing starts': 'first',
        'Housing under construction': 'first'
    })
    grouped_df["REF_DATE"] = pd.to_datetime(grouped_df["REF_DATE"])
    return grouped_df

def fill_missing_dates(df):
    """Fills missing monthly dates for each GEO."""
    full_dates = pd.date_range(start=df['REF_DATE'].min(), end=df['REF_DATE'].max(), freq='MS')
    all_geos = df['GEO'].unique()
    full_df = pd.MultiIndex.from_product([full_dates, all_geos], names=['REF_DATE', 'GEO']).to_frame(index=False)
    merged_df = full_df.merge(df, on=['REF_DATE', 'GEO'], how='left')
    return merged_df

def filter_geos(df):
    """Filters the DataFrame to keep only specified GEO values."""
    geo_list = [
        "Canada", "Quebec", "Ontario", "British Columbia", "Alberta",
        "Manitoba", "New Brunswick", "Newfoundland and Labrador",
        "Nova Scotia", "Saskatchewan", "Prince Edward Island"
    ]
    filtered_df = df[df['GEO'].isin(geo_list)].reset_index(drop=True)
    filtered_df.fillna(0, inplace=True)
    return filtered_df

def distribute_quarterly_values(group):
    """Distributes quarterly values evenly across the months within the quarter."""
    group = group.set_index("REF_DATE").asfreq("MS")  # Ensure monthly frequency

    cols_to_fill = ["Housing completions", "Housing starts", "Housing under construction"]

    for col in cols_to_fill:
        # Iterate through dates where data exists (quarter starts)
        for date in group[col].dropna().index:
            # Check if it's a known quarter start month (Jan, Apr, Jul, Oct)
            # This logic assumes the original data was quarterly starting in these months.
            if date.month in [1, 4, 7, 10]:
                value = group.loc[date, col]  # Get the original quarterly value

                # Calculate equal split for the three months of the quarter
                split_value = value / 3

                # Assign split values to the current month and the next two
                group.loc[date, col] = split_value
                next_month = date + pd.DateOffset(months=1)
                month_after_next = date + pd.DateOffset(months=2)

                if next_month in group.index:
                    group.loc[next_month, col] = split_value
                if month_after_next in group.index:
                    group.loc[month_after_next, col] = split_value

    return group.reset_index()


def apply_distribution(df):
    """Applies the value distribution logic grouped by GEO."""
    # Ensure REF_DATE is datetime
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    # Sort values to ensure correct processing order within groups
    df = df.sort_values(by=['GEO', 'REF_DATE'])
    grouped_df_filled = df.groupby("GEO", group_keys=False).apply(distribute_quarterly_values)
    return grouped_df_filled


def finalize_data(df):
    """Filters data from 1976 onwards and removes the temporary Index column."""
    df_final = df[df['REF_DATE'] >= '1976-01-01'].reset_index(drop=True)
    if 'Index' in df_final.columns:
        df_final.drop(columns=['Index'], inplace=True)
    return df_final

def save_data(df, output_path):
    """Saves the processed DataFrame to a CSV file."""
    df.to_csv(output_path, index=False)
    print(f"Processed housing started data saved to {output_path}")

def process_housing_started_data(input_path, output_path):
    """Main function to process housing started data."""
    df = load_data(input_path)
    df_cleaned = clean_data(df)
    df_pivoted = pivot_data(df_cleaned)
    df_grouped = group_and_aggregate(df_pivoted)
    df_filled_dates = fill_missing_dates(df_grouped)
    df_filtered_geos = filter_geos(df_filled_dates)
    # The distribution logic needs careful application after filling and filtering
    df_distributed = apply_distribution(df_filtered_geos)
    df_final = finalize_data(df_distributed)
    save_data(df_final, output_path)

if __name__ == '__main__':
    # Example usage:
    raw_path = '../../data/raw/HousingStarted_Raw.csv'  # Adjust relative path if needed
    processed_path = '../../data/processed/HousingStarted_Processed.csv' # Adjust relative path
    process_housing_started_data(raw_path, processed_path)

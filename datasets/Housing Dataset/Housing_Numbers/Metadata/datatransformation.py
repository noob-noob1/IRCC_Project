import pandas as pd
import os
from itertools import product

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
    
    # Reset index
    df = df.reset_index()
    
    return df

# File paths
file_paths = [
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-Alberta.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-BC.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-Manitoba.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-NewBrunswick.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-NewFoundland.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-NoviaScotia.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-Ontario.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-PEI.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-quebec.csv",
    r"datasets/Housing Dataset/Housing_Numbers/Metadata/number-of-households-canada-provinces-Saskachewan.csv"
]

# Process each file
processed_data = {}
for file in file_paths:
    province_name = os.path.basename(file).split("-")[-1].replace(".csv", "")
    processed_data[province_name] = process_file(file).rename(columns={"Number_of_Households": province_name})

# Merge all datasets
merged_df = list(processed_data.values())[0]
for province, df in processed_data.items():
    if province != list(processed_data.keys())[0]:
        merged_df = merged_df.merge(df, on="Year", how="outer")

# Adding Canada data
merged_df["Canada"] = merged_df.iloc[:, 1:].sum(axis=1)

# Load the uploaded dataset to align the date range
uploaded_data = pd.read_csv(r"datasets/Housing Dataset/Merge_of_all_Features/Number_of_Household.csv", parse_dates=["REF_DATE"])
min_date = uploaded_data["REF_DATE"].min()
max_date = uploaded_data["REF_DATE"].max()
common_date_range = pd.date_range(start=min_date, end=max_date, freq='MS')  # Start of the month

# Creating a new DataFrame with 'REF_DATE' and 'GEO' columns
time_series_data = pd.DataFrame(list(product(common_date_range, merged_df.columns[1:])), columns=["REF_DATE", "GEO"])

# Mapping values from merged dataset to new format
time_series_data["Number_of_Households"] = time_series_data.apply(
    lambda row: merged_df.loc[merged_df["Year"] == row["REF_DATE"].year, row["GEO"]].values[0], axis=1
)

# Convert REF_DATE to YYYY-MM-01 format
time_series_data["REF_DATE"] = time_series_data["REF_DATE"].dt.strftime("%Y-%m-01")

# Standardizing province names
province_mapping = {
    "Alberta": "Alberta",
    "BC": "British Columbia",
    "Manitoba": "Manitoba",
    "NewBrunswick": "New Brunswick",
    "NewFoundland": "Newfoundland and Labrador",
    "NovaScotia": "Nova Scotia",
    "Ontario": "Ontario",
    "PEI": "Prince Edward Island",
    "quebec": "Quebec",
    "Saskatchewan": "Saskatchewan",
    "Canada": "Canada"
}

time_series_data["GEO"] = time_series_data["GEO"].map(province_mapping)

# Save the final dataset
output_path = r"datasets/Housing Dataset/Merge_of_all_Features/Number_of_Household.csv"
time_series_data.to_csv(output_path, index=False)

print(f"Processing complete. Standardized dataset saved as '{output_path}'.")

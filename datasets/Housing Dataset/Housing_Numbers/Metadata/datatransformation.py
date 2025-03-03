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
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-Alberta.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-BC.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-Manitoba.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-NewBrunswick.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-NewFoundland.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-NoviaScotia.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-Ontario.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-PEI.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-quebec.csv",
    r"datasets\Housing Dataset\Housing_Numbers\number-of-households-canada-provinces-Saskachewan.csv"
]
# Process each file
processed_data = {}
for file in file_paths:
    province_name = os.path.basename(file).split("-")[-1].replace(".csv", "")
    processed_data[province_name] = process_file(file).rename(columns={"Number_of_Households": province_name})

# Merge all datasets
merged_df = pd.DataFrame()
for province, df in processed_data.items():
    if merged_df.empty:
        merged_df = df
    else:
        merged_df = merged_df.merge(df[['Year', province]], on="Year", how="outer")

# Generate a date range with months for each year
date_range = pd.date_range(start=f"{merged_df['Year'].min()}-01", end=f"{merged_df['Year'].max()}-12", freq='M')

# Creating a new DataFrame with 'REF_DATE' and 'Province' columns
time_series_data = pd.DataFrame(list(product(date_range, merged_df.columns[1:])), columns=["REF_DATE", "Province"])

# Mapping values from merged dataset to new format
time_series_data["Number_of_Households"] = time_series_data.apply(
    lambda row: merged_df.loc[merged_df["Year"] == row["REF_DATE"].year, row["Province"]].values[0], axis=1
)

# Convert REF_DATE to YYYY-MM-DD format
time_series_data["REF_DATE"] = time_series_data["REF_DATE"].dt.strftime("%Y-%m-%d")

# Standardizing province names
province_mapping = {
    "Alberta": "Alberta",
    "BC": "British Columbia",
    "Manitoba": "Manitoba",
    "NewBrunswick": "New Brunswick",
    "NewFoundland": "Newfoundland and Labrador",
    "NoviaScotia": "Nova Scotia",
    "Ontario": "Ontario",
    "PEI": "Prince Edward Island",
    "quebec": "Quebec",
    "Saskachewan": "Saskatchewan"
}

time_series_data["Province"] = time_series_data["Province"].map(province_mapping)

# Reordering columns
time_series_data = time_series_data[["REF_DATE", "Province", "Number_of_Households"]]

# Save the final dataset
time_series_data.to_csv("datasets\Housing Dataset\Housing_Numbers\Number_of_household.csv", index=False)

print("Processing complete. Standardized dataset saved as 'standardized_time_series_household_data.csv'.")
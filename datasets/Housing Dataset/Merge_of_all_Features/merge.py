import pandas as pd

# Load the datasets
file_households = r"datasets\Housing Dataset\Merge_of_all_Features\Number_of_Household.csv"
file_housing_started = r"datasets\Housing Dataset\Merge_of_all_Features\Housing_Started_Manipulated.csv"
file_nhpi = r"datasets\Housing Dataset\Merge_of_all_Features\NHPI_Manipulated.csv"

df_households = pd.read_csv(file_households)
df_housing_started = pd.read_csv(file_housing_started)
df_nhpi = pd.read_csv(file_nhpi)

# Convert date column to YYYY-MM-DD format
df_households['REF_DATE'] = pd.to_datetime(df_households['REF_DATE']).dt.strftime('%Y-%m-%d')
df_housing_started['REF_DATE'] = pd.to_datetime(df_housing_started['REF_DATE']).dt.strftime('%Y-%m-%d')
df_nhpi['REF_DATE'] = pd.to_datetime(df_nhpi['REF_DATE']).dt.strftime('%Y-%m-%d')

# Merge datasets on REF_DATE and GEO
df_merged = df_households.merge(df_housing_started, on=['REF_DATE', 'GEO'], how='outer')
df_merged = df_merged.merge(df_nhpi, on=['REF_DATE', 'GEO'], how='outer')

# Sort by date
df_merged = df_merged.sort_values(by=['REF_DATE', 'GEO'])

# Save the merged dataset
df_merged.to_csv("Merged_Time_Series_Data.csv", index=False)

print("Merged dataset saved successfully!")
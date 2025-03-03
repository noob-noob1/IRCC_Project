import pandas as pd

# Reading the interpolated CSV files
national_df = pd.read_csv('number-of-households-canada-csv_interpolated.csv')
bc_df = pd.read_csv('number-of-households-canada-provinces-BC_interpolated.csv')
quebec_df = pd.read_csv('number-of-households-canada-provinces-quebec_interpolated.csv')
ontario_df = pd.read_csv('number-of-households-canada-provinces-Ontario_interpolated.csv')

# Renaming the data columns to reflect their content
national_df.rename(columns={'M_Number_of_Households': 'NO OF HOUSEHOLDS'}, inplace=True)
bc_df.rename(columns={'M_Number_of_Households': 'NO OF HOUSEHOLDS_BC'}, inplace=True)
quebec_df.rename(columns={'M_Number_of_Households': 'NO OF HOUSEHOLDS_QUEBEC'}, inplace=True)
ontario_df.rename(columns={'M_Number_of_Households': 'NO OF HOUSEHOLDS_ONTARIO'}, inplace=True)

# Merging all DataFrames on the 'Year' column
final_df = national_df.merge(bc_df, on='Year', how='left')\
                      .merge(quebec_df, on='Year', how='left')\
                      .merge(ontario_df, on='Year', how='left')

# Arranging columns in the desired order
final_df = final_df[['Year', 'NO OF HOUSEHOLDS', 'NO OF HOUSEHOLDS_BC', 'NO OF HOUSEHOLDS_QUEBEC', 'NO OF HOUSEHOLDS_ONTARIO']]

# Saving the final DataFrame to a new CSV file
final_df.to_csv('number-of-households-canada-all_interpolated.csv', index=False)

import os
import glob
import pandas as pd
from functools import reduce

def merge_cleaned_files():
    # Define input and output directories
    input_dir = "cleaned_files"
    output_dir = "Merged"
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all CSV files in the cleaned_files folder
    file_pattern = os.path.join(input_dir, "*.csv")
    file_list = glob.glob(file_pattern)
    
    # Read each file into a DataFrame, ensuring "Year" is available as a column.
    df_list = []
    for file in file_list:
        df = pd.read_csv(file, encoding="latin1")
        # If "Year" is not already a column (it might be saved as an index), reset the index
        if "Year" not in df.columns:
            df.reset_index(inplace=True)
        # Make sure the Year column is numeric
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        df_list.append(df)
    
    # Merge all DataFrames on "Year" using an outer join with specified suffixes
    merged_df = df_list[0]
    for i, df in enumerate(df_list[1:], start=1):
        merged_df = pd.merge(
            merged_df,
            df,
            on="Year",
            how="outer",
            suffixes=('', f'_dup{i}')
        )
        # Drop duplicate columns created by the merge
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
    
    # Sort the merged DataFrame by Year
    merged_df.sort_values("Year", inplace=True)
    
    # Save the merged DataFrame to the Merged folder
    output_path = os.path.join(output_dir, "merged_data.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to: {output_path}")

if __name__ == '__main__':
    merge_cleaned_files()

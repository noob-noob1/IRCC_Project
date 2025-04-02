import pandas as pd
import os
import re

def append_cleaned_files():
    input_folder = "cleaned_files"
    output_folder = "Merged"
    os.makedirs(output_folder, exist_ok=True)

    # Get all cleaned CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    df_list = []
    for file in csv_files:
        # Read the file
        df = pd.read_csv(os.path.join(input_folder, file))

        # Extract province name by removing "en(" and ")" if present
        match = re.search(r"en\((.*?)\)", file)
        province_name = match.group(1) if match else file.split("-")[-1].replace(".csv", "").strip()

        # Add a column for province name
        df["Geo"] = province_name

        df_list.append(df)

    # Append all DataFrames
    appended_df = pd.concat(df_list, ignore_index=True)

    # Ensure the Year column is numeric for sorting
    appended_df["Year"] = pd.to_numeric(appended_df["Year"], errors="coerce")

    # Sort data by Year, so each year groups all provinces together
    appended_df = appended_df.sort_values(by=["Year", "Geo"])

    # Save appended DataFrame
    merged_file_path = os.path.join(output_folder, "appended_data.csv")
    appended_df.to_csv(merged_file_path, index=False)

    print(f"Appended file saved at: {merged_file_path}")

# Run the function
append_cleaned_files()


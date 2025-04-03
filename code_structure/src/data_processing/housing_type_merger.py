import pandas as pd
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_housing_type_data(input_folder="data/raw/housing-types", output_folder="data/processed/housing"):
    """
    Merges cleaned housing type CSV files from the input folder, adds a 'Geo' column
    based on the filename, sorts the data, and saves the merged file to the output folder.

    Args:
        input_folder (str): Path to the folder containing raw housing type CSV files.
        output_folder (str): Path to the folder where the processed merged CSV will be saved.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Ensured output directory exists: {output_folder}")

        # Get all CSV files from the raw data folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith(".csv") and os.path.isfile(os.path.join(input_folder, f))]
        logging.info(f"Found {len(csv_files)} CSV files in {input_folder}")

        if not csv_files:
            logging.warning(f"No CSV files found in {input_folder}. Skipping merge process.")
            return None

        df_list = []
        for file in csv_files:
            file_path = os.path.join(input_folder, file)
            logging.info(f"Processing file: {file_path}")
            try:
                # Read the file - Assuming the structure is similar to the cleaned files in the old script
                # The old script read from 'cleaned_files', implying some pre-processing.
                # We'll assume the raw files here need similar cleaning or are already clean enough.
                # If cleaning is needed, that logic should ideally be in a separate module.
                # For now, we proceed with merging.
                df = pd.read_csv(file_path)

                # Extract province name using regex, similar to the original script
                match = re.search(r"en\((.*?)\)", file)
                if match:
                    province_name = match.group(1)
                else:
                    # Fallback: attempt to extract from filename if pattern doesn't match
                    province_name = file.split('(')[-1].split(')')[0] if '(' in file and ')' in file else file.replace('.csv', '').split('-')[-1]
                    logging.warning(f"Could not extract province using 'en()' pattern for {file}. Using fallback: {province_name}")

                df["Geo"] = province_name.strip()
                logging.debug(f"Added 'Geo' column with value: {province_name.strip()} for file {file}")

                df_list.append(df)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
                continue # Skip this file and continue with others

        if not df_list:
            logging.error("No dataframes were successfully processed. Cannot merge.")
            return None

        # Append all DataFrames
        appended_df = pd.concat(df_list, ignore_index=True)
        logging.info("Successfully concatenated all processed dataframes.")

        # Ensure the Year column is numeric for sorting (handle potential non-numeric values)
        appended_df["Year"] = pd.to_numeric(appended_df["Year"], errors="coerce")
        # Drop rows where 'Year' could not be converted to numeric, as they are invalid for sorting
        original_rows = len(appended_df)
        appended_df.dropna(subset=["Year"], inplace=True)
        if len(appended_df) < original_rows:
            logging.warning(f"Dropped {original_rows - len(appended_df)} rows due to non-numeric 'Year' values.")

        # Convert 'Year' to integer after handling NaNs
        appended_df["Year"] = appended_df["Year"].astype(int)

        # Sort data by Year and then by Geo
        appended_df = appended_df.sort_values(by=["Year", "Geo"])
        logging.info("Sorted merged data by Year and Geo.")

        # Define the output file path
        merged_file_name = "Housing_Types_Merged.csv" # More descriptive name
        merged_file_path = os.path.join(output_folder, merged_file_name)

        # Save appended DataFrame
        appended_df.to_csv(merged_file_path, index=False)
        logging.info(f"Merged housing type data saved successfully to: {merged_file_path}")
        return merged_file_path

    except FileNotFoundError:
        logging.error(f"Input directory not found: {input_folder}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during the merge process: {e}")
        return None

if __name__ == "__main__":
    # Example of how to run the function directly
    logging.info("Running housing type merger script directly.")
    merge_housing_type_data()

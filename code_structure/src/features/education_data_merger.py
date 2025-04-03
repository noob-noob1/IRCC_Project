import pandas as pd
import os
import logging
from functools import reduce

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_education_data(processed_dir):
    """Loads the five processed education datasets."""
    datasets = {
        "educators": "Educators_Processed.csv",
        "epi": "EPI_Processed.csv",
        "expenditures": "Expenditures_Processed.csv",
        "graduation": "GraduationRate_Processed.csv",
        "participation": "ParticipationRate_Processed.csv"
    }
    dataframes = {}
    all_files_found = True

    logging.info(f"Loading processed education datasets from {processed_dir}...")
    for name, filename in datasets.items():
        file_path = os.path.join(processed_dir, filename)
        try:
            df = pd.read_csv(file_path)
            # Convert REF_DATE (Year) to datetime with day-month-year format (defaulting day and month to 01)
            df['REF_DATE'] = pd.to_datetime(df['REF_DATE'].astype(str) + '-01-01', errors='coerce')
            # Ensure GEO is string
            df['GEO'] = df['GEO'].astype(str)
            # Drop rows where REF_DATE conversion failed
            df.dropna(subset=['REF_DATE'], inplace=True)
            dataframes[name] = df
            logging.info(f"Successfully loaded and preprocessed {filename}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}. Merge cannot proceed without all files.")
            all_files_found = False
            raise FileNotFoundError(f"Required input file not found: {file_path}")
        except Exception as e:
            logging.error(f"Error loading or preprocessing file {file_path}: {e}")
            raise

    if not all_files_found:
        raise FileNotFoundError("One or more required processed education files were not found.")

    return list(dataframes.values()) # Return list of dataframes for reduce

def merge_education_data(dataframes):
    """Merges the processed education datasets using an outer join."""
    if not dataframes or len(dataframes) < 2:
        logging.error("Need at least two dataframes to merge.")
        raise ValueError("Need at least two dataframes to merge.")

    logging.info("Starting outer merge of education datasets...")
    # Use reduce with an outer join
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=["REF_DATE", "GEO"], how="outer"), dataframes)
    logging.info(f"Merge completed. Shape of merged data: {merged_df.shape}")

    # Sort for readability and consistency
    merged_df.sort_values(by=["REF_DATE", "GEO"], inplace=True)

    return merged_df

def analyze_missing_values(df):
    """Analyzes and logs missing values in the merged dataframe."""
    logging.info("Analyzing missing values in the merged dataset...")
    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_summary / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_summary[missing_summary > 0],
        'Missing %': missing_percent[missing_percent > 0]
    })

    if not missing_report.empty:
        logging.warning("Missing values found:")
        logging.warning("\n" + missing_report.to_string())
    else:
        logging.info("No missing values found in the merged dataset.")

def save_merged_education_data(df, output_path):
    """Saves the final merged education DataFrame to a CSV file."""
    try:
        # Convert REF_DATE to string format day/month/year for saving (e.g., 01-01-2024)
        df['REF_DATE'] = df['REF_DATE'].dt.strftime('%d-%m-%Y')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logging.info(f"Final merged education data saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving merged education data to {output_path}: {e}")
        raise

def merge_all_education_features(processed_education_dir, output_path):
    """Main function to load, merge, analyze, and save education feature data."""
    logging.info("Starting education feature merging pipeline...")
    try:
        list_of_dfs = load_processed_education_data(processed_education_dir)
        merged_df = merge_education_data(list_of_dfs)
        analyze_missing_values(merged_df)  # Log missing values
        save_merged_education_data(merged_df, output_path)
        logging.info("Education feature merging pipeline finished successfully.")
    except FileNotFoundError as e:
        logging.error(f"Aborting merge due to missing file: {e}")
    except (ValueError, KeyError) as e:
         logging.error(f"Aborting merge due to data error: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the education merge pipeline: {e}")

if __name__ == '__main__':
    # Example usage: Define paths relative to the project root
    processed_dir = 'data/processed/education'
    output_file = os.path.join('data/processed', 'Education_Features_Merged.csv')  # Save in parent processed dir

    merge_all_education_features(processed_dir, output_file)

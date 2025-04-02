import os
import pandas as pd
from src.data_processing.population_data_corrector import correct_population_dataset
from src.features.population_metrics_calculator import calculate_population_metrics
from src.data_demographics.data_preparation import prepare_demographics_data
from src.data_processing.household_data_processor import process_household_data
from src.data_processing.housing_started_processor import process_housing_started_data
from src.data_processing.nhpi_processor import process_nhpi_data
from src.features.housing_data_merger import merge_all_housing_features
from src.data_processing.educators_processor import process_educators_data # Added import for educators
from src.data_processing.epi_processor import process_epi_data # Added import for EPI
from src.data_processing.expenditures_processor import process_expenditures_data # Added import for Expenditures
from src.data_processing.graduation_rate_processor import process_graduation_rate_data # Added import for Graduation Rate
from src.data_processing.participation_rate_processor import process_participation_rate_data # Added import for Participation Rate
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

RAW_POPULATION_TIMESERIES_FILE = os.path.join(RAW_DATA_DIR, 'Population Timeseries.csv')
CANADA_POPULATION_FILE = os.path.join(RAW_DATA_DIR, 'canada_population_data.csv')
DEMOGRAPHICS_RAW_FILE = os.path.join(RAW_DATA_DIR, 'population_by_age_and_gender.csv')
RAW_HOUSING_STARTED_FILE = os.path.join(RAW_DATA_DIR, 'HousingStarted_Raw.csv')
RAW_NHPI_FILE = os.path.join(RAW_DATA_DIR, 'NHPI.csv')
RAW_HOUSEHOLD_DIR = os.path.join(RAW_DATA_DIR, 'household_numbers')
RAW_EDUCATORS_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Educators in public elementary and secondary schools.csv') # Added path for raw educators data
RAW_EPI_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Education price index (EPI), elementary and secondary.csv') # Added path for raw EPI data
RAW_EXPENDITURES_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Elementary and secondary private schools, by type of expenditure.csv') # Added path for raw expenditures data
RAW_GRADUATION_RATE_DIR = os.path.join(RAW_DATA_DIR, 'education_datasets', 'graduation_rates') # Added path for raw graduation rate data folder
RAW_PARTICIPATION_RATE_DIR = os.path.join(RAW_DATA_DIR, 'education_datasets', 'participation_rates') # Added path for raw participation rate data folder

CORRECTED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Demographics_Corrected.csv')
FINAL_METRICS_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Metrics_by_Year_and_Province.csv')
DEMOGRAPHICS_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Demographics.csv')
PROCESSED_HOUSEHOLD_FILE = os.path.join(PROCESSED_DATA_DIR, 'Household_Numbers_Processed.csv')
PROCESSED_HOUSING_STARTED_FILE = os.path.join(PROCESSED_DATA_DIR, 'HousingStarted_Processed.csv')
PROCESSED_NHPI_FILE = os.path.join(PROCESSED_DATA_DIR, 'NHPI_Processed.csv')
FINAL_MERGED_HOUSING_FILE = os.path.join(PROCESSED_DATA_DIR, 'Housing_Features_Merged.csv')
PROCESSED_EDUCATORS_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'Educators_Processed.csv') # Added path for processed educators data
PROCESSED_EPI_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'EPI_Processed.csv') # Added path for processed EPI data
PROCESSED_EXPENDITURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'Expenditures_Processed.csv') # Added path for processed expenditures data
PROCESSED_GRADUATION_RATE_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'GraduationRate_Processed.csv') # Added path for processed graduation rate data
PROCESSED_PARTICIPATION_RATE_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'ParticipationRate_Processed.csv') # Added path for processed participation rate data
REFERENCE_HOUSEHOLD_FILE = os.path.join(
    BASE_DIR, 'old_structure', 'datasets', 'Housing Dataset', 'Merge_of_all_Features', 'Number_of_Household.csv'
)

def main():
    """Run the data processing pipeline."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Step 1: Correct Population Data
    try:
        corrected_df = correct_population_dataset(RAW_POPULATION_TIMESERIES_FILE, CANADA_POPULATION_FILE)
        corrected_df.to_csv(CORRECTED_DATA_FILE, index=False)
        logging.info("Population data corrected.")
    except FileNotFoundError as e:
        logging.error(f"Missing file: '{e.filename}' in '{RAW_DATA_DIR}'.")
        return
    except Exception as e:
        logging.error(f"Error correcting population data: {e}")
        return

    # Step 2: Calculate Population Metrics
    try:
        metrics_df = calculate_population_metrics(corrected_df.copy())
        metrics_df.to_csv(FINAL_METRICS_FILE, index=False)
        logging.info("Population metrics calculated.")
    except Exception as e:
        logging.error(f"Error calculating population metrics: {e}")
        return

    # Step 3: Prepare Demographics Data
    try:
        demographics_df = prepare_demographics_data(DEMOGRAPHICS_RAW_FILE)
        demographics_df.to_csv(DEMOGRAPHICS_PROCESSED_FILE, index=False)
        logging.info("Demographics data prepared.")
    except Exception as e:
        logging.error(f"Error preparing demographics data: {e}")
        return

    # Step 4: Process Household Data
    try:
        if not os.path.exists(REFERENCE_HOUSEHOLD_FILE):
            logging.warning(f"Reference file not found at '{REFERENCE_HOUSEHOLD_FILE}'. Skipping household data.")
        else:
            process_household_data(RAW_HOUSEHOLD_DIR, PROCESSED_HOUSEHOLD_FILE, REFERENCE_HOUSEHOLD_FILE)
            logging.info("Household data processed.")
    except FileNotFoundError as e:
        logging.error(f"Missing household file: '{e.filename}' in '{RAW_HOUSEHOLD_DIR}'.")
    except Exception as e:
        logging.error(f"Error processing household data: {e}")

    # Step 5: Process Housing Started Data
    try:
        process_housing_started_data(RAW_HOUSING_STARTED_FILE, PROCESSED_HOUSING_STARTED_FILE)
        logging.info("Housing started data processed.")
    except FileNotFoundError as e:
        logging.error(f"Missing file: '{e.filename}'.")
    except Exception as e:
        logging.error(f"Error processing housing started data: {e}")

    # Step 6: Process NHPI Data
    try:
        process_nhpi_data(RAW_NHPI_FILE, PROCESSED_NHPI_FILE)
        logging.info("NHPI data processed.")
    except FileNotFoundError as e:
        logging.error(f"Missing file: '{e.filename}'.")
    except Exception as e:
        logging.error(f"Error processing NHPI data: {e}")

    # Step 7: Merge Housing Features
    try:
        required_files = [PROCESSED_HOUSEHOLD_FILE, PROCESSED_HOUSING_STARTED_FILE, PROCESSED_NHPI_FILE]
        if all(os.path.exists(f) for f in required_files):
            merge_all_housing_features(
                household_path=PROCESSED_HOUSEHOLD_FILE,
                housing_started_path=PROCESSED_HOUSING_STARTED_FILE,
                nhpi_path=PROCESSED_NHPI_FILE,
                output_path=FINAL_MERGED_HOUSING_FILE
            )
            logging.info("Housing features merged.")
        else:
            missing = [f for f in required_files if not os.path.exists(f)]
            logging.warning(f"Merge skipped. Missing files: {', '.join(missing)}")
    except Exception as e:
        logging.error(f"Error merging housing features: {e}")

    # Step 8: Process Educators Data
    logging.info(f"Step 8: Processing educators data from '{RAW_EDUCATORS_FILE}'.")
    try:
        # Ensure the specific subdirectory for education processed data exists
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'education'), exist_ok=True)
        process_educators_data(RAW_EDUCATORS_FILE, PROCESSED_EDUCATORS_FILE)
        logging.info(f"Educators data processed and saved to '{PROCESSED_EDUCATORS_FILE}'.")
    except FileNotFoundError as e:
        logging.error(f"Error processing educators data: Input file not found. Please ensure '{e.filename}' exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during educators data processing: {e}")

    # Step 9: Process EPI Data
    logging.info(f"Step 9: Processing EPI data from '{RAW_EPI_FILE}'.")
    try:
        # Ensure the specific subdirectory for education processed data exists
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'education'), exist_ok=True)
        process_epi_data(RAW_EPI_FILE, PROCESSED_EPI_FILE)
        logging.info(f"EPI data processed and saved to '{PROCESSED_EPI_FILE}'.")
    except FileNotFoundError as e:
        logging.error(f"Error processing EPI data: Input file not found. Please ensure '{e.filename}' exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during EPI data processing: {e}")

    # Step 10: Process Expenditures Data
    logging.info(f"Step 10: Processing expenditures data from '{RAW_EXPENDITURES_FILE}'.")
    try:
        # Ensure the specific subdirectory for education processed data exists
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'education'), exist_ok=True)
        process_expenditures_data(RAW_EXPENDITURES_FILE, PROCESSED_EXPENDITURES_FILE)
        logging.info(f"Expenditures data processed and saved to '{PROCESSED_EXPENDITURES_FILE}'.")
    except FileNotFoundError as e:
        logging.error(f"Error processing expenditures data: Input file not found. Please ensure '{e.filename}' exists.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during expenditures data processing: {e}")

    # Step 11: Process Graduation Rate Data
    logging.info(f"Step 11: Processing graduation rate data from '{RAW_GRADUATION_RATE_DIR}'.")
    try:
        # Ensure the specific subdirectory for education processed data exists
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'education'), exist_ok=True)
        process_graduation_rate_data(RAW_GRADUATION_RATE_DIR, PROCESSED_GRADUATION_RATE_FILE)
        logging.info(f"Graduation rate data processed and saved to '{PROCESSED_GRADUATION_RATE_FILE}'.")
    except FileNotFoundError as e:
        logging.error(f"Error processing graduation rate data: Input folder not found or contains no CSV files. Please ensure '{RAW_GRADUATION_RATE_DIR}' exists and contains data.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during graduation rate data processing: {e}")

    # Step 12: Process Participation Rate Data
    logging.info(f"Step 12: Processing participation rate data from '{RAW_PARTICIPATION_RATE_DIR}'.")
    try:
        # Ensure the specific subdirectory for education processed data exists
        os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'education'), exist_ok=True)
        process_participation_rate_data(RAW_PARTICIPATION_RATE_DIR, PROCESSED_PARTICIPATION_RATE_FILE)
        logging.info(f"Participation rate data processed and saved to '{PROCESSED_PARTICIPATION_RATE_FILE}'.")
    except FileNotFoundError as e:
        logging.error(f"Error processing participation rate data: Input folder not found or contains no CSV files. Please ensure '{RAW_PARTICIPATION_RATE_DIR}' exists and contains data.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during participation rate data processing: {e}")


    logging.info("Pipeline completed.")
    print("Final processed files:")
    print(f"- Population Metrics: {FINAL_METRICS_FILE}")
    print(f"- Demographics: {DEMOGRAPHICS_PROCESSED_FILE}")
    if os.path.exists(PROCESSED_HOUSEHOLD_FILE):
        print(f"- Household Data: {PROCESSED_HOUSEHOLD_FILE}")
    if os.path.exists(PROCESSED_HOUSING_STARTED_FILE):
        print(f"- Housing Started: {PROCESSED_HOUSING_STARTED_FILE}")
    if os.path.exists(PROCESSED_NHPI_FILE):
        print(f"- NHPI: {PROCESSED_NHPI_FILE}")
    if os.path.exists(FINAL_MERGED_HOUSING_FILE):
        print(f"- Merged Housing Features: {FINAL_MERGED_HOUSING_FILE}")
    if os.path.exists(PROCESSED_EDUCATORS_FILE):
        print(f"- Educators Data: {PROCESSED_EDUCATORS_FILE}")
    if os.path.exists(PROCESSED_EPI_FILE):
        print(f"- EPI Data: {PROCESSED_EPI_FILE}")
    if os.path.exists(PROCESSED_EXPENDITURES_FILE):
        print(f"- Expenditures Data: {PROCESSED_EXPENDITURES_FILE}")
    if os.path.exists(PROCESSED_GRADUATION_RATE_FILE):
        print(f"- Graduation Rate Data: {PROCESSED_GRADUATION_RATE_FILE}")
    if os.path.exists(PROCESSED_PARTICIPATION_RATE_FILE):
        print(f"- Participation Rate Data: {PROCESSED_PARTICIPATION_RATE_FILE}")

if __name__ == '__main__':
    main()

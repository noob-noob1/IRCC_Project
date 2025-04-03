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
from src.features.education_data_merger import merge_all_education_features # Added import for final education merge
from src.features.education_feature_enhancer import enhance_education_data # Added import for education enhancement
from src.data_processing.housing_type_merger import merge_housing_type_data
from src.features.housing_feature_engineer import prepare_housing_data # Added import for housing feature engineering
from src.machine_learning.housing_forecaster import run_housing_forecast # Added import for housing forecasting
from src.machine_learning.education_forecaster import forecast_education_data # Added import for education forecasting
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FORECASTED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'forecasted') # Added forecast directory constant
MODELS_DIR = os.path.join(BASE_DIR, 'models') # Added models directory constant

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

# Processed Data Directories
PROCESSED_EDUCATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'education') # Define education processed dir

# Output file to be saved in data/processed
CORRECTED_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Demographics_Corrected.csv')
FINAL_METRICS_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Metrics_by_Year_and_Province.csv')
DEMOGRAPHICS_PROCESSED_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Demographics.csv')
PROCESSED_HOUSEHOLD_FILE = os.path.join(PROCESSED_DATA_DIR, 'housing', 'Household_Numbers_Processed.csv')
PROCESSED_HOUSING_STARTED_FILE = os.path.join(PROCESSED_DATA_DIR, 'housing', 'HousingStarted_Processed.csv')
PROCESSED_NHPI_FILE = os.path.join(PROCESSED_DATA_DIR, 'housing', 'NHPI_Processed.csv')
FINAL_MERGED_HOUSING_FILE = os.path.join(PROCESSED_DATA_DIR, 'Housing_Features_Merged.csv')
HOUSING_FEATURES_ENGINEERED_FILE = os.path.join(PROCESSED_DATA_DIR, 'Housing_Features_Engineered.csv') # Added path for engineered housing features
PROCESSED_EDUCATORS_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'Educators_Processed.csv') # Added path for processed educators data
PROCESSED_EPI_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'EPI_Processed.csv') # Added path for processed EPI data
PROCESSED_EXPENDITURES_FILE = os.path.join(PROCESSED_DATA_DIR, 'education', 'Expenditures_Processed.csv') # Added path for processed expenditures data
PROCESSED_GRADUATION_RATE_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'GraduationRate_Processed.csv') # Use defined dir
PROCESSED_PARTICIPATION_RATE_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'ParticipationRate_Processed.csv') # Use defined dir
FINAL_MERGED_EDUCATION_FILE = os.path.join(PROCESSED_DATA_DIR, 'Education_Features_Merged.csv') # Added path for final merged education data
FINAL_ENHANCED_EDUCATION_FILE = os.path.join(PROCESSED_DATA_DIR, 'Education_Features_Enhanced.csv') # Added path for final enhanced education data
EDUCATION_FORECAST_FILE = os.path.join(FORECASTED_DATA_DIR, 'Education_Forecast_2024_2035.csv') # Corrected path for education forecast output
HOUSING_METRICS_OUTPUT_FILE = os.path.join(MODELS_DIR, 'housing_model_evaluation_metrics.csv') # Added path for housing metrics
HOUSING_FORECAST_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, 'housing_forecasts_next_12_years.csv') # Added path for housing forecast

# Update paths for individual processed education files to use PROCESSED_EDUCATION_DIR
PROCESSED_EDUCATORS_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'Educators_Processed.csv')
PROCESSED_EPI_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'EPI_Processed.csv')
PROCESSED_EXPENDITURES_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'Expenditures_Processed.csv')

def main():
    """Run the data processing pipeline."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(FORECASTED_DATA_DIR, exist_ok=True) # Ensure forecast directory exists
    os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists

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
        process_household_data(RAW_HOUSEHOLD_DIR, PROCESSED_HOUSEHOLD_FILE)
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

    # Step 7.5: Merge Housing Type Data
    try:
        housing_types_input_folder = os.path.join(RAW_DATA_DIR, 'housing-types')
        housing_types_output_folder = os.path.join(PROCESSED_DATA_DIR, 'housing')
        merged_file_path = merge_housing_type_data(housing_types_input_folder, housing_types_output_folder)
        if merged_file_path:
            logging.info(f"Housing type data merged: {merged_file_path}")
        else:
            logging.warning("Housing type data merging was skipped or failed.")
    except Exception as e:
        logging.error(f"Error merging housing type data: {e}")

    # Step 7.6: Prepare Housing Data (Feature Engineering)
    logging.info("Step 7.6: Preparing housing data (feature engineering).")
    try:
        if os.path.exists(FINAL_MERGED_HOUSING_FILE):
            engineered_housing_df = prepare_housing_data(FINAL_MERGED_HOUSING_FILE)
            engineered_housing_df.to_csv(HOUSING_FEATURES_ENGINEERED_FILE, index=False)
            logging.info(f"Engineered housing features saved to '{HOUSING_FEATURES_ENGINEERED_FILE}'.")
        else:
            logging.warning(f"Skipping housing feature engineering because the merged file is missing: {FINAL_MERGED_HOUSING_FILE}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the housing feature engineering step: {e}")

    # Step 7.7: Run Housing Forecasting
    logging.info("Step 7.7: Running housing forecasting.")
    try:
        if os.path.exists(HOUSING_FEATURES_ENGINEERED_FILE):
            run_housing_forecast(
                input_csv=HOUSING_FEATURES_ENGINEERED_FILE,
                metrics_output_csv=HOUSING_METRICS_OUTPUT_FILE,
                forecast_output_csv=HOUSING_FORECAST_OUTPUT_FILE
            )
            logging.info(f"Housing forecast generated. Metrics: '{HOUSING_METRICS_OUTPUT_FILE}', Forecast: '{HOUSING_FORECAST_OUTPUT_FILE}'.")
        else:
            logging.warning(f"Skipping housing forecasting because the engineered housing features file is missing: {HOUSING_FEATURES_ENGINEERED_FILE}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the housing forecasting step: {e}")


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

    # Step 13: Merge Education Features
    logging.info("Step 13: Merging all processed education features.")
    try:
        # Check if all required input files exist before attempting merge
        required_edu_files = [
            PROCESSED_EDUCATORS_FILE, PROCESSED_EPI_FILE, PROCESSED_EXPENDITURES_FILE,
            PROCESSED_GRADUATION_RATE_FILE, PROCESSED_PARTICIPATION_RATE_FILE
        ]
        if all(os.path.exists(f) for f in required_edu_files):
            merge_all_education_features(
                processed_education_dir=PROCESSED_EDUCATION_DIR,
                output_path=FINAL_MERGED_EDUCATION_FILE
            )
            logging.info(f"Final merged education data saved to '{FINAL_MERGED_EDUCATION_FILE}'.")
        else:
            missing_edu = [f for f in required_edu_files if not os.path.exists(f)]
            logging.warning(f"Skipping final education merge step because one or more input files are missing: {', '.join(missing_edu)}")

    except Exception as e:
        logging.error(f"An unexpected error occurred during the final education merge step: {e}")

    # Step 14: Enhance Merged Education Data (Interpolation & Feature Engineering)
    logging.info("Step 14: Enhancing merged education data.")
    try:
        if os.path.exists(FINAL_MERGED_EDUCATION_FILE):
            enhance_education_data(
                input_path=FINAL_MERGED_EDUCATION_FILE,
                output_path=FINAL_ENHANCED_EDUCATION_FILE
            )
            logging.info(f"Final enhanced education data saved to '{FINAL_ENHANCED_EDUCATION_FILE}'.")
        else:
            logging.warning(f"Skipping education data enhancement because the merged file is missing: {FINAL_MERGED_EDUCATION_FILE}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the education data enhancement step: {e}")

    # Step 15: Run Education Forecasting
    logging.info("Step 15: Running education forecasting.")
    try:
        if os.path.exists(FINAL_ENHANCED_EDUCATION_FILE):
            forecast_education_data(
                input_path=FINAL_ENHANCED_EDUCATION_FILE,
                output_path=EDUCATION_FORECAST_FILE # Pass the full output file path
            )
            # The function inside education_forecaster handles logging success/failure and path
        else:
            logging.warning(f"Skipping education forecasting because the enhanced education file is missing: {FINAL_ENHANCED_EDUCATION_FILE}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during the education forecasting step: {e}")


    logging.info("Pipeline completed.")
    print("Final processed, model, and forecast files:")
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
    if os.path.exists(HOUSING_FEATURES_ENGINEERED_FILE):
        print(f"- Engineered Housing Features: {HOUSING_FEATURES_ENGINEERED_FILE}")
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
    if os.path.exists(FINAL_MERGED_EDUCATION_FILE):
        print(f"- Merged Education Features: {FINAL_MERGED_EDUCATION_FILE}")
    if os.path.exists(FINAL_ENHANCED_EDUCATION_FILE):
        print(f"- Enhanced Education Features: {FINAL_ENHANCED_EDUCATION_FILE}")

    if os.path.exists(HOUSING_METRICS_OUTPUT_FILE):
        print(f"- Housing Metrics: {HOUSING_METRICS_OUTPUT_FILE}")
    if os.path.exists(HOUSING_FORECAST_OUTPUT_FILE):
        print(f"- Housing Forecast: {HOUSING_FORECAST_OUTPUT_FILE}")
    if os.path.exists(EDUCATION_FORECAST_FILE):
        print(f"- Education Forecast: {EDUCATION_FORECAST_FILE}")


if __name__ == '__main__':
    main()

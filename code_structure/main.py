# main.py
import logging
import os
import pandas as pd  # Added for type hinting and potential future use

# Data Processing Imports
from src.data_processing.educators_processor import process_educators_data
from src.data_processing.epi_processor import process_epi_data
from src.data_processing.expenditures_processor import process_expenditures_data
from src.data_processing.graduation_rate_processor import process_graduation_rate_data
from src.data_processing.household_data_processor import process_household_data
from src.data_processing.housing_started_processor import process_housing_started_data
from src.data_processing.housing_type_merger import merge_housing_type_data
from src.data_processing.nhpi_processor import process_nhpi_data
from src.data_processing.participation_rate_processor import process_participation_rate_data
from src.data_processing.population_demographics_processor import correct_population_dataset as process_population_demographics # New import

# Data Demographics Imports
from src.data_demographics.data_preparation import prepare_demographics_data

# Features Imports
from src.features.education_data_merger import merge_all_education_features
from src.features.education_feature_enhancer import enhance_education_data
from src.features.housing_data_merger import merge_all_housing_features
from src.features.housing_feature_engineer import prepare_housing_data
from src.features.population_metrics_calculator import calculate_population_metrics

# Machine Learning Imports
from src.machine_learning.education_forecaster import forecast_education_data
from src.machine_learning.housing_forecaster import run_housing_forecast
from src.machine_learning.housing_type_predictor import predict_housing_types # Added import

# --- Configuration ---

# Basic logging setup
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Directory and File Path Constants ---

# Core directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
FORECASTED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'forecasted')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Raw data file paths
RAW_POPULATION_TIMESERIES_FILE = os.path.join(RAW_DATA_DIR, 'Population Timeseries.csv')
CANADA_POPULATION_FILE = os.path.join(RAW_DATA_DIR, 'canada_population_data.csv')
RAW_PROVINCES_AGE_GENDER_FILE = os.path.join(RAW_DATA_DIR, 'Demographic_data', 'Age Gender Distribution Provinces.csv') # New constant
RAW_CANADA_AGE_GENDER_FILE = os.path.join(RAW_DATA_DIR, 'Demographic_data', 'Age Gender Distribution Canada.csv') # New constant
RAW_HOUSING_STARTED_FILE = os.path.join(RAW_DATA_DIR, 'HousingStarted_Raw.csv')
RAW_NHPI_FILE = os.path.join(RAW_DATA_DIR, 'NHPI.csv')
RAW_HOUSEHOLD_DIR = os.path.join(RAW_DATA_DIR, 'household_numbers')
RAW_HOUSING_TYPES_DIR = os.path.join(RAW_DATA_DIR, 'housing-types') # Added for clarity
RAW_EDUCATORS_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Educators in public elementary and secondary schools.csv')
RAW_EPI_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Education price index (EPI), elementary and secondary.csv')
RAW_EXPENDITURES_FILE = os.path.join(RAW_DATA_DIR, 'education_datasets', 'Elementary and secondary private schools, by type of expenditure.csv')
RAW_GRADUATION_RATE_DIR = os.path.join(RAW_DATA_DIR, 'education_datasets', 'graduation_rates')
RAW_PARTICIPATION_RATE_DIR = os.path.join(RAW_DATA_DIR, 'education_datasets', 'participation_rates')

# Processed data directories
PROCESSED_HOUSING_DIR = os.path.join(PROCESSED_DATA_DIR, 'housing') # Added for clarity
PROCESSED_EDUCATION_DIR = os.path.join(PROCESSED_DATA_DIR, 'education')

# Processed & Output file paths
FINAL_POPULATION_DEMOGRAPHICS_FILE = os.path.join(PROCESSED_DATA_DIR, 'Population_Demographics_by_Year_and_Province_and_Canada.csv') # New output file
FINAL_METRICS_FILE = os.path.join(MODELS_DIR, 'Population_Metrics_by_Year_and_Province.csv') # Keep for separate metrics calculation
PROCESSED_HOUSEHOLD_FILE = os.path.join(PROCESSED_HOUSING_DIR, 'Household_Numbers_Processed.csv')
PROCESSED_HOUSING_STARTED_FILE = os.path.join(PROCESSED_HOUSING_DIR, 'HousingStarted_Processed.csv')
PROCESSED_NHPI_FILE = os.path.join(PROCESSED_HOUSING_DIR, 'NHPI_Processed.csv')
PROCESSED_HOUSING_TYPES_FILE = os.path.join(PROCESSED_HOUSING_DIR, 'HousingTypes_Merged.csv') # Added default name
FINAL_MERGED_HOUSING_FILE = os.path.join(PROCESSED_DATA_DIR, 'Housing_Features_Merged.csv')
HOUSING_FEATURES_ENGINEERED_FILE = os.path.join(PROCESSED_DATA_DIR, 'Housing_Features_Engineered.csv')
PROCESSED_EDUCATORS_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'Educators_Processed.csv')
PROCESSED_EPI_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'EPI_Processed.csv')
PROCESSED_EXPENDITURES_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'Expenditures_Processed.csv')
PROCESSED_GRADUATION_RATE_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'GraduationRate_Processed.csv')
PROCESSED_PARTICIPATION_RATE_FILE = os.path.join(PROCESSED_EDUCATION_DIR, 'ParticipationRate_Processed.csv')
FINAL_MERGED_EDUCATION_FILE = os.path.join(PROCESSED_DATA_DIR, 'Education_Features_Merged.csv')
FINAL_ENHANCED_EDUCATION_FILE = os.path.join(PROCESSED_DATA_DIR, 'Education_Features_Enhanced.csv')
EDUCATION_FORECAST_FILE = os.path.join(FORECASTED_DATA_DIR, 'Education_Forecast_2024_2035.csv')
HOUSING_METRICS_OUTPUT_FILE = os.path.join(MODELS_DIR, 'housing_model_evaluation_metrics.csv')
HOUSING_FORECAST_OUTPUT_FILE = os.path.join(FORECASTED_DATA_DIR, 'housing_forecasts_next_12_years.csv')
HOUSING_TYPE_PREDICTIONS_INPUT_FILE = os.path.join(PROCESSED_HOUSING_DIR, 'Housing_Types_Merged.csv') # Input for type prediction
HOUSING_TYPE_PREDICTIONS_OUTPUT_FILE = os.path.join(FORECASTED_DATA_DIR, 'housing_type_predictions.csv') # Output for type prediction

# --- Helper Functions ---

def setup_directories():
    """Create necessary output directories if they don't exist."""
    logging.info("Setting up output directories...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(FORECASTED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_EDUCATION_DIR, exist_ok=True)
    os.makedirs(PROCESSED_HOUSING_DIR, exist_ok=True)
    logging.info("Output directories ensured.")

# --- Pipeline Step Functions ---

def run_population_demographics_processing(
    pop_ts_file: str,
    can_pop_file: str,
    prov_age_gender_file: str,
    can_age_gender_file: str,
    output_file: str
) -> pd.DataFrame | None:
    """
    Runs the comprehensive population and demographics processing using the new module.

    Args:
        pop_ts_file: Path to the raw population timeseries CSV.
        can_pop_file: Path to the Canada population data CSV.
        prov_age_gender_file: Path to the provinces age/gender distribution CSV.
        can_age_gender_file: Path to the Canada age/gender distribution CSV.
        output_file: Path to save the final processed demographics CSV.

    Returns:
        A pandas DataFrame containing the processed data, or None if an error occurs.
    """
    logging.info("Step 1: Processing Population and Demographics Data.")
    try:
        processed_df = process_population_demographics(
            file_path=pop_ts_file,
            canada_population_file=can_pop_file,
            provinces_age_gender_file=prov_age_gender_file,
            canada_age_gender_file=can_age_gender_file,
            output_path=output_file # Pass the output path directly to the module function
        )
        if processed_df is not None:
            logging.info(f"Population and demographics data processed and saved to '{output_file}'.")
            return processed_df
        else:
            logging.error("Population and demographics processing failed (check module logs).")
            return None
    except FileNotFoundError as e:
        logging.error(f"Missing input file for demographics processing: '{e.filename}'.")
        return None
    except Exception as e:
        logging.error(f"Error during population/demographics processing step: {e}")
        return None


def run_population_metrics_calculation(input_demographics_df: pd.DataFrame, output_file: str) -> bool:
    """
    Calculates population metrics from the processed population/demographics data.

    Args:
        input_demographics_df: DataFrame with processed population and demographics data.
        output_file: Path to save the calculated metrics CSV.

    Returns:
        True if successful, False otherwise.
    """
    logging.info("Step 2: Calculating population metrics.")
    if input_demographics_df is None:
        logging.error("Skipping population metrics calculation: Input demographics DataFrame is None.")
        return False
    try:
        # Assuming calculate_population_metrics can work with the structure of the new DataFrame
        # It might need adjustments if its expected input columns changed significantly.
        metrics_df = calculate_population_metrics(input_demographics_df.copy()) # Use copy
        metrics_df.to_csv(output_file, index=False)
        logging.info(f"Population metrics calculated and saved to '{output_file}'.")
        return True
    except Exception as e:
        logging.error(f"Error calculating population metrics: {e}")
        return False

def run_household_data_processing(raw_dir: str, output_file: str) -> bool:
    """
    Processes household data files from a directory.

    Args:
        raw_dir: Directory containing raw household data files.
        output_file: Path to save the combined/processed household data CSV.

    Returns:
        True if successful, False otherwise.
    """
    logging.info("Step 4: Processing household data.")
    try:
        process_household_data(raw_dir, output_file)
        logging.info(f"Household data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Missing household data folder or files: '{e.filename}' in '{raw_dir}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing household data: {e}")
        return False

def run_housing_started_processing(raw_file: str, output_file: str) -> bool:
    """
    Processes raw housing started data.

    Args:
        raw_file: Path to the raw housing started CSV.
        output_file: Path to save the processed housing started CSV.

    Returns:
        True if successful, False otherwise.
    """
    logging.info("Step 5: Processing housing started data.")
    try:
        process_housing_started_data(raw_file, output_file)
        logging.info(f"Housing started data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Missing file: '{e.filename}'. Cannot process housing started data.")
        return False
    except Exception as e:
        logging.error(f"Error processing housing started data: {e}")
        return False

def run_nhpi_processing(raw_file: str, output_file: str) -> bool:
    """
    Processes raw New Housing Price Index (NHPI) data.

    Args:
        raw_file: Path to the raw NHPI CSV.
        output_file: Path to save the processed NHPI CSV.

    Returns:
        True if successful, False otherwise.
    """
    logging.info("Step 6: Processing NHPI data.")
    try:
        process_nhpi_data(raw_file, output_file)
        logging.info(f"NHPI data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Missing file: '{e.filename}'. Cannot process NHPI data.")
        return False
    except Exception as e:
        logging.error(f"Error processing NHPI data: {e}")
        return False

def run_housing_feature_merge(household_path: str, housing_started_path: str, nhpi_path: str, output_path: str) -> bool:
    """
    Merges processed household, housing started, and NHPI data.

    Args:
        household_path: Path to processed household data CSV.
        housing_started_path: Path to processed housing started data CSV.
        nhpi_path: Path to processed NHPI data CSV.
        output_path: Path to save the merged housing features CSV.

    Returns:
        True if successful or skipped due to missing files, False on error during merge.
    """
    logging.info("Step 7: Merging housing features.")
    required_files = [household_path, housing_started_path, nhpi_path]
    if not all(os.path.exists(f) for f in required_files):
        missing = [f for f in required_files if not os.path.exists(f)]
        logging.warning(f"Skipping housing features merge. Missing processed files: {', '.join(missing)}")
        return True # Return True as skipping isn't a fatal error for the pipeline flow here
    try:
        merge_all_housing_features(
            household_path=household_path,
            housing_started_path=housing_started_path,
            nhpi_path=nhpi_path,
            output_path=output_path
        )
        logging.info(f"Housing features merged and saved to '{output_path}'.")
        return True
    except Exception as e:
        logging.error(f"Error merging housing features: {e}")
        return False

def run_housing_type_merge(input_folder: str, output_folder: str) -> bool:
    """
    Merges housing type data files from an input folder.

    Args:
        input_folder: Folder containing raw housing type CSVs.
        output_folder: Folder to save the merged housing type CSV.

    Returns:
        True if successful or skipped, False on error during merge.
    """
    logging.info("Step 7.5: Merging housing type data.")
    try:
        # Note: merge_housing_type_data should ideally return the output path or status
        # Assuming it saves to a known location or handles logging internally for now
        merged_file_path = merge_housing_type_data(input_folder, output_folder)
        if merged_file_path and os.path.exists(merged_file_path):
            logging.info(f"Housing type data merged to: {merged_file_path}")
        elif merged_file_path is None: # Indicates handled skip/fail within function
            logging.warning("Housing type data merging was skipped or failed (check previous logs).")
        else: # Function might return path even if saving failed
            logging.warning(f"Housing type merge function ran, but output file '{merged_file_path}' may not exist.")
        return True # Treat as non-fatal for pipeline flow
    except FileNotFoundError:
        logging.error(f"Error merging housing type data: Input folder not found '{input_folder}'.")
        return False
    except Exception as e:
        logging.error(f"Error merging housing type data: {e}")
        return False

# --- New Pipeline Step for Housing Type Prediction ---
def run_housing_type_prediction(input_file: str, output_file: str) -> bool:
    """
    Runs the housing type prediction based on year filtering.

    Args:
        input_file: Path to the merged housing types CSV.
        output_file: Path to save the housing type predictions CSV.

    Returns:
        True if successful or skipped, False on error.
    """
    logging.info("Step 7.8: Running housing type prediction (filtering).")
    if not os.path.exists(input_file):
        logging.warning(f"Skipping housing type prediction. Input file missing: {input_file}")
        # Check if the file name used here matches the actual output of run_housing_type_merge
        # It might be PROCESSED_HOUSING_TYPES_FILE if that global var was updated reliably
        if input_file != PROCESSED_HOUSING_TYPES_FILE and os.path.exists(PROCESSED_HOUSING_TYPES_FILE):
             logging.warning(f"Attempting prediction with potentially correct file: {PROCESSED_HOUSING_TYPES_FILE}")
             input_file = PROCESSED_HOUSING_TYPES_FILE # Try the potentially correct path
        else:
             return True # Still skip if neither path works

    try:
        predict_housing_types(input_file, output_file)
        # Logging is handled within predict_housing_types
        return True
    except Exception as e:
        logging.error(f"Error during housing type prediction: {e}")
        return False


def run_housing_feature_engineering(merged_file: str, output_file: str) -> bool:
    """
    Performs feature engineering on the merged housing data.

    Args:
        merged_file: Path to the merged housing features CSV.
        output_file: Path to save the engineered housing features CSV.

    Returns:
        True if successful or skipped, False on error.
    """
    logging.info("Step 7.6: Preparing housing data (feature engineering).")
    if not os.path.exists(merged_file):
        logging.warning(f"Skipping housing feature engineering. Merged file missing: {merged_file}")
        return True # Non-fatal skip
    try:
        engineered_housing_df = prepare_housing_data(merged_file)
        engineered_housing_df.to_csv(output_file, index=False)
        logging.info(f"Engineered housing features saved to '{output_file}'.")
        return True
    except Exception as e:
        logging.error(f"Error during housing feature engineering: {e}")
        return False

def run_housing_forecasting(engineered_file: str, metrics_output: str, forecast_output: str) -> bool:
    """
    Runs the housing forecasting model.

    Args:
        engineered_file: Path to the engineered housing features CSV.
        metrics_output: Path to save the model evaluation metrics CSV.
        forecast_output: Path to save the forecast results CSV.

    Returns:
        True if successful or skipped, False on error.
    """
    logging.info("Step 7.7: Running housing forecasting.")
    if not os.path.exists(engineered_file):
        logging.warning(f"Skipping housing forecasting. Engineered features file missing: {engineered_file}")
        return True # Non-fatal skip
    try:
        run_housing_forecast(
            input_csv=engineered_file,
            metrics_output_csv=metrics_output,
            forecast_output_csv=forecast_output
        )
        # Specific logging is handled within run_housing_forecast
        return True
    except Exception as e:
        logging.error(f"Error during housing forecasting: {e}")
        return False

def run_educators_processing(raw_file: str, output_file: str) -> bool:
    """Processes raw educators data."""
    logging.info(f"Step 8: Processing educators data.")
    try:
        process_educators_data(raw_file, output_file)
        logging.info(f"Educators data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Error processing educators data: Input file not found '{e.filename}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing educators data: {e}")
        return False

def run_epi_processing(raw_file: str, output_file: str) -> bool:
    """Processes raw Education Price Index (EPI) data."""
    logging.info(f"Step 9: Processing EPI data.")
    try:
        process_epi_data(raw_file, output_file)
        logging.info(f"EPI data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Error processing EPI data: Input file not found '{e.filename}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing EPI data: {e}")
        return False

def run_expenditures_processing(raw_file: str, output_file: str) -> bool:
    """Processes raw education expenditures data."""
    logging.info(f"Step 10: Processing expenditures data.")
    try:
        process_expenditures_data(raw_file, output_file)
        logging.info(f"Expenditures data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError as e:
        logging.error(f"Error processing expenditures data: Input file not found '{e.filename}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing expenditures data: {e}")
        return False

def run_graduation_rate_processing(raw_dir: str, output_file: str) -> bool:
    """Processes graduation rate data files from a directory."""
    logging.info(f"Step 11: Processing graduation rate data.")
    try:
        process_graduation_rate_data(raw_dir, output_file)
        logging.info(f"Graduation rate data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError:
        logging.error(f"Error processing graduation rate data: Input folder not found or empty '{raw_dir}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing graduation rate data: {e}")
        return False

def run_participation_rate_processing(raw_dir: str, output_file: str) -> bool:
    """Processes participation rate data files from a directory."""
    logging.info(f"Step 12: Processing participation rate data.")
    try:
        process_participation_rate_data(raw_dir, output_file)
        logging.info(f"Participation rate data processed and saved to '{output_file}'.")
        return True
    except FileNotFoundError:
        logging.error(f"Error processing participation rate data: Input folder not found or empty '{raw_dir}'.")
        return False
    except Exception as e:
        logging.error(f"Error processing participation rate data: {e}")
        return False

def run_education_feature_merge(processed_edu_dir: str, output_path: str) -> bool:
    """Merges all processed education-related datasets."""
    logging.info("Step 13: Merging all processed education features.")
    required_edu_files = [
        PROCESSED_EDUCATORS_FILE, PROCESSED_EPI_FILE, PROCESSED_EXPENDITURES_FILE,
        PROCESSED_GRADUATION_RATE_FILE, PROCESSED_PARTICIPATION_RATE_FILE
    ]
    if not all(os.path.exists(f) for f in required_edu_files):
        missing_edu = [f for f in required_edu_files if not os.path.exists(f)]
        logging.warning(f"Skipping final education merge. Missing processed files: {', '.join(missing_edu)}")
        return True # Non-fatal skip
    try:
        merge_all_education_features(
            processed_education_dir=processed_edu_dir,
            output_path=output_path
        )
        logging.info(f"Final merged education data saved to '{output_path}'.")
        return True
    except Exception as e:
        logging.error(f"Error during final education merge: {e}")
        return False

def run_education_enhancement(merged_file: str, output_file: str) -> bool:
    """Enhances merged education data (interpolation, feature engineering)."""
    logging.info("Step 14: Enhancing merged education data.")
    if not os.path.exists(merged_file):
        logging.warning(f"Skipping education data enhancement. Merged file missing: {merged_file}")
        return True # Non-fatal skip
    try:
        enhance_education_data(
            input_path=merged_file,
            output_path=output_file
        )
        logging.info(f"Final enhanced education data saved to '{output_file}'.")
        return True
    except Exception as e:
        logging.error(f"Error during education data enhancement: {e}")
        return False

def run_education_forecasting(enhanced_file: str, output_file: str) -> bool:
    """Runs the education forecasting model."""
    logging.info("Step 15: Running education forecasting.")
    if not os.path.exists(enhanced_file):
        logging.warning(f"Skipping education forecasting. Enhanced file missing: {enhanced_file}")
        return True # Non-fatal skip
    try:
        forecast_education_data(
            input_path=enhanced_file,
            output_path=output_file
        )
        # Specific logging handled within forecast_education_data
        return True
    except Exception as e:
        logging.error(f"Error during education forecasting: {e}")
        return False

def print_summary():
    """Prints a summary of generated output files and their status."""
    logging.info("Pipeline execution finished. Printing summary...")
    print("\n--- Final Output File Summary ---")
    files_to_print = {
        "Processed Population & Demographics": FINAL_POPULATION_DEMOGRAPHICS_FILE, # Updated entry
        "Population Metrics": FINAL_METRICS_FILE,
        "Processed Household Data": PROCESSED_HOUSEHOLD_FILE,
        "Processed Housing Started": PROCESSED_HOUSING_STARTED_FILE,
        "Processed NHPI": PROCESSED_NHPI_FILE,
        "Merged Housing Features": FINAL_MERGED_HOUSING_FILE,
        "Engineered Housing Features": HOUSING_FEATURES_ENGINEERED_FILE,
        "Processed Educators Data": PROCESSED_EDUCATORS_FILE,
        "Processed EPI Data": PROCESSED_EPI_FILE,
        "Processed Expenditures Data": PROCESSED_EXPENDITURES_FILE,
        "Processed Graduation Rate": PROCESSED_GRADUATION_RATE_FILE,
        "Processed Participation Rate": PROCESSED_PARTICIPATION_RATE_FILE,
        "Merged Education Features": FINAL_MERGED_EDUCATION_FILE,
        "Enhanced Education Features": FINAL_ENHANCED_EDUCATION_FILE,
        "Housing Model Metrics": HOUSING_METRICS_OUTPUT_FILE,
        "Housing Forecast": HOUSING_FORECAST_OUTPUT_FILE,
        "Housing Type Predictions": HOUSING_TYPE_PREDICTIONS_OUTPUT_FILE, # Added
        "Education Forecast": EDUCATION_FORECAST_FILE,
    }
    for name, path in files_to_print.items():
        status = "Generated" if os.path.exists(path) else "Not generated (check logs)"
        print(f"- {name}: {path} ({status})")
    print("--- End Summary ---")


# --- Main Pipeline Execution ---

def main():
    """Runs the complete data processing and forecasting pipeline."""
    setup_directories()

    # --- Population & Demographics Processing ---
    # Run the new comprehensive processing step
    processed_pop_demo_df = run_population_demographics_processing(
        pop_ts_file=RAW_POPULATION_TIMESERIES_FILE,
        can_pop_file=CANADA_POPULATION_FILE,
        prov_age_gender_file=RAW_PROVINCES_AGE_GENDER_FILE,
        can_age_gender_file=RAW_CANADA_AGE_GENDER_FILE,
        output_file=FINAL_POPULATION_DEMOGRAPHICS_FILE
    )

    if processed_pop_demo_df is None:
        logging.critical("Population and demographics processing failed. Cannot proceed with dependent steps. Exiting.")
        return # Stop pipeline if this core step fails

    # Run metrics calculation using the output of the new step
    run_population_metrics_calculation(processed_pop_demo_df, FINAL_METRICS_FILE)

    # --- Housing Processing & Forecasting ---
    run_household_data_processing(RAW_HOUSEHOLD_DIR, PROCESSED_HOUSEHOLD_FILE)
    run_housing_started_processing(RAW_HOUSING_STARTED_FILE, PROCESSED_HOUSING_STARTED_FILE)
    run_nhpi_processing(RAW_NHPI_FILE, PROCESSED_NHPI_FILE)
    run_housing_feature_merge(
        PROCESSED_HOUSEHOLD_FILE, PROCESSED_HOUSING_STARTED_FILE, PROCESSED_NHPI_FILE, FINAL_MERGED_HOUSING_FILE
    )
    run_housing_type_merge(RAW_HOUSING_TYPES_DIR, PROCESSED_HOUSING_DIR) # Pass output *folder*
    run_housing_feature_engineering(FINAL_MERGED_HOUSING_FILE, HOUSING_FEATURES_ENGINEERED_FILE)
    run_housing_forecasting(
        HOUSING_FEATURES_ENGINEERED_FILE, HOUSING_METRICS_OUTPUT_FILE, HOUSING_FORECAST_OUTPUT_FILE
    )
    # Run the new housing type prediction step
    run_housing_type_prediction(HOUSING_TYPE_PREDICTIONS_INPUT_FILE, HOUSING_TYPE_PREDICTIONS_OUTPUT_FILE)

    # --- Education Processing & Forecasting ---
    run_educators_processing(RAW_EDUCATORS_FILE, PROCESSED_EDUCATORS_FILE)
    run_epi_processing(RAW_EPI_FILE, PROCESSED_EPI_FILE)
    run_expenditures_processing(RAW_EXPENDITURES_FILE, PROCESSED_EXPENDITURES_FILE)
    run_graduation_rate_processing(RAW_GRADUATION_RATE_DIR, PROCESSED_GRADUATION_RATE_FILE)
    run_participation_rate_processing(RAW_PARTICIPATION_RATE_DIR, PROCESSED_PARTICIPATION_RATE_FILE)
    run_education_feature_merge(PROCESSED_EDUCATION_DIR, FINAL_MERGED_EDUCATION_FILE)
    run_education_enhancement(FINAL_MERGED_EDUCATION_FILE, FINAL_ENHANCED_EDUCATION_FILE)
    run_education_forecasting(FINAL_ENHANCED_EDUCATION_FILE, EDUCATION_FORECAST_FILE)

    # --- Final Summary ---
    print_summary()


if __name__ == '__main__':
    main()

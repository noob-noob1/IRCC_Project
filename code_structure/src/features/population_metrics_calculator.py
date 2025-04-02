import pandas as pd
import numpy as np # Import numpy for safe division
import logging

# Configure logging: You can adjust the level and format as needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_population_metrics(df):
    """
    Calculates various population metrics based on PRs, TRs, Births, Deaths, and Population Estimate.

    Args:
        df (pandas.DataFrame): DataFrame containing corrected population data with columns:
                               'Total PRs', 'Total TRs', 'Total Births', 'Total Deaths', 'Population Estimate'.

    Returns:
        pandas.DataFrame: The input DataFrame with added columns for calculated metrics:
                          'Net Migration', 'Natural Increase', 'Net Population Change',
                          'Net Migration Rate', 'Natural Growth Rate', 'Population Growth Rate (%)'.
    """
    logging.info("Starting calculate_population_metrics function.")
    logging.info(f"Input DataFrame shape: {df.shape}")
    logging.debug(f"Input DataFrame columns: {df.columns.tolist()}")

    # Ensure input columns are numeric, coercing errors to NaN
    numeric_cols = ['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths', 'Population Estimate']
    for col in numeric_cols:
        if col in df.columns:
            logging.debug(f"Converting column '{col}' to numeric.")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            na_count = df[col].isna().sum()
            if na_count > 0:
                logging.warning(f"Column '{col}': {na_count} NaN values introduced after converting to numeric.")
        else:
            logging.warning(f"Column '{col}' not found. Filling with 0 for metric calculation.")
            df[col] = 0

    logging.debug("Input columns converted to numeric (or filled with 0 if missing).")

    # Calculate Net Migration and Natural Increase
    df['Net Migration'] = df['Total PRs'] - df['Total TRs']
    df['Natural Increase'] = df['Total Births'] - df['Total Deaths']
    logging.debug("Calculated 'Net Migration' and 'Natural Increase'.")
    logging.debug(f"Sample of 'Net Migration': {df['Net Migration'].head().tolist()}")
    logging.debug(f"Sample of 'Natural Increase': {df['Natural Increase'].head().tolist()}")


    # Calculate Net Population Change
    df['Net Population Change'] = df['Net Migration'] + df['Natural Increase']
    logging.debug("Calculated 'Net Population Change'.")
    logging.debug(f"Sample of 'Net Population Change': {df['Net Population Change'].head().tolist()}")

    # Calculate Rates (per 1000 for migration/natural growth, % for overall growth)
    logging.debug("Calculating rates.")
    df['Net Migration Rate'] = np.divide(df['Net Migration'] * 1000, df['Population Estimate'], out=np.zeros_like(df['Net Migration'] * 1000, dtype=float), where=df['Population Estimate']!=0)
    df['Natural Growth Rate'] = np.divide(df['Natural Increase'] * 1000, df['Population Estimate'], out=np.zeros_like(df['Natural Increase'] * 1000, dtype=float), where=df['Population Estimate']!=0)
    df['Population Growth Rate (%)'] = np.divide(df['Net Population Change'] * 100, df['Population Estimate'], out=np.zeros_like(df['Net Population Change'] * 100, dtype=float), where=df['Population Estimate']!=0)
    logging.debug("Calculated 'Net Migration Rate', 'Natural Growth Rate', and 'Population Growth Rate (%)'.")
    logging.debug(f"Sample of 'Net Migration Rate': {df['Net Migration Rate'].head().tolist()}")
    logging.debug(f"Sample of 'Natural Growth Rate': {df['Natural Growth Rate'].head().tolist()}")
    logging.debug(f"Sample of 'Population Growth Rate (%)': {df['Population Growth Rate (%)'].head().tolist()}")


    # Fill any resulting NaNs in rate columns with 0
    rate_cols = ['Net Migration Rate', 'Natural Growth Rate', 'Population Growth Rate (%)']
    for col in rate_cols:
        na_count_before_fill = df[col].isna().sum()
        if na_count_before_fill > 0:
            logging.warning(f"Column '{col}': Filling {na_count_before_fill} NaN values with 0.")
        df[col] = df[col].fillna(0)
        na_count_after_fill = df[col].isna().sum()
        if na_count_after_fill > 0:
            logging.error(f"Column '{col}': Still has {na_count_after_fill} NaN values after fillna(0). Check for issues.")


    logging.info("Finished calculate_population_metrics function.")
    logging.info(f"Output DataFrame shape: {df.shape}")
    logging.debug(f"Output DataFrame columns: {df.columns.tolist()}")
    logging.debug(f"First 5 rows of output DataFrame:\n{df.head()}")
    return df

if __name__ == '__main__':
    # Example Usage with some potential issues for debugging
    data = {
        'Total PRs': [100, 200, 'invalid', 400, 500],
        'Total TRs': [50, 100, 150, 200, 250],
        'Total Births': [20, 30, 40, 50, 60],
        'Total Deaths': [10, 15, 20, 25, 30],
        'Population Estimate': [1000, 2000, 3000, 0, 5000] # Include a 0 population to test division
    }
    example_df = pd.DataFrame(data)

    result_df = calculate_population_metrics(example_df)
    print(result_df)
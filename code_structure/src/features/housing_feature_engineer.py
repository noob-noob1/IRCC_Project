import pandas as pd
import numpy as np

def load_and_trim_data(file_path, start_date="1986-01-01", end_date="2024-10-01"):
    """
    Loads the dataset, converts REF_DATE to datetime, and trims by date range.

    Args:
        file_path (str): Path to the input CSV file.
        start_date (str): Start date for trimming (YYYY-MM-DD).
        end_date (str): End date for trimming (YYYY-MM-DD).

    Returns:
        pandas.DataFrame: Trimmed DataFrame.
    """
    df = pd.read_csv(file_path)
    df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])
    df_trimmed = df[(df['REF_DATE'] >= start_date) & (df['REF_DATE'] <= end_date)].copy()
    print(f"Data loaded and trimmed. Shape: {df_trimmed.shape}")
    return df_trimmed

def handle_missing_values(df):
    """
    Handles missing values using interpolation and median fill.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with missing values handled.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].interpolate(method='linear')

    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled remaining NaNs in {col} with median: {median_val}")

    print("Missing values handled.")
    print("Missing Values After Handling:")
    print(df.isnull().sum())
    return df

def engineer_features(df):
    """
    Engineers time-based, lag, and rolling mean features.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with engineered features.
    """
    # Time-based features
    df['Year'] = df['REF_DATE'].dt.year
    df['Month'] = df['REF_DATE'].dt.month
    df['Quarter'] = df['REF_DATE'].dt.quarter
    df['Day'] = df['REF_DATE'].dt.day
    print("Time-based features created.")

    # Lag features
    lag_columns = ['Number_of_Households', 'Housing completions', 'Housing starts',
                   'Housing under construction', 'House only NHPI', 'Land only NHPI',
                   'Total (house and land) NHPI']
    
    # Ensure lag columns exist before creating lags
    existing_lag_columns = [col for col in lag_columns if col in df.columns]
    if len(existing_lag_columns) < len(lag_columns):
        print(f"Warning: Some lag columns not found in DataFrame: {set(lag_columns) - set(existing_lag_columns)}")

    for col in existing_lag_columns:
        for lag in [1, 3, 6]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    print("Lag features created.")

    # Rolling mean features
    for col in existing_lag_columns:
        df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3).mean()
    print("Rolling mean features created.")

    # Drop NaNs introduced by lag/rolling features
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} rows with NaNs introduced by feature engineering.")
    print(f"Final shape after feature engineering: {df.shape}")

    return df

def prepare_housing_data(input_path):
    """
    Main function to orchestrate the data preparation steps.

    Args:
        input_path (str): Path to the raw merged housing data CSV.

    Returns:
        pandas.DataFrame: Processed DataFrame ready for modeling.
    """
    print(f"Starting housing data preparation for: {input_path}")
    df_trimmed = load_and_trim_data(input_path)
    df_handled = handle_missing_values(df_trimmed)
    df_featured = engineer_features(df_handled)
    print("Housing data preparation complete.")
    return df_featured

if __name__ == '__main__':
    # Example usage:
    input_file = "../../data/processed/Housing_Features_Merged.csv" # Relative path from src/features
    output_file = "../../data/processed/Housing_Features_Engineered.csv" # Example output path

    processed_df = prepare_housing_data(input_file)

    print("\nProcessed Data Head:")
    print(processed_df.head())

    # Optionally save the processed data
    # processed_df.to_csv(output_file, index=False)
    # print(f"\nProcessed data saved to {output_file}")

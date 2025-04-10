import pandas as pd
import os


def predict_housing_types(input_path, output_path):
    """
    Reads housing data, filters for future years (2025-2035),
    and saves the predictions to a CSV file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the output CSV file should be saved.
    """
    try:
        # Read the input CSV
        df = pd.read_csv(input_path)

        # Filter for years between 2025 and 2035 (inclusive of 2025, exclusive of 2036)
        df_filtered = df[
            (df["Year"] >= 2025) & (df["Year"] < 2036)
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the filtered data to the output CSV
        df_filtered.to_csv(output_path, index=False)

    except FileNotFoundError as e:
        # Consider using logging instead of print for production code
        print(
            f"Error: Input file not found at {input_path}"
        )  # Kept error prints for now
        raise e  # Re-raise exception after printing
    except KeyError as e:
        print(
            f"Error: Column '{e}' not found in the input file. Please ensure the CSV has a 'Year' column."
        )
        raise e  # Re-raise exception after printing
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e  # Re-raise exception after printing

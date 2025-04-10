import pandas as pd
import logging
import traceback # Keep traceback for error logging

# Configure basic logging for errors only
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def correct_population_dataset(
    file_path: str,
    canada_population_file: str,
    provinces_age_gender_file: str,
    canada_age_gender_file: str,
    output_path: str | None = None
) -> pd.DataFrame | None:
    """
    Corrects population timeseries, integrates age/gender data, and merges demographic info.
    (Logic modified to match script2 for identical output content).

    Args:
        file_path: Path to the base population timeseries CSV.
        canada_population_file: Path to the official Canada population data CSV.
        provinces_age_gender_file: Path to the provinces age/gender distribution CSV.
        canada_age_gender_file: Path to the Canada age/gender distribution CSV.
        output_path: Optional path to save the final cleaned dataset CSV.

    Returns:
        A pandas DataFrame containing the final cleaned and merged data, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        df['Province'] = df['Province'].str.strip()
        df_unique = df.drop_duplicates(subset=["Year", "Province"])
        df_unique = df_unique[df_unique["Year"] >= 1991]

        canada_population_df = pd.read_csv(canada_population_file)
        canada_population_df = canada_population_df[canada_population_df["Year"].isin(df_unique["Year"].unique())]

        common_columns = ["Year", "Province", "Total PRs", "Total TRs", "Total Births", "Total Deaths", "Population Estimate"]
        df_unique = df_unique[common_columns]
        canada_population_df = canada_population_df[common_columns]

        df_corrected = df_unique[df_unique["Province"] != "Canada"].copy()
        grouped_df = df_corrected.groupby('Year', as_index=False)[['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths']].sum()
        grouped_df['Province'] = 'Canada'

        canada_population_df = canada_population_df.set_index('Year')
        grouped_df_indexed = grouped_df.set_index('Year')

        for col in ['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths']:
             # Check if column exists in grouped_df_indexed before combining
             if col in grouped_df_indexed.columns:
                 canada_population_df[col] = grouped_df_indexed[col].combine_first(canada_population_df[col])

        canada_population_df = canada_population_df.reset_index()

        df_final = pd.concat([df_corrected, canada_population_df], ignore_index=True)
        df_final = df_final.sort_values(by=["Year", "Province"]).reset_index(drop=True)

        # --- Clean Age-Gender Data ---
        def clean_age_gender(df_ag):
            df_ag = df_ag[['REF_DATE', 'GEO', 'Gender', 'Age group', 'VALUE']].copy()
            df_ag.columns = ['Year_Str', 'Province', 'Gender', 'Age_Group', 'Population']
            df_ag['Province'] = df_ag['Province'].str.strip()
            df_ag = df_ag[df_ag['Gender'].isin(['Men+', 'Women+'])].dropna(subset=['Population'])
            df_ag['Year_DT'] = pd.to_datetime(df_ag['Year_Str'].astype(str) + '-01-01', errors='coerce')
            df_ag = df_ag.dropna(subset=['Year_DT'])
            df_ag['REF_DATE'] = df_ag['Year_DT'].dt.strftime('01-%m-%Y')
            df_ag['Year'] = df_ag['Year_DT'].dt.year
            df_ag = df_ag[df_ag['Year'] >= 1991]
            return df_ag[['REF_DATE', 'Year', 'Province', 'Gender', 'Age_Group', 'Population']]

        provinces_df = pd.read_csv(provinces_age_gender_file)
        canada_df = pd.read_csv(canada_age_gender_file)
        provinces_clean = clean_age_gender(provinces_df)
        canada_clean = clean_age_gender(canada_df)
        combined = pd.concat([provinces_clean, canada_clean], ignore_index=True)

        # --- Pivot to Male/Female columns ---
        pivoted = combined.pivot_table(
            index=['REF_DATE', 'Year', 'Province', 'Age_Group'],
            columns='Gender',
            values='Population',
            aggfunc='sum'
        ).reset_index().rename(columns={'Men+': 'Male_Population', 'Women+': 'Female_Population'})

        # Ensure columns exist before calculation, fill missing with NaN
        if 'Male_Population' not in pivoted.columns: pivoted['Male_Population'] = pd.NA
        if 'Female_Population' not in pivoted.columns: pivoted['Female_Population'] = pd.NA
        pivoted['Combined_Population'] = pivoted['Male_Population'] + pivoted['Female_Population']

        # --- Filter only valid, non-overlapping age groups ---
        age_group_map = {
            '0 to 14 years': 'Children (0–14)',
            '15 to 64 years': 'Working Age (15–64)',
            '90 years and older': 'Elderly (90+)',
            'All ages': 'All Ages'
        }
        pivoted['Age_Group_Category'] = pivoted['Age_Group'].map(age_group_map)
        pivoted = pivoted.dropna(subset=['Age_Group_Category'])

        # --- Aggregate gender and combined values ---
        aggregated = pivoted.groupby(['REF_DATE', 'Year', 'Province', 'Age_Group_Category']).agg({
            'Male_Population': 'sum',
            'Female_Population': 'sum',
            'Combined_Population': 'sum'
        }).reset_index()


        # --- Merge demographic totals for birth/death/estimate data ---
        demographic_info = df_final[['Year', 'Province', 'Total PRs', 'Total TRs', 'Total Births', 'Total Deaths', 'Population Estimate']]
        final_df = pd.merge(aggregated, demographic_info, on=['Year', 'Province'], how='left')

        # --- Fill missing values from 'All Ages' row within same year-province ---
        def fill_missing_from_all_ages(df):
            fill_cols = ['Total PRs', 'Total TRs', 'Total Births', 'Total Deaths', 'Population Estimate']
            all_ages_ref = df[df['Age_Group_Category'] == 'All Ages'][['Year', 'Province'] + fill_cols].copy()

            df_filled = pd.merge(
                df,
                all_ages_ref,
                on=['Year', 'Province'],
                suffixes=('', '_ref'),
                how='left'
            )

            for col in fill_cols:
                if f'{col}_ref' in df_filled.columns:
                    df_filled[col] = df_filled[col].fillna(df_filled[f'{col}_ref'])
                    df_filled.drop(columns=[f'{col}_ref'], inplace=True)
                else:
                     # Use warning level for potentially missing data during processing
                     logging.warning(f"Reference column {col}_ref not found during fillna step. Check 'All Ages' data.")


            return df_filled

        final_df = fill_missing_from_all_ages(final_df)

        # --- Export final version ---
        if output_path:
            try:
                final_df.to_csv(output_path, index=False)
            except Exception as e:
                logging.error(f"Error saving final dataset to {output_path}: {e}")
                return None

        return final_df

    except FileNotFoundError as e:
        logging.error(f"Missing input file: {e.filename}")
        return None
    except KeyError as e:
        logging.error(f"Missing expected column in input data: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during population demographics processing: {e}")
        logging.error(traceback.format_exc())
        return None

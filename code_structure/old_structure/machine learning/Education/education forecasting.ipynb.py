# %%
import pandas as pd
import numpy as np
from prophet import Prophet
from functools import reduce

# Load dataset
file_path = "D:\Personal Projects\IRCC_Project\machine learning\Education\Enhanced_Education_Dataset.csv"
edu_df = pd.read_csv(file_path)

# Prepare REF_DATE and GEO
edu_df['REF_DATE'] = pd.to_datetime(edu_df['REF_DATE'], format='%d-%m-%Y')
edu_df['Year'] = edu_df['REF_DATE'].dt.year
edu_df = edu_df.rename(columns={'Province': 'GEO'})

# Filter from 1991 onward
edu_df = edu_df[edu_df['Year'] >= 1991]

# Define educator columns to forecast
educator_types = {
    'Full-time educators': 'Full-time',
    'Part-time educators': 'Part-time',
    'Total, work status': 'Total'
}

# Prepare results container
forecast_by_type = {label: [] for label in educator_types.values()}

# Forecast for each educator type
for column, label in educator_types.items():
    temp_df = edu_df[['REF_DATE', 'GEO', column]].copy()
    pivot_df = temp_df.pivot(index='REF_DATE', columns='GEO', values=column)
    pivot_df.index = pd.to_datetime(pivot_df.index)
    monthly_df = pivot_df.resample('MS').interpolate(method='linear')

    for geo in monthly_df.columns:
        df_geo = monthly_df[[geo]].reset_index()
        df_geo.columns = ['ds', 'y']

        # Remove outliers using IQR
        q1, q3 = df_geo['y'].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df_geo = df_geo[(df_geo['y'] >= lower) & (df_geo['y'] <= upper)]

        if len(df_geo) < 10:
            continue

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.5
        )
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='yearly', period=365.25, fourier_order=10)
        model.fit(df_geo)

        future = pd.date_range(start='2024-01-01', end='2035-12-01', freq='MS')
        future_df = pd.DataFrame({'ds': future})
        forecast = model.predict(future_df)

        result = forecast[['ds', 'yhat']].copy()
        result.columns = ['REF_DATE', label]
        result['GEO'] = geo
        result['REF_DATE'] = result['REF_DATE'].dt.strftime('%d-%m-%Y')  # match original format

        forecast_by_type[label].append(result)

# Merge forecasts side-by-side
merged_forecasts = []
for label in ['Full-time', 'Part-time', 'Total']:
    df = pd.concat(forecast_by_type[label], ignore_index=True)
    df = df[['REF_DATE', 'GEO', label]]
    merged_forecasts.append(df)

# Combine all forecast types into one DataFrame
final_df = reduce(lambda left, right: pd.merge(left, right, on=['REF_DATE', 'GEO']), merged_forecasts)

# Ensure final columns and sort
final_df = final_df[['REF_DATE', 'GEO', 'Full-time', 'Part-time', 'Total']]
final_df = final_df.sort_values(by='REF_DATE').reset_index(drop=True)

# Export to CSV
final_df.to_csv("Educator_Typewise_Forecast_2024_2035.csv", index=False)

# Preview
print(final_df.head())
print("✅ Forecast saved to 'Educator_Typewise_Forecast_2024_2035.csv'")


# %%
edu_df.head()

# %%




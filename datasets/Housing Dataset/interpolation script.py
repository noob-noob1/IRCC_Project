# the script relies on data from 1976 to 2011
# first, it interpolates the value for each year
# then, it forecasts the values for the years after (until 2026)

import numpy as np
import pandas as pd 

df = pd.read_csv('number-of-households-canada-csv.csv')
df.set_index('year', inplace=True)

# Extend index to 2026
df = df.reindex(range(df.index.min(), 2027))

# Interpolation for known years
df.loc[:2011, 'high_growth'] = df.loc[:2011, 'high_growth'].interpolate(method='linear').round()

# Forecast using linear regression
known = df.loc[:2011, 'high_growth']
coeffs = np.polyfit(list(known.index), known.values, 1)
for year in df.index:
    if year > 2011 and pd.isna(df.at[year, 'high_growth']):
        df.at[year, 'high_growth'] = int(np.rint(np.polyval(coeffs, year)))

# Reset index, rename, and export
df.reset_index(inplace=True)
df.rename(columns={'index': 'year', 'high_growth': 'growth'}, inplace=True)
df['growth'] = df['growth'].astype(int)
df.to_csv('number-of-households-canada-csv_interpolated.csv', index=False)
print(df)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# After loading the data, the datatype of REF_DATE was changed to datetime format and the data is trimmed to range from "1986-01-01" till "2024-10-01".

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
file_path = "Final_Merged_Data.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Convert REF_DATE to datetime format
df['REF_DATE'] = pd.to_datetime(df['REF_DATE'])

# Define the date range
start_date = "1986-01-01"
end_date = "2024-10-01"

# Filter the dataset
df_trimmed = df[(df['REF_DATE'] >= start_date) & (df['REF_DATE'] <= end_date)]


# Display the first few rows of the trimmed dataset
print(df_trimmed.head())

# Save the trimmed dataset if needed
df_trimmed.to_csv("Trimmed_Time_Series_Data.csv", index=False)

# %%
df_trimmed.head()

# %% [markdown]
# Checking data structure and quality

# %%
# Check data structure and quality
print("Dataset Information:")
df_trimmed.info()
print("\nMissing Values:")
print(df_trimmed.isnull().sum())
print("\nBasic Statistics:")
print(df_trimmed.describe())
print("\nUnique Values:")
print(df_trimmed.nunique())

# %% [markdown]
# Handle missing values
# 

# %%
# Exclude non-numeric columns before performing numerical operations
numeric_cols = df_trimmed.select_dtypes(include=[np.number]).columns

# Interpolation to fill in missing values with a smooth trend, avoiding sudden jumps
df_trimmed[numeric_cols] = df_trimmed[numeric_cols].interpolate(method='linear')

# Fill remaining missing values with the median of each numeric column to prevent extreme values from skewing data
# Handle columns that still have missing values explicitly
for col in numeric_cols:
    if df_trimmed[col].isnull().sum() > 0:
        df_trimmed[col].fillna(df_trimmed[col].median(), inplace=True)


# %%
# Check data structure after handling missing values
print("Dataset Information: After Handling Missing Values")
df_trimmed.info()
print("\nMissing Values After Handling:")
print(df_trimmed.isnull().sum())

# %% [markdown]
# ## Feature Engineering
# 

# %%
# Extract year, month, quarter, and day to enable seasonal and trend analysis
df_trimmed['Year'] = df_trimmed['REF_DATE'].dt.year
df_trimmed['Month'] = df_trimmed['REF_DATE'].dt.month
df_trimmed['Quarter'] = df_trimmed['REF_DATE'].dt.quarter
df_trimmed['Day'] = df_trimmed['REF_DATE'].dt.day

# %%
# Check structure after feature engineering
print("Dataset Information: After Feature Engineering")
df_trimmed.info()
df_trimmed.head()


# %% [markdown]
# Creating Lag Features (Using 1, 3, and 6 months lag).
# These help the model recognize past patterns and predict future trends

# %%
lag_columns = ['Number_of_Households', 'Housing completions', 'Housing starts',
               'Housing under construction', 'House only NHPI', 'Land only NHPI',
               'Total (house and land) NHPI']

for col in lag_columns:
    for lag in [1, 3, 6]:  # Using past values from 1, 3, and 6 months ago to capture short- and mid-term trends
        df_trimmed[f'{col}_lag_{lag}'] = df_trimmed[col].shift(lag)

# %%
# Check structure after creating lag features
print("Dataset Information: After Lag Features")
df_trimmed.info()
df_trimmed.head()

# %% [markdown]
# Creating Rolling Mean Features (3-month moving average) which smooths out fluctuations to highlight long-term trends.

# %%
for col in lag_columns:
    df_trimmed[f'{col}_rolling_mean_3'] = df_trimmed[col].rolling(window=3).mean()

# Drop rows with NaN values introduced by shifting (since lag features create missing values at the start)
df_trimmed.dropna(inplace=True)

# %%
print("Dataset Information: After Final Cleaning")
df_trimmed.info()
df_trimmed.head()


# %% [markdown]
# Plotting time series trends for key variables after missing value handling
# 

# %%
# plt.figure(figsize=(12, 6))
# for col in lag_columns:
#     plt.plot(df_trimmed['REF_DATE'], df_trimmed[col], label=col)
# plt.xlabel("Date")
# plt.ylabel("Values")
# plt.title("Time Series Trends After Handling Missing Values")
# plt.legend()
# plt.show()

# %%

# fig, axes = plt.subplots(len(lag_columns), 1, figsize=(12, 3 * len(lag_columns)), sharex=True)
# fig.suptitle("Time Series Trends After Handling Missing Values", fontsize=16)

# for i, col in enumerate(lag_columns):
#     axes[i].plot(df_trimmed['REF_DATE'], df_trimmed[col], label=col, color='tab:blue')
#     axes[i].set_ylabel(col)
#     axes[i].legend()
#     axes[i].grid()

# plt.xlabel("Date")
# plt.show()


# %%
df_trimmed.tail()

# %%


# %% [markdown]
#  Saving the cleaned dataset to avoid reprocessing later

# %%

# Save the dataset after feature engineering
df_trimmed.to_csv("Trimmed_Time_Series_Data after Feature Engineering.csv", index=False)

# %%
# df_trimmed.to_csv("Trimmed_Time_Series_Data.csv", index=False)

# %%
df_trimmed.head()

# %%




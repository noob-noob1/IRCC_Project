import pandas as pd

def process_file(input_file, output_file):
    # Read csv:
    df = pd.read_csv(input_file, encoding='latin1', header=2)
    
    # Remove all unnecessary rows
    df = pd.concat([df.iloc[:8], df.iloc[13:18]]).reset_index(drop=True)
    
    # Keep only relevant columns
    df = df.iloc[:, [1, 2]]
    
    # convert columns to type nmuber and ensure it's integer
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    # setting the year column as index
    df.set_index('Year', inplace=True)
    
    # create rows for missing years
    df = df.reindex(range(int(df.index.min()), 2037))
    
    # Converting the data column to numeric 
    data_col = df.columns[0]
    df[data_col] = pd.to_numeric(df[data_col], errors='coerce')
    
    # interpolating the value for the missing years
    df[data_col] = df[data_col].interpolate(method='linear').round()
    
    # Filling remaining NaN values with 0 and convert to integer
    df[data_col] = df[data_col].fillna(0).astype(int)
    
    # Resetting the index to make 'Year' a column again
    df = df.reset_index()
    
    # Renaming the data column to "M_Number_of_Households"
    df = df.rename(columns={data_col: 'M_Number_of_Households'})
    
    # Saving the processed DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
# Processing to final output:
process_file('number-of-households-canada-provinces-BC.csv', 'number-of-households-canada-provinces-BC_interpolated.csv')
process_file('number-of-households-canada-provinces-Ontario.csv', 'number-of-households-canada-provinces-Ontario_interpolated.csv')
process_file('number-of-households-canada-provinces-quebec.csv', 'number-of-households-canada-provinces-quebec_interpolated.csv')

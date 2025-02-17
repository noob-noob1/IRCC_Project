import os
import pandas as pd

# Define the directory containing the CSV files
directory = "datasets\PR Data\PR Ontario datasets"  # Update this path to your folder location
output_file = "merged_text_data.txt"  # Output text file

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]

# Open a text file to save the extracted content
with open(output_file, "w", encoding="utf-8") as out_file:
    for file in csv_files:
        file_path = os.path.join(directory, file)
        
        try:
            # Load CSV into DataFrame
            df = pd.read_csv(file_path)

            # Extract file name (used as section headers in the text)
            out_file.write(f"### {file} ###\n\n")

            # Convert DataFrame to text format
            text_data = df.to_string(index=False)

            # Write extracted text to file
            out_file.write(text_data + "\n\n")
        
        except Exception as e:
            out_file.write(f"Error reading {file}: {e}\n\n")

print(f"Extraction complete. Data saved in '{output_file}'.")

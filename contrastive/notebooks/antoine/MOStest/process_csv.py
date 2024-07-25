import pandas as pd
import sys

def process_file(input_filename):
    """
    Description:
    -------------
    The `process_csv` script processes CSV files to ensure they conform to a 
    specific format required for further analysis. It checks if the required 
    column 'IID' is present, renames and formats it if necessary, and saves 
    the processed file with a new name. 

    Input:
    ------
    - `input_file_path` (str): The path to the input CSV file that needs processing. 
    This file should have a column named 'ID' or 'IID' with integer values or strings 
    representing integers.

    Output:
    -------
    - Saves a new CSV file with the processed data to the `output_file_path`. 
    The file will be tab-separated and have 'IID' as the identifier column, with 
    values formatted as integers.

    Example:
    ---------
    ```bash
    python3 process_csv.py /path/to/input_file.csv
    ```
    """
    # Define file paths
    input_file_path = f'{input_filename}'
    output_file_path = f'{input_filename.replace(".csv", "_IID.csv")}'

    # Load the CSV file
    df = pd.read_csv(input_file_path)

    # Check if 'IID' column exists and is of type int
    if 'IID' not in df.columns or not pd.api.types.is_integer_dtype(df['IID']):
        if 'ID' in df.columns:
            # Rename 'ID' to 'IID'
            df.rename(columns={'ID': 'IID'}, inplace=True)
            # Keep the last 7 characters and convert to int
            df['IID'] = df['IID'].astype(str).str[-7:].astype(int)
            # Save the modified file
            df.to_csv(output_file_path, sep='\t', index=False)
        else:
            raise ValueError("Column 'ID' does not exist in the file.")
    else:
        # File already in the right shape
        df.to_csv(output_file_path, sep='\t', index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_csv.py <filename>")
        sys.exit(1)
    
    input_filename = sys.argv[1]
    process_file(input_filename)

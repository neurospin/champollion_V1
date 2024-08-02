import pandas as pd
import statsmodels.api as sm
import sys
from scipy.stats import norm
import argparse

def get_covars_expression(covars_list):
    """
    Generates a list of covariate expressions for use in regression models.

    Parameters:
    -----------
    covars_list : list of str
        A list of covariates specified for the model.

    Returns:
    --------
    list of str
        A list of covariate expressions formatted for use in statistical models.
    """   
    new_covars_list = []
    for cov in covars_list:
        if cov == 'Age':
            new_covars_list.extend(['Age', 'I(Age*Age)'])
        elif cov == 'Sex':
            if 'Age' in covars_list:
                new_covars_list.extend(['I(Age*Sex)', 'I(Age*Age*Sex)'])
        else:
            new_covars_list.append(cov)
            
    return new_covars_list

def process_files(latent_file_path, covar_file_path):
    """
    Processes and pre-residualizes data from a latent file and a covariate file.

    Parameters:
    -----------
    latent_file_path : str
        Path to the latent file (tab-separated) containing latent 
        variables.
    covar_file_path : str
        Path to the covariate file (tab-separated) containing 
        covariates.

    Returns:
    --------
    None
        The function saves the processed data to a new file with '_pre_residualized' 
        suffix.
    """
    # Load data
    latent_df = pd.read_csv(latent_file_path, sep='\t')
    covar_df = pd.read_csv(covar_file_path, sep='\t')
    
    # Identify columns that are not of type int or float
    non_numeric_columns = covar_df.select_dtypes(exclude=['int', 'float']).columns

    # Convert boolean columns to 1 and 0
    for col in non_numeric_columns:
        if covar_df[col].dtype == bool:
            covar_df[col] = covar_df[col].astype(int)

    # Merge the dataframes on 'IID'
    if '#FID' in latent_df.columns:
        merged_df = pd.merge(latent_df.drop('#FID', axis=1), covar_df, on='IID', how='inner')
        covars_list = covar_df.drop(['IID', '#FID'], axis=1).columns
    elif 'FID' in latent_df.columns:
        merged_df = pd.merge(latent_df.drop('FID', axis=1), covar_df, on='IID', how='inner')
        covars_list = covar_df.drop(['IID', 'FID'], axis=1).columns

    print()
    list_cov = get_covars_expression(covars_list)
    print("List of covariates:")
    print(list_cov)
    print()

    # Generate interaction terms
    if 'I(Age*Age)' in list_cov and 'Age' in merged_df.columns:
        merged_df['I(Age*Age)'] = merged_df['Age'] ** 2
    if 'I(Age*Sex)' in list_cov and 'Age' in merged_df.columns and 'Sex' in merged_df.columns:
        merged_df['I(Age*Sex)'] = merged_df['Age'] * merged_df['Sex']
    if 'I(Age*Age*Sex)' in list_cov and 'Age' in merged_df.columns and 'Sex' in merged_df.columns:
        merged_df['I(Age*Age*Sex)'] = (merged_df['Age'] ** 2) * merged_df['Sex']

    # Process each phenotype column
    phenotype_cols = [col for col in latent_df.columns if col not in ['IID','#FID','FID']]

    print()
    print("Pre-residualize with sm.OLS by keeping only the residuals for each dimension for each covariate.")
    print()

    # Pre-residualize and transform data
    for dim_i in phenotype_cols:
        X = merged_df[list_cov]
        X = sm.add_constant(X)  # Adds a constant term to the predictor
        y = merged_df[dim_i]
        model = sm.OLS(y, X, missing='drop').fit()

        # Get residuals
        residuals = model.resid
        merged_df[dim_i] = residuals

    print()
    print("Apply quantile normalization as advised.")
    # Apply quantile normalization
    for dim_i in phenotype_cols:
        ecdf_values = merged_df[dim_i].rank(method='average') / len(merged_df[dim_i])
        qnorm_values = norm.ppf(ecdf_values - 0.5 / len(merged_df[dim_i]))
        merged_df[dim_i] = qnorm_values

    # Define the output file path
    output_file_path = latent_file_path.replace('.phe', '_pre_residualized.txt')
    print()
    print(f"Save the pre-residualized file at {output_file_path} with a sep='\\t'.")
    # Save the pre_residualized file
    if '#FID' in latent_df.columns:
        merged_df[['#FID','IID']+phenotype_cols].to_csv(output_file_path, sep='\t', index=False)
    elif 'FID' in latent_df.columns:
        merged_df[['FID','IID']+phenotype_cols].to_csv(output_file_path, sep='\t', index=False)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Pre-residualize phenotype data using covariates and latent variables.")
    parser.add_argument('latent_file_path', type=str, help='Path to the latent file (tab-separated) containing latent variables.')
    parser.add_argument('covar_file_path', type=str, help='Path to the covariate file (tab-separated) containing covariates.')

    args = parser.parse_args()

    process_files(args.latent_file_path, args.covar_file_path)

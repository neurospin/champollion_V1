"""
Summary:
    prepare_for_fuma.py is a script designed to prepare the summary statistics data from a GWAS (Genome-Wide Association Study) for analysis with FUMA (Functional Mapping and Annotation). 
    The script combines summary statistics from multiple chromosomes, computes effect sizes and standard errors, and saves the combined data to a compressed file.

Arguments:
    --pheno: Path to the pre-residualized phenotype file.
    --sumstats_folder: Path to the folder containing summary statistics files. Be careful, the folder must only contain files from a same study.
    --nb_subjects: Number of subjects in the study (calculated from the pre residualized phenotype file).

Example:
    python3 prepare_for_fuma.py --pheno /path/to/pheno_pre_residualized.txt --sumstats_folder /path/to/sumstats_folder --nb_subjects 12345
"""


import os
import argparse
import numpy as np
import pandas as pd

def compute_effect_size(row, n):
    z = row['Z_FAKE']
    p = row['N']/n
    se = 1 / np.sqrt(2*p*(1 - p)*(n + z**2))
    beta = z * se
    return pd.Series({'beta': beta, 'SE': se})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for FUMA')
    parser.add_argument('--pheno', type=str, required=True, help='Path to pre-residualized phenotype file')
    parser.add_argument('--sumstats_folder', type=str, required=True, help='Path to the folder containing summary statistics')
    parser.add_argument('--nb_subjects', type=int, required=False, help='Number of subjects')

    args = parser.parse_args()

    pheno_path = args.pheno
    sumstats_folder = args.sumstats_folder
    n_subjects = args.nb_subjects + 1

    # Extract the {ref} prefix from the phenotype file name
    # ref_prefix = os.path.basename(pheno_path).split('_pheno_pre_residualized.txt')[0]
    ref_prefix = os.path.basename(pheno_path).split('_pheno.phe')[0]

    sumstats_all_chr = []
    print("Files:")
    for file in os.listdir(sumstats_folder):
        if file.endswith('.most_perm.sumstats'): 
            sumstats_path = os.path.join(sumstats_folder, file)
            mostest_output = pd.read_csv(sumstats_path, sep='\t')
            print(f"{file} loaded")
            # is it a correct way to calculate beta and the SE ?
            mostest_output[['beta', 'SE']] = mostest_output.apply(compute_effect_size, axis=1, args=(n_subjects,))
            sumstats_all_chr.append(mostest_output)

    sumstats_all_chr_df = pd.concat(sumstats_all_chr)
    output_path = os.path.join(sumstats_folder, f"FUMA/{ref_prefix}_mostest_all_chr.most_perm.sumstats.gz") #.gz
    sumstats_all_chr_df.to_csv(output_path, sep='\t', index=False, compression='gzip')

    sumstats_all_chr = []
    print("Files:")
    for file in os.listdir(sumstats_folder):
        if file.endswith('.most_orig.sumstats'): 
            sumstats_path = os.path.join(sumstats_folder, file)
            mostest_output = pd.read_csv(sumstats_path, sep='\t')
            print(f"{file} loaded")
            # is it a correct way to calculate beta and the SE ?
            mostest_output[['beta', 'SE']] = mostest_output.apply(compute_effect_size, axis=1, args=(n_subjects,))
            sumstats_all_chr.append(mostest_output)

    sumstats_all_chr_df = pd.concat(sumstats_all_chr)
    output_path = os.path.join(sumstats_folder, f"FUMA/{ref_prefix}_mostest_all_chr.most_orig.sumstats.gz") #.gz
    sumstats_all_chr_df.to_csv(output_path, sep='\t', index=False, compression='gzip')


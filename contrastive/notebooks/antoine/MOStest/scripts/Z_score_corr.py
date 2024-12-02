import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def correlation_heatmap(z_score, path_to_save):
    '''
    This function calculates and visualizes the correlation matrix of SNPs from a given region,
    creating a heatmap without using seaborn.
    
    Parameters:
    z_score (DataFrame): The input DataFrame containing SNP data. It should include 'CHR', 'SNP', 'PVAL', and additional dimensions (dim1, dim2, ...).

    Returns:
    correlation_matrix (DataFrame): The correlation matrix of SNPs based on their dimensions.
    
    Visualization:
    The function will plot a heatmap of the correlation matrix.
    '''

    # Drop non-numerical columns if necessary
    df_numerical = z_score.drop(['CHR', 'SNP', 'PVAL', 'N', 'FREQ'], axis=1) #['CHR', 'SNP', 'PVAL']
    df_transposed = df_numerical.T

    # Calculate correlation between rows
    correlation_matrix = df_transposed.corr()
    
    # Create labels for the axes
    labels = z_score['CHR'].astype(str) + " - " + z_score['SNP']
    correlation_matrix.columns = labels
    correlation_matrix.index = labels
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 12))
    cax = ax.matshow(correlation_matrix, cmap='coolwarm')

    # Add color bar with custom size and position
    cbar = fig.colorbar(cax, ax=ax, shrink=0.7, fraction=0.046, pad=0.04)

    # Set tick labels and adjust font size
    #ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    # Display every nth label to avoid clutter (adjust `step` based on the number of labels)
    step = max(1, len(labels) // 40)
    #ax.set_xticks(np.arange(0, len(labels), step))
    ax.set_yticks(np.arange(0, len(labels), step))

    # Update tick labels and alignment
    #ax.set_xticklabels(labels[::step], rotation=45, ha='center', fontsize=10)  # Center alignment
    ax.set_yticklabels(labels[::step], fontsize=10)

    # Adjust the spacing of the labels
    #ax.tick_params(axis='x', which='major', pad=10)  # Increase space between labels and axis

    # Title and axis labels
    ax.set_title("Correlation Matrix Between SNPs", pad=40, fontsize=14)
    ax.set_xlabel("SNP (CHR - SNP ID)", fontsize=12)
    ax.set_ylabel("SNP (CHR - SNP ID)", fontsize=12)

    # Adjust layout for better fit
    plt.tight_layout()
    plt.savefig(f'{path_to_save}/Correlation_Matrix_SNPs_MOSTest.eps', format='eps')
    #plt.show()
    
    return correlation_matrix


z_score = pd.read_csv('/volatile/ad279118/Irene/MOSTEST/Results/CINGULATE_left_mostest_all_chr.most_orig.zmat.csv', sep='\t')

correlation_heatmap(z_score, 'pou')


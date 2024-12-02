import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
nb_zscores_tables = 10
nb_z_per_summary = 1500

# Colors for each Z-score set
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']

# Initialize the plot
plt.figure(figsize=(14, 10))  # Increased height for better spacing

# Loop over each dataset, overwrite df, and plot

for i in range(nb_zscores_tables):
    # Generate the dataset
    df = pd.DataFrame(np.random.randn(nb_z_per_summary, 5), columns=['CHR', 'SNP', 'PVAL', 'N', 'FREQ'])
    
    # Random chromosome assignment (1 to 22)
    df['CHR'] = np.random.randint(1, 23, size=nb_z_per_summary)
    # Random SNP positions (within 1 Mbp per chromosome)
    df['POS'] = np.random.randint(1, 1_000_000, size=nb_z_per_summary)
    # Random P-values (0 to 1)
    df['PVAL'] = np.abs(np.random.rand(nb_z_per_summary))
    df['neg_log_pval'] = -np.log10(df['PVAL'])
    
    # Sort by chromosome and position
    df.sort_values(by=['CHR', 'POS'], inplace=True)
    
    # Calculate cumulative position for x-axis alignment
    chrom_offsets = df.groupby('CHR')['POS'].max().cumsum().shift(fill_value=0)
    df['x_val'] = df.apply(lambda row: row['POS'] + chrom_offsets.loc[row['CHR']], axis=1)
    
    # Apply vertical offset based on the dataset index
    df['neg_log_pval_shifted'] = df['neg_log_pval']
    
    # Plot the SNPs for this dataset
    plt.scatter(df['x_val'], df['neg_log_pval_shifted'], 
                c=colors[i % len(colors)], s=2, label=f'Z_score{i}', alpha=0.7)

# Customize plot labels and title
plt.xlabel('Genomic Position')
plt.ylabel('-log10(p-value)')
plt.title('Manhattan Plot of Multiple tests')

# Add chromosome labels at their midpoints
chromosome_ticks = [chrom_offsets[chrom] + df[df['CHR'] == chrom]['POS'].max() / 2 for chrom in sorted(df['CHR'].unique())]
chromosome_labels = [f"Chr {chrom}" for chrom in sorted(df['CHR'].unique())]

plt.xticks(chromosome_ticks, chromosome_labels, rotation=45)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
plt.tight_layout()
plt.show()




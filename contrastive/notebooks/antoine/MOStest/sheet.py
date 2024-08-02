import pandas as pd

pheno = pd.read_csv('~/Bureau/Irene/MOSTEST/Data_exemple/pre_residualized_pheno.txt',sep='\t')# delim_whitespace=True)
pheno['IID'] = [f'id2_{i}' for i in range(20000,0,-2)]

print(pheno.head())
print()

fam = pd.read_csv('~/Bureau/Irene/MOSTEST/Data_exemple/chr21.fam', delim_whitespace=True, header=None)

print(fam.head())
print()

fam.iloc[:,2] = fam.iloc[:,2] + [i for i in range(0,10000)]
fam.iloc[:,3] = fam.iloc[:,3] + [i for i in range(0,10000)]

print(fam.head())
print()
pheno.to_csv('~/Bureau/Irene/MOSTEST/Data_exemple/pre_residualized_pheno.txt', sep='\t', index=False)
fam.to_csv('~/Bureau/Irene/MOSTEST/Data_exemple/chr21_for_reorder.fam',sep=' ', header=False, index=False)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from sklearn.decomposition import PCA


#base_path = '/neurospin/dico/jlaval/Output/5_models_orbital_3_layer_proj/pepper_05_*/right_21045_UKB_random_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj/ResNet_*/right_21043_UKB_random_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj/pepper_05_*/three_datasets_UKB_random_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0/FIP_right/20-16-33_3/three_datasets_UKB_random_epoch*_embeddings/full_embeddings.csv'
#base_path = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0/FIP_left/14-56-46_3/three_datasets_UKB_random_epoch*_embeddings/full_embeddings.csv'
base_path = '/neurospin/dico/jlaval/Output/5_models_FIP_right_3_layer_proj/Conv_net_preV1/16-48-07_*/three_datasets_UKB_random_embeddings/full_embeddings.csv'

# Use glob to find all matching files
file_paths = glob.glob(base_path)


for file in file_paths:
    embeddings_UKB = pd.read_csv(file, index_col=0)
    print("\n", file)
    # Print basic statistics
    #print("\n", embeddings_UKB.describe(), "\n")

    print("Minimum std among the dimensions:")
    print(embeddings_UKB.std(axis=0).min(), "\n")

    print("Maximum std among the dimensions:")
    print(embeddings_UKB.std(axis=0).max(), "\n")

    # Perform PCA
    n_components = 32
    pca = PCA(n_components=n_components)
    pca.fit(embeddings_UKB)

    print("Explained variance ratio for each PC:")
    print(pca.explained_variance_ratio_, "\n")

    # Number of components to explain 99% of the variance
    print((np.cumsum(pca.explained_variance_ratio_) < 0.99).sum()+1)

    if False:
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
        plt.grid(True)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA on the Combined Embeddings')
        plt.show()
    
    nb_dim_to_keep = (np.cumsum(pca.explained_variance_ratio_) < 0.99).sum()+1
    pca = PCA(n_components=nb_dim_to_keep)
    pca.fit(embeddings_UKB)
    pca_embeddings_UKB = pd.DataFrame(pca.transform(embeddings_UKB), columns=[f'dim{i}' for i in range(1,nb_dim_to_keep+1)],  index=embeddings_UKB.index)
    print(pca_embeddings_UKB.head())
    initial_path = file.replace('/full_embeddings.csv', '')
    path_to_save=f'{initial_path}/full_pc.csv'
    pca_embeddings_UKB.to_csv(path_to_save)
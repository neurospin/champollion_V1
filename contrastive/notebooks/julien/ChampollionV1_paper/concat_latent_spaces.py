import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

#savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/'
#savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/'
savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/'
#dfs_dir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/embeddings/ukb40_epoch80_embeddings'
#dfs_dir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/embeddings/ukb40_epoch80_embeddings'
n_dims = 32
#nb_subs = None
nb_subs = 10000
max_iter = 100
#metric = 'roc_auc_ovr_weighted'
metric = 'balanced_accuracy'

regions_to_treat = ['SOr_left',
 'SOr_right',
 'FColl-SRh_left',
 'SFmedian-SFpoltr-SFsup_left',
 'SFinf-BROCA-SPeCinf_left',
 'SPoC_left',
 'fronto-parietal_medial_face_left',
 'FIP_left',
 'CINGULATE_left',
 'SC-SPoC_left',
 'SFinter-SFsup_left',
 'FCMpost-SpC_left',
 'SsP-SPaint_left',
 'SOr-SOlf_left',
 'FPO-SCu-ScCal_left',
 'LARGE_CINGULATE_left',
 'SFmarginal-SFinfant_left',
 'SFint-FCMant_left',
 'STi-STs-STpol_left',
 'SFint-SR_left',
 'Lobule_parietal_sup_left',
 'STi-SOTlat_left',
 'SPeC_left',
 'STsbr_left',
 'ScCal-SLi_left',
 'STs_left',
 'FCLp-subsc-FCLa-INSULA_left',
 'SC-sylv_left',
 'SC-SPeC_left',
 'OCCIPITAL_left',
 'FColl-SRh_right',
 'SFmedian-SFpoltr-SFsup_right',
 'SFinf-BROCA-SPeCinf_right',
 'SPoC_right',
 'fronto-parietal_medial_face_right',
 'FIP_right',
 'CINGULATE_right',
 'SC-SPoC_right',
 'SFinter-SFsup_right',
 'FCMpost-SpC_right',
 'SsP-SPaint_right',
 'SOr-SOlf_right',
 'FPO-SCu-ScCal_right',
 'LARGE_CINGULATE_right',
 'SFmarginal-SFinfant_right',
 'SFint-FCMant_right',
 'STi-STs-STpol_right',
 'SFint-SR_right',
 'Lobule_parietal_sup_right',
 'STi-SOTlat_right',
 'SPeC_right',
 'STsbr_right',
 'ScCal-SLi_right',
 'STs_right',
 'FCLp-subsc-FCLa-INSULA_right',
 'SC-sylv_right',
 'SC-SPeC_right',
 'OCCIPITAL_right']

# UKB # TODO : labels dir is also a parameter ...
dataset = 'ukb40'

keywords = ['_left', '_right']
all_matches = []
for var in keywords:
    pattern = f"{savedir}*{var}/*/{dataset}_random_embeddings/train_val_embeddings.csv" ## TODO : Make sure only one model per region, take highest epoch, and print
    matches = glob.glob(pattern)
    all_matches.extend(matches)
# Optional: remove duplicates
dfs_dirs = list(set(all_matches))
dfs_dirs.sort()

regions_treated = [elem.split(savedir)[1].split('/')[0] for elem in dfs_dirs]
print(regions_treated)
print(f'Number of regions treated : {len(regions_treated)}')
print(f'Missing regions : {set(regions_to_treat) - set(regions_treated)}')


labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/participants_sex_age.csv')
labels.columns = ['ID'] + labels.columns[1:].tolist()
label = 'Sex'

## HCP
"""
dataset = 'hcp'
dfs_dirs = glob.glob("/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/*/*/hcp_random_epoch80_embeddings/full_embeddings.csv")
#labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/FIP_labels.csv')
#label = 'Right_FIP'
labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/hcp_OFC_labels.csv')
label = 'Left_OFC'
labels.columns = ['ID'] + labels.columns[1:].tolist()
"""

### TO REWRTIE, BECAUSE CAN CONCAT DIRECTLY THE NPYS

## load and rename the columns of each df
embd_list = []
print('Loading the embeddings...')
for i, directory in enumerate(tqdm(dfs_dirs)):
    embd=pd.read_csv(directory, nrows=nb_subs)
    embd.columns = ['ID'] + [f'dim_{i}_{j}' for j in range(n_dims)]
    embd_list.append(embd)
## merge all the dfs
print('Merging the embeddings...')
embd = embd_list[0]
for i in tqdm(range(1, len(embd_list))):
    embd = pd.merge(embd, embd_list[i], on='ID', how='outer')
# drop all 'ID' columns but the first
embd = embd.loc[:, ~embd.columns.duplicated()]

# get the labels
embd_with_labels = embd.merge(labels, on='ID', how='inner')

# cross validation with stratified kfold for Sex and Age
embd_with_labels['age_bin'] = pd.qcut(embd_with_labels['Age'], q=10, labels=False, duplicates='drop')
embd_with_labels['stratify_label'] = embd_with_labels['Sex'].astype(str) + "_" + embd_with_labels['age_bin'].astype(str)
stratify_labels = embd_with_labels['stratify_label']
skf = StratifiedKFold(n_splits=5, shuffle=False)

Y = embd_with_labels[label]
# keep only the columns whose name starts with 'dim'
X = embd_with_labels.loc[:, embd_with_labels.columns.str.startswith('dim')]
# apply standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

cv=skf.split(X, stratify_labels)

model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter)
#param_grid = {
#    'C': [0.001, 0.01, 0.1, 1, 10, 100],
#    'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1],
#}
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0., 0.2, 0.4, 0.6, 0.8, 1.],
}

clf = GridSearchCV(model, param_grid, cv=cv, refit=False, scoring=metric, n_jobs=-1)


print(embd_with_labels)


print('Training the classifier...')

clf.fit(X, Y)
print(f'best params : {clf.best_params_}')
print(f'best {metric} : {clf.best_score_}')
#print("Number of iterations:", clf.best_estimator_.n_iter_)

# Convert cv_results_ to DataFrame
results_df = pd.DataFrame(clf.cv_results_)
# Save to CSV
results_df.to_csv(os.path.join(savedir, f"{dataset}_{label}_global_classif_{metric}_{nb_subs}subs_{max_iter}iter.csv"), index=False)


# same with PCA
# redefine the clf
#clf = GridSearchCV(model, param_grid, cv=5, refit=True, scoring=metric, n_jobs=-1)

# select n_features to keep 99.9% of the variance
#pca = PCA(n_components=0.999)
#X_pca = pca.fit_transform(X)
#print(f'original shape: {X.shape}')
#print(f'reduced shape: {X_pca.shape}')

#clf.fit(X_pca, Y)
#print(f'best params : {clf.best_params_}')
#print(f'best score : {clf.best_score_}')
#print("Number of iterations:", clf.best_estimator_.n_iter_)
# Convert cv_results_ to DataFrame
#results_df = pd.DataFrame(clf.cv_results_)
# Save to CSV
#results_df.to_csv(os.path.join(savedir, f"{dataset}_{label}_global_classif_{nb_subs}subs_{max_iter}iter_pca.csv"), index=False)

#######
# using the best params, fit the model again with all the data
#######

model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter, C=clf.best_params_['C'], l1_ratio=clf.best_params_['l1_ratio'])
model.fit(X, Y)

# eval the model on external dataset
external_dataset = 'hcp'
#dfs_dirs = glob.glob("/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/*/*/hcp_random_epoch80_embeddings/full_embeddings.csv")


## use the same regions as UKB, and concat in the same order
#regions = [('_').join(elem.split('_')[:2]) for elem in dfs_dirs_ukb]
keywords = ['_left', '_right']
all_matches = []
for var in keywords:
    pattern = f"{savedir}*{var}/*/{external_dataset}_random_embeddings/train_val_embeddings.csv" ## TODO : ALL MODELS DID NOT RUN 80 EPOCHS, LOAD WITH GLOB.GLOB
    matches = glob.glob(pattern)
    all_matches.extend(matches)
# Optional: remove duplicates
dfs_dirs = list(set(all_matches))
dfs_dirs.sort()
print(dfs_dirs)

regions_treated = [elem.split(savedir)[1].split('/')[0] for elem in dfs_dirs]
print(regions_treated)
print(f'Number of regions treated : {len(regions_treated)}')
print(f'Missing regions : {set(regions_to_treat) - set(regions_treated)}')

#labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/FIP/FIP_labels.csv')
#label = 'Right_FIP'
external_labels = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/hcp/participants.csv', usecols=['Subject', 'Gender'])
external_labels.columns = ['ID', 'Sex']
# replace Sex M, F with 1, 0
external_labels['Sex'] = [1 if sex=='M' else 0 for sex in external_labels['Sex'].tolist()]


## load and rename the columns of each df
embd_list = []
print('Loading the embeddings...')
for i, directory in enumerate(tqdm(dfs_dirs)):
    embd=pd.read_csv(directory)
    embd.columns = ['ID'] + [f'dim_{i}_{j}' for j in range(n_dims)]
    embd_list.append(embd)
## merge all the dfs
print('Merging the embeddings...')
embd = embd_list[0]
for i in tqdm(range(1, len(embd_list))):
    embd = pd.merge(embd, embd_list[i], on='ID', how='outer')
# drop all 'ID' columns but the first
#embd = embd.loc[:, ~embd.columns.duplicated()]

# get the labels
embd_with_labels = embd.merge(external_labels, on='ID', how='inner')

print(embd_with_labels)

Y = embd_with_labels[label].to_numpy()
# keep only the columns whose name starts with 'dim'
X = embd_with_labels.loc[:, embd_with_labels.columns.str.startswith('dim')].to_numpy()
# apply standardization
X = scaler.transform(X)

Y_proba = model.predict_proba(X)
# get the balanced accuracy_score
print(f'roc_auc_score: {roc_auc_score(Y, Y_proba[:, 1])}')
Y_pred = model.predict(X)
print(f'balanced accuracy score: {balanced_accuracy_score(Y, Y_pred)}')

######
# save the model
######
# Coefficients and intercept
coef = model.coef_          # shape: (1, n_features) for binary classification
intercept = model.intercept_

# Combine into a DataFrame
weights_df = pd.DataFrame(coef, columns=[f'feature_{i}' for i in range(coef.shape[1])])
weights_df['intercept'] = intercept  # add intercept as a column

# Save to CSV
weights_df.to_csv(os.path.join(savedir, f'{external_dataset}_{label}_{metric}_maxiter{max_iter}_{nb_subs}_logreg_weights.csv'), index=False)


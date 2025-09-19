import numpy as np
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut

dataset_localization = '/neurospin/dico/data/deep_folding/current/datasets/' # Jean Zay : '/lustre/fswork/projects/rech/tgu/umy22uu/Runs/70_self-supervised_two-regions/Input/'
n_jobs = 30

#### FIP right ####
embds_dir = '/neurospin/dico/jlaval/Output/nsubjects/FIP_right_1000/1_1000_subjectsname13-50-18_138/hcp_random_epoch400_embeddings/full_embeddings.csv'
labels_dir = f'{dataset_localization}/hcp/FIP/FIP_labels.csv'
label = 'Right_FIP'
splits_basedir = f'{dataset_localization}/hcp/FIP/Right/train_val_split_'
test_subs_dir = f'{dataset_localization}/hcp/FIP/Right/test_split.csv'
label_type = 'binary'
subject_name = 'Subject'

#### SOr_left ####
# embds_dir = '/neurospin/dico/jlaval/Output/nsubjects/SOr_left_1000/1_1000_subjectsname13-59-27_128/hcp_random_epoch400_embeddings/full_embeddings.csv'
# labels_dir = f'{dataset_localization}/hcp/hcp_OFC_labels.csv'
# label = 'Left_OFC'
# splits_basedir = f'{dataset_localization}/orbital_patterns/Troiani/Left/train_val_split_'
# test_subs_dir = f'{dataset_localization}/orbital_patterns/Troiani/Left/test_split.csv'
# label_type = 'multiclass'
# subject_name = 'Subject'

#### LARGE_CINGULATE_right ####
# embds_dir = '/neurospin/dico/jlaval/Output/nsubjects/LARGE_CINGULATE_right_1000/1_1000_subjectsname13-59-14_121/ACCpatterns_random_epoch400_embeddings/full_embeddings.csv'
# labels_dir = f'{dataset_localization}/ACCpatterns/subjects_labels.csv'
# label = 'Right_PCS'
# splits_basedir = f'{dataset_localization}/ACCpatterns/splits/Right/train_val_split_'
# test_subs_dir = f'{dataset_localization}/ACCpatterns/splits/Right/test_split.csv'
# label_type = 'binary'
# subject_name = 'long_name'

# load
embds = pd.read_csv(embds_dir)
embds.columns = ['ID'] + [f'dim{i}' for i in range(embds.shape[1]-1)]
# remove duplicates
embds = embds.drop_duplicates(subset=['ID'])
labels = pd.read_csv(labels_dir)

# restrict embds to Subjects with labels
embds = embds[embds['ID'].isin(labels[subject_name])].reset_index(drop=True)
# same for labels
labels = labels[labels[subject_name].isin(embds['ID'])].reset_index(drop=True)
# align labels and embds on 'ID'
labels = labels.merge(embds[['ID']], left_on=subject_name, right_on='ID', how='right')
# order all by ID
embds = embds.sort_values(by='ID').reset_index(drop=True)
labels = labels.sort_values(by='ID').reset_index(drop=True)
subjects = embds['ID']

# define X, Y and subjects
X = embds.drop(columns=['ID'])
# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)
Y = labels[label]
subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X), 'Y': Y})

# custom splits
root_dir = '/'.join(splits_basedir.split('/')[:-1])
basedir = splits_basedir.split('/')[-1]
splits_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(basedir) and '.csv' in f]
splits_subs = [pd.read_csv(file, header=None) for file in splits_dirs]
labels = np.concatenate([[i] * len(K) for i, K in enumerate(splits_subs)])
splits_subs_and_labels = pd.concat(splits_subs)
splits_subs_and_labels.columns=['ID']
splits_subs_and_labels['labels'] = labels
df = subs_embeddings.merge(splits_subs_and_labels, on='ID')
groups, X_train_val, Y_train_val = df['labels'], np.vstack(df['X'].values), df['Y']
logo = LeaveOneGroupOut()
cv = [*(logo.split(X_train_val, Y_train_val, groups=groups))]

# LINEAR PROBING
parameters={'l1_ratio': np.linspace(0,1,11), 'C': [10**k for k in range(-3,4)]}
model = LogisticRegression(solver='saga', penalty='elasticnet',
                            max_iter=2000, random_state=0)
clf = GridSearchCV(model, parameters, cv=cv, scoring='roc_auc_ovr_weighted', refit=False, n_jobs=-1)

# fit cross-validation
clf.fit(X_train_val,Y_train_val)
print(f'best params : {clf.best_params_}')
print(f'best score : {clf.best_score_}')
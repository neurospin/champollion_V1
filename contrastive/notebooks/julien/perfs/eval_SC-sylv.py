import numpy as np
import os 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score, GridSearchCV

##########
embds_dir = "/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/SC-sylv_left/name07-58-00_111/hcp_random_embeddings/full_embeddings.csv"
parameters={'l1_ratio': np.linspace(0,1,11), 'alpha': [10**k for k in range(-3,4)], 'max_iter': [10000]}
dataset_localization = '/neurospin/dico/data/deep_folding/current/datasets/' # Jean Zay : '/lustre/fswork/projects/rech/tgu/umy22uu/Runs/70_self-supervised_two-regions/Input/'
hemisphere = 'left'  # 'left' or 'right'
n_jobs = 30
##########

labels_dir = f'{dataset_localization}/hcp/hcp_isomap_labels_SC-sylv_{hemisphere}.csv'
label_list = [f'Isomap_central_{hemisphere}_dim{k}' for k in range(1,7)]
splits_basedir = f'{dataset_localization}/hcp/Isomap/splits/train_val_split_'
test_subs_dir = f'{dataset_localization}/hcp/Isomap/splits/test_split.csv'
subject_name = 'Subject'

# store score for each regression
cross_val_r2_list = []
test_r2_list = []

# load embeddings
embds = pd.read_csv(embds_dir)
embds.columns = ['ID'] + [f'dim{i}' for i in range(embds.shape[1]-1)]
# remove duplicates
embds = embds.drop_duplicates(subset=['ID'])
# load labels
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

# get the custom splits
subs_embeddings = pd.DataFrame({'ID': subjects, 'X': list(X)})
root_dir = '/'.join(splits_basedir.split('/')[:-1])
basedir = splits_basedir.split('/')[-1]
splits_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(basedir) and '.csv' in f]
splits_subs = [pd.read_csv(file, header=None) for file in splits_dirs]
folds = np.concatenate([[i] * len(K) for i, K in enumerate(splits_subs)])
splits_subs = pd.concat(splits_subs)
splits_subs.columns=['ID']
splits_subs['labels'] = folds
df = subs_embeddings.merge(splits_subs, on='ID')
groups, X_train_val = df['labels'], np.vstack(df['X'].values)

# get test
test_subjects = pd.read_csv(test_subs_dir, header=None)
test_subjects.columns = ['ID']
subs_embeddings_test = subs_embeddings.merge(test_subjects, on='ID')
X_test = np.vstack(subs_embeddings_test['X'].values)

for label in label_list:

    # merge label with embeddings
    # first, train val
    df_label = labels[['ID', label]].rename(columns={label: 'Y'})
    df_y = df.merge(df_label, on='ID')
    Y_train_val = df_y['Y']
    # then, test
    subs_embeddings_test_y = subs_embeddings_test.merge(df_label, on='ID')
    Y_test = subs_embeddings_test_y['Y']

    # instantiate cross-validation
    logo = LeaveOneGroupOut()
    cv = [*(logo.split(X_train_val, Y_train_val, groups=groups))]
    # define model
    model = ElasticNet()
    clf = GridSearchCV(model, parameters, cv=cv, scoring='r2', refit=True, n_jobs=n_jobs)

    # fit cross-validation
    clf.fit(X_train_val,Y_train_val)
    print(f'best params : {clf.best_params_}')
    print(f'best score : {clf.best_score_}')
    md = clf.best_estimator_
    # compute r2 on cross-validation
    cross_val_r2 = cross_val_score(md, X_train_val, Y_train_val, cv=cv, scoring='r2')
    print(f'Cross-val R2: {cross_val_r2.mean():.3f}')

    # compute r2 on test
    test_r2 = md.score(X_test, Y_test)
    print(f'Test R2: {test_r2:.3f}')

    cross_val_r2_list.append(cross_val_r2)
    test_r2_list.append(test_r2)

mean_cross_val_r2 = np.mean(cross_val_r2_list)
mean_test_r2 = np.mean(test_r2_list)
print(f'Mean cross-val R2 across labels: {mean_cross_val_r2:.3f}')
print(f'Mean test R2 across labels: {mean_test_r2:.3f}')
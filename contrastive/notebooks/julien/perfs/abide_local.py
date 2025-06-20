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
from sklearn.model_selection import LeaveOneGroupOut

#savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/'
#savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/'
savedir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation/'
#dfs_dir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/embeddings/ukb40_epoch80_embeddings'
#dfs_dir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation_latent_256/embeddings/ukb40_epoch80_embeddings'
n_dims = 32
max_iter = 100
metric = 'roc_auc'

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

# select the train val subjects
train_val_subjects_dirs = glob.glob('/neurospin/dico/data/deep_folding/current/datasets/aggregate_autism/splits/train_val_*')
# load each split subject file and create cv splits from them
train_val_subjects = []
for i, directory in enumerate(train_val_subjects_dirs):
    train_val_subjects.append(pd.read_csv(directory, sep='\t', header=None))
train_val_subjects = pd.concat(train_val_subjects, axis=0)
train_val_subjects.columns = ['ID']
train_val_subjects['ID'] = train_val_subjects['ID'].astype(str)

# select the test subjects
test_subjects = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/aggregate_autism/splits/internal_test.csv', header=None)
test_subjects.columns = ['ID']
test_subjects['ID'] = test_subjects['ID'].astype(str)
# select the test extra
test_extra_subjects = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/aggregate_autism/splits/external_test.csv', header=None)
test_extra_subjects.columns = ['ID']
test_extra_subjects['ID'] = test_extra_subjects['ID'].astype(str)

# load labels
labels1 = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/abide1/20231108_participants.tsv', usecols=['participant_id', 'diagnosis'], sep='\t')
labels2 = pd.read_csv('/neurospin/dico/data/deep_folding/current/datasets/abide2/20231108_participants.tsv', usecols=['participant_id', 'diagnosis'], sep='\t')
labels = pd.concat([labels1, labels2], axis=0)
labels.columns = ['ID'] + labels.columns[1:].tolist()
label = 'diagnosis'

# get the custom cv
splits_basedir = '/neurospin/dico/data/deep_folding/current/datasets/aggregate_autism/splits/train_val_split_'
root_dir = '/'.join(splits_basedir.split('/')[:-1])
basedir = splits_basedir.split('/')[-1]
splits_dirs = [os.path.join(root_dir,f) for f in os.listdir(root_dir) if f.startswith(basedir) and '.csv' in f]
splits_subs = [pd.read_csv(file, header=None) for file in splits_dirs]
splits = np.concatenate([[i] * len(K) for i, K in enumerate(splits_subs)])
splits_subs_and_labels = pd.concat(splits_subs)
splits_subs_and_labels.columns=['ID']
splits_subs_and_labels['ID'] = splits_subs_and_labels['ID'].astype(str)
splits_subs_and_labels['ID'] = ['sub-'+elem for elem in splits_subs_and_labels['ID']]
splits_subs_and_labels['splits'] = splits

dataset = 'agg_abide'
keywords = ['_left', '_right']
all_matches = []
for var in keywords:
    pattern = f"{savedir}*{var}/*/{dataset}_random_embeddings/full_embeddings.csv" ## TODO : Make sure only one model per region, take highest epoch, and print
    matches = glob.glob(pattern)
    all_matches.extend(matches)
# Optional: remove duplicates
dfs_dirs = list(set(all_matches))
dfs_dirs.sort()

results = {}


## iterate on the regions
for i, directory in enumerate(tqdm(dfs_dirs)):

    embd=pd.read_csv(directory)
    embd = embd.loc[:, ~embd.columns.duplicated()]

    # fit standard scaler on embd
    std = StandardScaler()
    embd_matrix = embd.loc[:, embd.columns.str.startswith('dim')]
    std.fit(embd_matrix)

    embd = embd.loc[:, ~embd.columns.duplicated()]
    embd['ID'] = embd['ID'].astype(str)

    ## restrict embeddings to one single run
    # remove the 'run' information
    embd['ID'] = embd['ID'].str.split('_ses').str[0]
    # remove rows with duplicate ID
    embd = embd.drop_duplicates(subset=['ID'], keep='first')


    # add a train val label
    embd["train_val"] = embd['ID'].apply(lambda x: 1 if any((i in x) for i in train_val_subjects['ID'].tolist()) else 0)
    # add a inter set label
    embd["test"] = embd['ID'].apply(lambda x: 1 if any((i in x) for i in test_subjects['ID'].tolist()) else 0)
    # add an extra set label
    embd["test_extra"] = embd['ID'].apply(lambda x: 1 if any((i in x) for i in test_extra_subjects['ID'].tolist()) else 0)

    # add label
    embd = pd.merge(embd, labels, on='ID', how='left')

    embd_train_val = embd.loc[embd['train_val']==1]

    # merge with embd
    embd_train_val = pd.merge(embd_train_val, splits_subs_and_labels, on='ID', how='left')

    X = embd_train_val.loc[:, embd_train_val.columns.str.startswith('dim')]
    Y = embd_train_val.loc[:, label]

    # apply std and pca to X
    X = std.transform(X)

    groups = embd_train_val.loc[:, 'splits']
    logo = LeaveOneGroupOut()
    cv = [*(logo.split(X, Y, groups=groups))]

    model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'l1_ratio': [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.],
    }

    clf = GridSearchCV(model, param_grid, cv=cv, refit=False, scoring=metric, n_jobs=-1)
    clf.fit(X, Y)
    best_params = clf.best_params_
    print(f'Best parameters: {best_params}')
    print(f'Best score: {clf.best_score_}')

    model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=max_iter, C=clf.best_params_['C'], l1_ratio=clf.best_params_['l1_ratio'])
    model.fit(X, Y)

    ## eval overfitting on train val set
    Y_pred_proba = model.predict_proba(X)
    roc_auc_train_val = roc_auc_score(Y, Y_pred_proba[:, 1])
    print(f'ROC AUC on train val set: {roc_auc_train_val}')

    ## eval on test set
    embd_test = embd.loc[embd['test']==1]
    X_test = embd_test.loc[:, embd_test.columns.str.startswith('dim')]
    # apply std and pca to X
    X_test = std.transform(X_test)

    Y_test = embd_test.loc[:, label]
    Y_test_pred_proba = model.predict_proba(X_test)
    # compute roc_auc
    roc_auc_test = roc_auc_score(Y_test, Y_test_pred_proba[:, 1])
    print(f'ROC AUC on test set: {roc_auc_test}')

    results[directory] = {
        'best_params': best_params,
        'best_score': clf.best_score_,
        'roc_auc_train_val': roc_auc_train_val,
        'roc_auc_test': roc_auc_test
    }

    ## eval on test extra set
    #embd_test_extra = embd.loc[embd['test_extra']==1]
    #X_test_extra = embd_test_extra.loc[:, embd_test_extra.columns.str.startswith('dim')]
    # apply std and pca to X
    #X_test_extra = std.transform(X_test_extra)
    #X_test_extra = pca.transform(X_test_extra)

    #Y_test_extra = embd_test_extra.loc[:, label]
    #Y_test_extra_pred_proba = model.predict_proba(X_test_extra)
    # compute roc_auc
    #roc_auc_extra = roc_auc_score(Y_test_extra, Y_test_extra_pred_proba[:, 1])
    #print(f'ROC AUC on test extra set: {roc_auc_extra}')

# save results
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df = results_df.reset_index()
results_df.to_csv(os.path.join(savedir, f'pred_asd_local_ndims{n_dims}.csv'), index=False)


# %% [markdown]
# ## Purpose: to see the the evolution of correlation between the predictions through epochs

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os


# %% [markdown]
# #### Load the embeddings for a given model

# %%
def get_models_path(rootdir):
    """
    Select all the model directories from a multirun session.

    Parameters
    ----------
    rootdir: str
        path which leads to the directories of the models
    
    Return
    ------
    list_model: list of str
        list of the directories of the different models
    """
    for d in os.listdir(rootdir):
        if d.endswith('.csv'):
    #        with open(f"{rootdir}{el}", 'r') as csvfile:
    #            for line in csvfile:
    #                 print(line)
            list_model = pd.read_csv(f'{rootdir}{d}')
            list_model = np.array([i[0] for i in list_model.to_numpy()])
        else :
            list_model = [d for d in os.listdir(rootdir) if '.' not in d]
            return list_model
    list_model = list_model[[model in os.listdir(rootdir) for model in list_model]]
    return list_model

def check_embeddings(path):
    """
    Check if the embeddings were created for ACCP, HCP and UKB.

    Parameters
    ----------
    path: str
        path which leads to the directories of the embeddings for one given model
    
    Return
    ------
    check_value: bol
        True if ACCP, UKB and HCP have at least one representant
    """
    check_value_ACCP = False
    check_value_HCP = False
    check_value_UKB = False

    list_ebdd_ACCP = ['acc_', 'accp']
    list_ebdd_HCP = ['hcp_']
    list_ebdd_UKB = ['ukb_', 'ukbi']

    for dir in os.listdir(path):
        if dir[0:4].lower() in list_ebdd_ACCP:
            check_value_ACCP = True
        if dir[0:4].lower() in list_ebdd_HCP:
            check_value_HCP = True
        if dir[0:4].lower() in list_ebdd_UKB:
            check_value_UKB = True

    if not check_value_ACCP:
        print('ACCP embeddings are missing')
    if not check_value_HCP:
        print('HCP embeddings are missing')
    if not check_value_UKB:
        print('UKB embeddings are missing')
        
    return check_value_ACCP & check_value_HCP & check_value_UKB


# où trouver le nombre d'epoques ? Est-ce dans le fichier .yaml ?   
# cette fonction est provisoire
def get_epochs(path):
    '''
    Parameters
    ----------
    path: str
        path which leads to the directories of the embeddings for one given model
    
    Return
    ------
    epochs: set of int
        set of the epoch for which we can access to the model weights
    '''        

    epochs = []
    for dir in os.listdir(path):
        if dir.endswith('embeddings'):
            num = ''
            for c in dir[-14:-10]:
                if c.isdigit():
                    num = num + c
            if num != '':
                epochs.append(int(num))
    
    return set(epochs)

def loader(path, epoch, datasets=['ACCP', 'HCP', 'UKB']):
    '''
    Parameters
    ----------
    path: str
        path which leads to the directories of the embeddings for one given model
    
    Return
    ------
    epochs: set of int
        set of the epoch for which we can access to the model weights
    '''
    embeddings_ACCP, embeddings_HCP, embeddings_UKB = None, None, None

    if 'ACCP' in datasets:
        if os.path.isfile(f"{path}ACCP_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}ACCP_random_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}acc_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}acc_random_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}accp_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}accp_random_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}ACC_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}ACC_random_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}ACCP_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}ACCP_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}acc_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}acc_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}accp_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}accp_epoch{epoch}_embeddings/full_embeddings.csv")
        elif os.path.isfile(f"{path}ACC_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_ACCP = pd.read_csv(f"{path}ACC_epoch{epoch}_embeddings/full_embeddings.csv")
        else :
            return print(f'ACCP embeddings for epoch {epoch} not found at {path}')

    if 'HCP' in datasets: 
        if os.path.isfile(f"{path}HCP_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_HCP = pd.read_csv(f"{path}HCP_random_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}hcp_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_HCP = pd.read_csv(f"{path}hcp_random_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}HCP_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_HCP = pd.read_csv(f"{path}HCP_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}hcp_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_HCP = pd.read_csv(f"{path}hcp_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        else :
            return print(f'HCP embeddings for epoch {epoch} not found at {path}')

    if 'UKB' in datasets:
        if os.path.isfile(f"{path}UKB_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_UKB = pd.read_csv(f"{path}UKB_random_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}ukb_random_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_UKB = pd.read_csv(f"{path}ukb_random_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}UKB_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_UKB = pd.read_csv(f"{path}UKB_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        elif os.path.isfile(f"{path}ukb_epoch{epoch}_embeddings/full_embeddings.csv"):
            embeddings_UKB = pd.read_csv(f"{path}ukb_epoch{epoch}_embeddings/full_embeddings.csv", index_col=0)
        else :
            return print(f'UKB embeddings for epoch {epoch} not found at {path}')

    return embeddings_ACCP, embeddings_HCP, embeddings_UKB

# %% [markdown]
# Labelization of ACCP

# %%
def encoder(df, columns):
    for col in columns:
        code = {'present':1,
                'absent':0}
        df[col] = df[col].map(code)
    return df

labels_ACCP = pd.read_csv("/neurospin/dico/data/deep_folding/current/datasets/ACCpatterns/subjects_labels.csv")
labels_ACCP = labels_ACCP[['long_name','Left_PCS', 'Right_PCS']]

encoder(labels_ACCP, ['Left_PCS', 'Right_PCS']) 
labels_ACCP['Asymmetry'] = abs(labels_ACCP.Left_PCS - labels_ACCP.Right_PCS)
labels_ACCP['Two_PCS'] = labels_ACCP.Left_PCS & labels_ACCP.Right_PCS
labels_ACCP['Zero_PCS'] = (1-labels_ACCP.Left_PCS) & (1-labels_ACCP.Right_PCS)
labels_ACCP['Left_without_Right_PCS'] = (labels_ACCP.Left_PCS) & (1-labels_ACCP.Right_PCS)
labels_ACCP['Right_without_Left_PCS'] = (1-labels_ACCP.Left_PCS) & (labels_ACCP.Right_PCS)

# %% [markdown]
# #### Scale the embeddings (fit on UKB)

# %%
list_to_drop = ['Asymmetry','Left_PCS','Right_PCS','Two_PCS','Zero_PCS','Right_without_Left_PCS','Left_without_Right_PCS']

def chose_target(target, embeddings_ACCP, labels_ACCP):
    
    ebdd_lbl_ACCP = embeddings_ACCP.set_index('ID').join(labels_ACCP.set_index('long_name'))
    X = ebdd_lbl_ACCP.drop(list_to_drop, axis=1)
    y = ebdd_lbl_ACCP[target]
    return ebdd_lbl_ACCP, X, y

def scale_based_on_UKB(embeddings_ACCP, embeddings_HCP, embeddings_UKB):

    scaler = StandardScaler()
    scaler.fit(embeddings_UKB.to_numpy())
    scl_bdd_accp = scaler.transform(embeddings_ACCP.to_numpy())
    scl_bdd_hcp = scaler.transform(embeddings_HCP.to_numpy())
    scl_bdd_ukb = scaler.transform(embeddings_UKB.to_numpy())

    return scl_bdd_accp, scl_bdd_hcp, scl_bdd_ukb, scaler

# %% [markdown]
# #### Classify the embeddings (fit on ACCP)

# %%
def classifier():
    
    model = SVC(kernel='linear', probability=True,
                random_state=42,
                C=0.01, class_weight='balanced', decision_function_shape='ovr')

    return model

# %% [markdown]
# #### Compare the predicted labels (on HCP)

# %%
from functools import reduce

def get_corr(pred_dic):

    # Merge all dataframes on 'IID'
    merged_df = reduce(lambda left, right: pd.merge(left, right, on='IID'), pred_dic.values())

    # Calculate the correlation matrix
    correlation_matrix = merged_df.drop('IID', axis=1).corr()

    return correlation_matrix

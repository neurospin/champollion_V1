from soma import aims
import numpy as np
import pickle 
import pandas as pd

#data = pd.read_pickle('/neurospin/dico/data/deep_folding/current/datasets/hcp/crops/2mm/S.C.-sylv./mask/Rskeleton.pkl')
#data = np.load('/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40/crops/2mm/S.C.-sylv./mask/Rskeleton.npy')

#print(data.shape)

out_path = "/neurospin/dico/cmendoza/Runs/01_betavae_sulci_crops/Output/2025-06-05/17-32-49/"

for vol in ['input','output']:
    vol_npy = np.load(out_path+vol+'.npy')[0,0,:,:,:].astype(np.float32)
    print('Volume shape',vol_npy.shape)
    vol_nifty = aims.Volume(vol_npy)
    aims.write(vol_nifty, out_path+vol+'.nii.gz')

for sub in ['1000021','1000325','1000458']:
    for vol in ['_input','_output']:
        vol_npy = np.load(out_path+'subjects/'+sub+vol+'.npy').astype(np.float32)
        print('Volume shape',vol_npy.shape)
        vol_nifty = aims.Volume(vol_npy)
        aims.write(vol_nifty, out_path+'/subjects/'+sub+vol+'.nii.gz')
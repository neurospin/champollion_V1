# %%
import json
import numpy as np
import os
import subprocess
import glob
from soma import aims, aimsalgo
import scipy
from scipy import ndimage
import seaborn as sns
import pandas as pd

from sulci.registration.spam import spam_register

import anatomist.api as ana
from soma.qt_gui.qtThread import QtThreadCall
from soma.qt_gui.qt_backend import Qt

from soma.aimsalgo import MorphoGreyLevel_S16

from p_tqdm import p_map

# Global static variables
_AIMS_BINARY_ONE = 32767
_threshold_for_spam = 10
_dilation = 5
_threshold = 0
_dilation_final = 5
_threshold_final = 0
_parallel = True

import matplotlib
from matplotlib import pyplot as plt

from p_tqdm import p_map

# Just for the volume dimensions
spam_file = '/neurospin/dico/data/deep_folding/current/mask/2mm/regions/L/Sc.Cal.-S.Li._left.nii.gz'
spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")


def copy_volume_info(vol, dtype='S16'):
    vol_copy = aims.Volume(vol.getSize(), dtype)
    vol_copy.copyHeaderFrom(vol.header())
    vol_copy.np[:] = 0 
    return vol_copy 


before_all = copy_volume_info(spam_vol, 'S16')
after_all = copy_volume_info(spam_vol, 'S16')


list_after_crop_files = glob.glob("/tmp/after_cropped*.nii.gz")

print(list_after_crop_files[:5])

subject_l = [l.split("after_cropped")[-1].split('.nii.gz')[0] for l in list_after_crop_files]

list_after_cropped = [aims.read(after_crop_file) for after_crop_file in list_after_crop_files]

print("Crops successfully read")

for sub_name in subject_l:
    before_path = f"/tmp/before{sub_name}.nii.gz"
    after_path = f"/tmp/after{sub_name}.nii.gz"
    before = aims.read(before_path)
    after = aims.read(after_path)
    before_all.np[:] += (before.np > 0).astype(np.int16)
    after_all.np[:] += (after.np > 0).astype(np.int16)

# %%
diff_all = after_all - before_all

# %%
print("diff_all unique: ", np.unique(diff_all.np))

# %%
# Reads initial spam volume
spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
# spam_vol.np[spam_vol.np < _threshold_for_spam] = 0


# %%
# Visualization
# spam = a.toAObject(spam_vol)
# before_a = a.toAObject(before_all)
# diff_a = a.toAObject(diff_all)
# before_a.setPalette("Blues")
# diff_a.setPalette("bwr")
# # spam_before = a.toAObject(before_all)
# # spam_before.setPalette("Blues")
# # spam_after = a.toAObject(after_all)Utilities to generate foldlabels from graph
# # spam_after.setPalette("Reds")
# w = a.createWindow('Sagittal')
# w.addObjects(spam)
# # w.addObjects(before_a)
# w.addObjects(diff_a)
# # w.addObjects(spam_before)
# # w.addObjects(spam_after)

# crop_a = a.toAObject(after_cropped)
# w_crop = a.createWindow('Sagittal')
# w_crop.addObjects(crop_a)


# %%
# spam = a.loadObject(spam_file)
# spam.setPalette("Blues")
# list_after_a = [a.toAObject(after) for after in list_after]
# for after in list_after_a:
#     after.setPalette("RED-lfusion")
# w = a.createWindow('Sagittal')
# w.addObjects(list_after_a)
# w.addObjects(spam)

# %%
print("Crop shapes: ")
for l in list_after_cropped[:5]:
    print(l.shape, np.unique(l.np))

# %%

# # Reads initial spam volume
# spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")

# df_dict = {"spam": spam_vol.np[spam_vol.np>0].flatten(),
#            "diff": diff_all.np[spam_vol.np>0].flatten()}
# df = pd.DataFrame(df_dict)
# g = sns.lmplot(data=df, x="spam", y="diff")

# def annotate(data, **kws):
#     r, p = scipy.stats.pearsonr(data['spam'], data['diff'])
#     ax = plt.gca()
#     ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
#             transform=ax.transAxes)
    
# g.map_dataframe(annotate)

# %%

# # Reads initial spam volume
# spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")

# df_dict = {"spam": spam_vol.np[diff_all.np!=0].flatten(),
#            "diff": diff_all.np[diff_all.np!=0].flatten()}
# df = pd.DataFrame(df_dict)
# g = sns.lmplot(data=df, x="spam", y="diff")

# def annotate(data, **kws):
#     r, p = scipy.stats.pearsonr(data['spam'], data['diff'])
#     ax = plt.gca()
#     ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
#             transform=ax.transAxes)
    
# g.map_dataframe(annotate)

# # %%

# # Reads initial spam volume
# spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")

# df_dict = {"spam": spam_vol.np[spam_vol.np>_threshold_for_spam].flatten(),
#            "diff": diff_all.np[spam_vol.np>_threshold_for_spam].flatten()}
# df = pd.DataFrame(df_dict)
# g = sns.lmplot(data=df, x="spam", y="diff")

# def annotate(data, **kws):
#     r, p = scipy.stats.pearsonr(data['spam'], data['diff'])
#     ax = plt.gca()
#     ax.text(.05, .9, 'r={:.2f}, p={:.2g}'.format(r, p),
#             transform=ax.transAxes)
    
# g.map_dataframe(annotate)

# %%

# plt.hist(diff_all.np[(spam_vol.np==0) & (diff_all.np != 0)].flatten(), bins=30)

# # %%

# plt.hist(diff_all.np[(spam_vol.np<_threshold_for_spam) & (diff_all.np != 0)].flatten(), bins=30)

# %%
print(diff_all.np[(spam_vol.np==0) & (diff_all.np != 0)].flatten().mean())
print(diff_all.np[(spam_vol.np==0) & (diff_all.np != 0)].flatten().std())

# %%
print(diff_all.np[(spam_vol.np<_threshold_for_spam) & (diff_all.np != 0)].flatten().mean())
print(diff_all.np[(spam_vol.np<_threshold_for_spam) & (diff_all.np != 0)].flatten().std())

# %%



# %%
subject_df = pd.DataFrame({"Subject":subject_l})
subject_df.to_csv("/tmp/Lskeleton_subject.csv", index=False)
subject_df.head()


# %%
after_cropped_npy = np.stack(list_after_cropped, axis=0)
np.save("/tmp/Lskeleton.npy", after_cropped_npy)

aims.write(diff_all, f"/tmp/diff_all.nii.gz")




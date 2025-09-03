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

# a = ana.Anatomist()



# %% [markdown]
# 

# %%


# %%
def dilate(mask, radius=_dilation):
    """Makes a dilation radius _dilation, in mm
    """
    arr = mask.np
    # Binarization of mask
    arr[arr < 1] = 0
    if radius > 0:
        arr[arr >= 1] = _AIMS_BINARY_ONE
        # Dilates initial volume of 10 mm
        morpho = MorphoGreyLevel_S16()
        dilate = morpho.doDilation(mask, radius)
        arr_dilate = dilate.np
        arr_dilate[arr_dilate >= 1] = 1
        return dilate
    else:
        arr[arr >= 1] = 1
        return mask

# %%
spam_file = '/neurospin/dico/data/deep_folding/current/mask/2mm/regions/R/S.C.-sylv._right.nii.gz'
skel_path = '/neurospin/dico/data/deep_folding/current/datasets/hcp/skeletons/2mm/R'
skel_files = glob.glob(f'{skel_path}/*.nii.gz')
skel_files[:5]

# %%
# Reads initial spam volume
spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
spam_vol.np[spam_vol.np < _threshold_for_spam] = 0

# Reads initial spam volume
spam_vol_untouched = aims.read(spam_file, dtype="Volume_FLOAT")

# %%
# Makes gaussian filter
# spam_vol_for_gaussian = aims.read(spam_file, dtype="Volume_FLOAT")
# spam_vol_for_gaussian.np[spam_vol_for_gaussian.np < _threshold_for_spam] = 0.
# spam_vol_for_gaussian.np[spam_vol_for_gaussian.np < _threshold_for_spam] = 1.
# g = aimsalgo.Gaussian3DSmoothing_FLOAT(10., 10., 10.)
# spam_vol_for_gaussian = g.doit(spam_vol_for_gaussian)
# spam_vol_for_gaussian.np[:] = spam_vol_for_gaussian.np / spam_vol_for_gaussian.max()

# # Applies filter to spam_vol
# spam_vol.np[:] = spam_vol.np * spam_vol_for_gaussian.np

# %%
print("Unique of spam_vol = ", np.unique(spam_vol.np))

# %%
def compute_bbox_mask(arr):

    # Gets location of bounding box as slices
    objects_in_image = ndimage.find_objects(arr)
    print(f"ndimage.find_objects(arr) = {objects_in_image}")
    if not objects_in_image:
        raise ValueError("There are only 0s in array!!!")

    loc = objects_in_image[0]
    bbmin = []
    bbmax = []

    for slicing in loc:
        bbmin.append(slicing.start)
        bbmax.append(slicing.stop)

    return np.array(bbmin), np.array(bbmax)

# %%
def copy_volume_info(vol, dtype='S16'):
    vol_copy = aims.Volume(vol.getSize(), dtype)
    vol_copy.copyHeaderFrom(vol.header())
    vol_copy.np[:] = 0 
    return vol_copy 

# %%
def do_masking_dilation(spam_vol, skel_vol, dilation, threshold, do_binarization):
   
    spam_vol = aims.Volume_FLOAT(spam_vol)
    skel_vol = aims.Volume_S16(skel_vol)
    
    # Do binarization for registration
    if do_binarization:
        skel_vol.np[:] = (skel_vol.np > 0).astype(np.int16)

    # Makes binarization and dilation on spam
    mask_result = copy_volume_info(spam_vol, 'S16')
    mask_result.np[:] = spam_vol.np
    
    # # Filter mask with Gaussian filter
    # arr_filter = scipy.ndimage.gaussian_filter(
    #     mask_result.np.astype(float),
    #     sigma=0.5,
    #     order=0,
    #     output=None,
    #     mode='reflect',
    #     truncate=4.0)
    # mask_result.np[:] = (arr_filter > 0.001).astype(int)
    
    # Threshold mask
    print("before threshold", np.unique(mask_result.np, return_counts=True)) 
    mask_result.np[mask_result.np <= threshold] = 0.
    
    # Dilates mask
    mask_result.np[:] = dilate(mask_result, dilation).np
    print("after threshold and dilation", np.unique(mask_result.np, return_counts=True))
    
    # Do the actual masking
    skel_vol.np[mask_result.np <= 0] = 0
    
    return skel_vol, mask_result

# %%
def realign(spam_vol: aims.Volume_FLOAT, skel_vol: aims.Volume_S16, sub_name: str):
    """Realigns skeleton mask to spam
    
    skel_f is a file name of skeleton file"""
    
    spam_vol = aims.Volume_FLOAT(spam_vol)
    skel_vol = aims.Volume_S16(skel_vol)
    
    # Masks with first dilation and threshold
    skel_vol_before, mask_dilated = do_masking_dilation(spam_vol_untouched, skel_vol, _dilation, _threshold, True)
    aims.write(skel_vol_before, f"/tmp/skel_before{sub_name}.nii.gz")
    
    # Makes realignment
    out_tr = spam_register(spam_vol,
                        skel_vol_before,
                        do_mask=False,
                        R_angle_var=np.pi / 8,
                        t_var=5.,
                        verbose=False,
                        in_log=False,
                        calibrate_distrib=30)
    aims.write(out_tr, f'/tmp/transform{sub_name}.trm')
    print(out_tr.np)
    
    # Masks with final dilation and threshold
    skel_vol, spam_vol = do_masking_dilation(spam_vol_untouched, skel_vol, _dilation_final, _threshold_final, False)
    aims.write(skel_vol, f"/tmp/skel_final_before{sub_name}.nii.gz")
    aims.write(spam_vol, f"/tmp/spam_final_before{sub_name}.nii.gz")
    
    # Applies the realignment
    subprocess.check_call(f"AimsApplyTransform -i /tmp/skel_final_before{sub_name}.nii.gz -o /tmp/skel_final_realigned{sub_name}.nii.gz -m /tmp/transform{sub_name}.trm -t 0", shell=True)
    subprocess.check_call(f"AimsApplyTransform -i /tmp/spam_final_before{sub_name}.nii.gz -o /tmp/spam_final_realigned{sub_name}.nii.gz -m /tmp/transform{sub_name}.trm", shell=True)
    
    # loads realigned file:
    before = aims.read(f"/tmp/skel_final_before{sub_name}.nii.gz")
    after = aims.read(f"/tmp/skel_final_realigned{sub_name}.nii.gz")
    spam_after = aims.read(f"/tmp/spam_final_realigned{sub_name}.nii.gz")
    
    return before, after, spam_after, mask_dilated

# %%
before_all = copy_volume_info(spam_vol, 'S16')
after_all = copy_volume_info(spam_vol, 'S16')
list_after_cropped = []
subject_l = []
all_l = []

def realign_one_subject(skel_f):
    global before_all, after_all, spam_vol
    # Gets skeleton and subject name
    print(skel_f)
    sub_name = skel_f.split('_')[-1].split('.')[0]
    skel_vol = aims.read(skel_f)
    
    # Realign
    b, r, s, mask_dilated = realign(spam_vol, skel_vol, sub_name)
    
    # Gets volume before and after
    before = copy_volume_info(spam_vol, 'S16')
    before += b
    after = copy_volume_info(spam_vol, 'S16')
    after += r
    spam_after = copy_volume_info(spam_vol, 'S16')
    spam_after += s  
    
    before_path = f"/tmp/before{sub_name}.nii.gz"
    after_path = f"/tmp/after{sub_name}.nii.gz"
    aims.write(before, before_path)
    aims.write(after, after_path)
    
    # Crop volume to mask_dilated size
    bbmin, bbmax = compute_bbox_mask(mask_dilated)
    after_cropped = aims.VolumeView(after, bbmin, bbmax - bbmin)
    aims.write(after_cropped, f"/tmp/after_cropped{sub_name}.nii.gz")
    
    return (sub_name, after_cropped, before_path, after_path)

chosen_skel_files = skel_files

if _parallel:
    all_l = p_map(realign_one_subject, chosen_skel_files, num_cpus=20)
else:
    for skel_f in chosen_skel_files:
        all_l.append(realign_one_subject(skel_f))
    
for sub_name, after_cropped, before_path, after_path in all_l:
    subject_l.append(sub_name)
    list_after_cropped.append(after_cropped)
    # Adds volume to concatenate volume
    before = aims.read(before_path)
    after = aims.read(after_path)
    before_all.np[:] += (before.np > 0).astype(np.int16)
    after_all.np[:] += (after.np > 0).astype(np.int16)

# %%
diff_all = after_all - before_all

# %%
np.unique(diff_all.np)

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
for l in list_after_cropped:
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
subject_df.to_csv("/tmp/Rskeleton_subject.csv", index=False)
subject_df.head()


# %%
after_cropped_npy = np.stack(list_after_cropped, axis=0)
np.save("/tmp/Rskeleton.npy", after_cropped_npy)

aims.write(diff_all, f"/tmp/diff_all.nii.gz")




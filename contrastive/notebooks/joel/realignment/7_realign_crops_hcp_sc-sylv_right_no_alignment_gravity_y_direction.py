#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.
""" Realigns skeleton images on local sulcal SPAM

"""
#########################################
# Imports
#########################################
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

import matplotlib
from matplotlib import pyplot as plt

from p_tqdm import p_map

#########################################
# Global static variables
#########################################

_AIMS_BINARY_ONE = 32767
_threshold_for_spam = 0
_dilation = 0
_threshold = 0
_dilation_final = 0
_threshold_final = 0
_parallel = True
# Output path
_mm_skeleton_path = "/neurospin/dico/jchavas/Runs/70_self-supervised_two-regions/Output/realigned/SCsylv-right/7_no_alignment"

#########################################
# Used paths
#########################################
spam_file = '/neurospin/dico/data/deep_folding/current/mask/2mm/regions/R/S.C.-sylv._right.nii.gz'
skel_path = '/neurospin/dico/data/deep_folding/current/datasets/hcp/skeletons/2mm/R'
skel_files = glob.glob(f'{skel_path}/*.nii.gz')
print("First target skeleton files:\n", skel_files[:5])
    
path = "/neurospin/dico/data/deep_folding/current/models/Champollion_V0_trained_on_UKB40/SC-sylv_right"
lab = "gravityCenter_y"
model_path = glob.glob(f"{path}/*")[0]
embeddings_file = f"{model_path}/hcp_random_epoch80_embeddings/full_embeddings.csv"
participants_file = "/neurospin/dico/data/human/hcp/derivatives/morphologist-2023/morphometry/spam_recognition/morpho_talairach/morpho_S.C._right.dat"

side = "R" # "R" or "L"
region = "S.C.-sylv." # "ORBITAL", "CINGULATE", "SC-sylv", "F.I.P."
database = 'hcp'

#########################################
# Function definitions
#########################################

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


def copy_volume_info(vol, dtype='S16'):
    vol_copy = aims.Volume(vol.getSize(), dtype)
    vol_copy.copyHeaderFrom(vol.header())
    vol_copy.np[:] = 0 
    return vol_copy 


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
    print("after threshold and dilation",
          np.unique(mask_result.np, return_counts=True))
    
    # Do the actual masking
    skel_vol.np[mask_result.np <= 0] = 0
    
    return skel_vol, mask_result


def realign(spam_vol: aims.Volume_FLOAT, skel_vol: aims.Volume_S16, sub_name: str):
    """Realigns skeleton mask to spam
    
    skel_f is a file name of skeleton file"""
    
    spam_vol = aims.Volume_FLOAT(spam_vol)
    skel_vol = aims.Volume_S16(skel_vol)
    
    # Masks with first dilation and threshold
    skel_vol_before, mask_dilated = do_masking_dilation(spam_vol_untouched, skel_vol, _dilation, _threshold, True)
    aims.write(skel_vol_before, f"/tmp/skel_before{sub_name}.nii.gz")
    
    # Makes realignment
    # out_tr = spam_register(spam_vol,
    #                     skel_vol_before,
    #                     do_mask=False,
    #                     R_angle_var=np.pi / 8,
    #                     t_var=5.,
    #                     verbose=False,
    #                     in_log=False,
    #                     calibrate_distrib=30)
    out_tr = aims.AffineTransformation3d(aims.Quaternion([0, 0, 0, 1]))
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
    
    os.makedirs(f"{_mm_skeleton_path}", exist_ok=True)
    aims.write(after_cropped,
               f"{_mm_skeleton_path}/after_cropped{sub_name}.nii.gz")
    
    return (sub_name, after_cropped, before_path, after_path)


#########################################
# Main
#########################################
if __name__ == "__main__":

    # Reads initial spam volume
    spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
    spam_vol.np[spam_vol.np < _threshold_for_spam] = 0

    # Reads initial spam volume
    spam_vol_untouched = aims.read(spam_file, dtype="Volume_FLOAT")


    print("Unique of spam_vol = ", np.unique(spam_vol.np))


    before_all = copy_volume_info(spam_vol, 'S16')
    after_all = copy_volume_info(spam_vol, 'S16')
    list_after_cropped = []
    subject_l = []
    all_l = []

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

    diff_all = after_all - before_all

    print(np.unique(diff_all.np))

    # Reads initial spam volume
    spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
    # spam_vol.np[spam_vol.np < _threshold_for_spam] = 0

    for l in list_after_cropped:
        print(l.shape, np.unique(l.np))


    print(diff_all.np[(spam_vol.np==0) & (diff_all.np != 0)].flatten().mean())
    print(diff_all.np[(spam_vol.np==0) & (diff_all.np != 0)].flatten().std())

    print(diff_all.np[(spam_vol.np<_threshold_for_spam) & (diff_all.np != 0)].flatten().mean())
    print(diff_all.np[(spam_vol.np<_threshold_for_spam) & (diff_all.np != 0)].flatten().std())

    subject_df = pd.DataFrame({"Subject":subject_l})
    subject_df.to_csv("/tmp/Rskeleton_subject.csv", index=False)
    subject_df.head()

    after_cropped_npy = np.stack(list_after_cropped, axis=0)
    np.save("/tmp/Rskeleton.npy", after_cropped_npy)

    aims.write(diff_all, f"/tmp/diff_all.nii.gz")

    # #########################################
    # # Launch anatomist
    # #########################################
    # a = ana.Anatomist()

    # #########################################
    # # Prepare for visualization of averages
    # #########################################
    # participants = pd.read_csv(participants_file, sep=' ', index_col=0)
    # participants = participants[[lab]].dropna()
    # print(participants.head())
    
    # emb = pd.read_csv(f"{embeddings_file}", index_col=0)
    # merged = participants[[lab]].merge(emb, left_index=True, right_index=True)
    # label = merged.iloc[:,0:1]
    # print(label.head())
    
    

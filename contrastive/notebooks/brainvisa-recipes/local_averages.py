#!python
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

"""

Exemple of command:

bv python3 Anatomist_direction_visu.py 
              CINGULATE. \
              R \
              Sorted_projection/regression_on_latent/rs3020595_G.csv \
              6
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd

import anatomist.api as ana
# from soma.qt_gui.qtThread import QtThreadCall
from os.path import basename
from soma.qt_gui.qt_backend import Qt
from soma import aims, aimsalgo

from scipy import ndimage
from sulci.registration.spam import spam_register

from soma.aimsalgo import MorphoGreyLevel_S16

from p_tqdm import p_map

# Global static variables
_AIMS_BINARY_ONE = 32767
_dilation = 5
_threshold = 0
_dilation_final = 5
_threshold_final = 0
_edge_smoothing = 10.

a = ana.Anatomist()


def to_bucket(obj):
    """Converts an object to a bucket if it isn't one already."""
    if obj.type() == obj.BUCKET:
        return obj
    avol = a.toAimsObject(obj)
    c = aims.Converter(intype=avol, outtype=aims.BucketMap_VOID)
    abck = c(avol)
    bck = a.toAObject(abck)
    bck.releaseAppRef()
    return bck


def build_gradient(pal):
    """Builds a gradient palette."""
    gw = ana.cpp.GradientWidget(
        None, 'gradientwidget',
        pal.header()['palette_gradients'])
    gw.setHasAlpha(True)
    nc = pal.shape[0]
    rgbp = gw.fillGradient(nc, True)
    rgb = rgbp.data()
    npal = pal.np['v']
    pb = np.frombuffer(rgb, dtype=np.uint8).reshape((nc, 4))
    npal[:, 0, 0, 0, :] = pb
    npal[:, 0, 0, 0, :3] = npal[:, 0, 0, 0,
                                :3][:, ::-1]  # Convert BGRA to RGBA
    pal.update()


def buckets_average(subject_id_list, dataset_name_list, region, side):
    """Computes the average bucket volumes for a list of subjects."""
    dic_vol = {}
    dim = 0
    rep = 0

    if len(subject_id_list) == 0:
        return False

    # Find a valid volume for dimension checking
    while dim == 0 and rep < len(subject_id_list):
        dataset_name = dataset_name_list[rep]
        #dataset = 'UkBioBank' if dataset_name_list[rep].lower(
        #) in ['ukb', 'ukbiobank', 'projected_ukb'] else 'UkBioBank40'
        mm_skeleton_path = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset_name}/crops/2mm/{region}/mask/{side}crops"

        file_path = f"{mm_skeleton_path}/{subject_id_list[rep]}_cropped_skeleton.nii.gz"
        if os.path.isfile(file_path):
            sum_vol = aims.read(file_path).astype(float)
            dim = sum_vol.shape
            sum_vol.fill(0)
        else:
            print(f'FileNotFound: {file_path}')
        rep += 1

    # Process each subject
    for subject_id, dataset_name in zip(subject_id_list, dataset_name_list):
        #dataset = 'UkBioBank' if dataset_name.lower(
        #) in ['ukb', 'ukbiobank', 'projected_ukb'] else 'UkBioBank40'
        mm_skeleton_path = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset_name}/crops/2mm/{region}/mask/{side}crops"

        file_path = f"{mm_skeleton_path}/{subject_id}_cropped_skeleton.nii.gz"
        if os.path.isfile(file_path):
            vol = aims.read(file_path)
            if vol.np.shape != dim:
                raise ValueError(
                    f"{subject_id_list[0]} and {subject_id} "
                    "must have the same dimensions")

            # Convert to binary structure
            struc3D = (vol.np > 0).astype(int)
            dic_vol[subject_id] = struc3D
            # Accumulate binary volumes
            sum_vol.np[:] += struc3D
        else:
            print(f'FileNotFound: {file_path}')

    # Normalize the accumulated volume
    sum_vol.np[:] /= len(subject_id_list)
    print(f"{sum_vol.shape}: max = {sum_vol.np.max()}")
    return sum_vol


def copy_volume_info(vol, dtype='S16'):
    vol_copy = aims.Volume(vol.getSize(), dtype)
    vol_copy.copyHeaderFrom(vol.header())
    vol_copy.np[:] = 0
    return vol_copy


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


def do_masking_dilation(spam_vol, skel_vol,
                        dilation, threshold, do_binarization):

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
    # print("before threshold", np.unique(mask_result.np, return_counts=True))
    mask_result.np[mask_result.np <= threshold] = 0
    mask_result.np[mask_result.np > threshold] = 1

    # Dilates mask
    mask_result.np[:] = dilate(mask_result, dilation).np
    # print("after threshold and dilation",
    #       np.unique(mask_result.np, return_counts=True))

    # Do the actual masking
    skel_vol.np[mask_result.np <= 0] = 0

    return skel_vol, mask_result


def realign(spam_vol: aims.Volume_FLOAT, skel_vol: aims.Volume_S16,
            do_edge_smoothing: bool,
            sub_name: str):
    """Realigns skeleton mask to spam

    skel_f is a file name of skeleton file"""

    spam_vol = aims.Volume_FLOAT(spam_vol)
    spam_vol_untouched = aims.Volume_FLOAT(spam_vol)
    skel_vol = aims.Volume_S16(skel_vol)

    # Masks with first dilation and threshold
    skel_vol_before, mask_dilated = do_masking_dilation(
        spam_vol, skel_vol, _dilation, _threshold, True)
    aims.write(skel_vol_before, f"/tmp/skel_before{sub_name}.nii.gz")
    # print(np.unique(spam_vol.np))
    mask_dilated.np[:] = (mask_dilated.np > 0).astype(np.int16)

    if do_edge_smoothing:
        g = aimsalgo.Gaussian3DSmoothing_FLOAT(
            _edge_smoothing, _edge_smoothing, _edge_smoothing)
        mask_vol = copy_volume_info(mask_dilated, 'FLOAT')
        mask_vol.np[:] = mask_dilated.np
        mask_vol = g.doit(mask_vol)
        mask_vol.np[:] = mask_vol.np / mask_vol.max()
        spam_vol.np[:] = spam_vol.np * spam_vol.np * mask_vol.np

    # Makes realignment
    out_tr = spam_register(spam_vol,
                           skel_vol_before,
                           do_mask=False,
                           R_angle_var=np.pi / 8,
                           t_var=10.,
                           verbose=False,
                           in_log=False,
                           calibrate_distrib=30)
    aims.write(out_tr, f'/tmp/transform{sub_name}.trm')
    # print(out_tr.np)

    # Masks with final dilation and threshold
    skel_vol, spam_vol = do_masking_dilation(
        spam_vol_untouched, skel_vol, _dilation_final, _threshold_final, True)
    aims.write(skel_vol, f"/tmp/skel_final_before{sub_name}.nii.gz")
    aims.write(spam_vol, f"/tmp/spam_final_before{sub_name}.nii.gz")

    # Applies the realignment
    subprocess.check_call(
        f"AimsApplyTransform -i /tmp/skel_final_before{sub_name}.nii.gz -o /tmp/skel_final_realigned{sub_name}.nii.gz -m /tmp/transform{sub_name}.trm -t 0", shell=True)
    subprocess.check_call(
        f"AimsApplyTransform -i /tmp/spam_final_before{sub_name}.nii.gz -o /tmp/spam_final_realigned{sub_name}.nii.gz -m /tmp/transform{sub_name}.trm", shell=True)

    # loads realigned file:
    before = aims.read(f"/tmp/skel_final_before{sub_name}.nii.gz")
    after = aims.read(f"/tmp/skel_final_realigned{sub_name}.nii.gz")
    spam_after = aims.read(f"/tmp/spam_final_realigned{sub_name}.nii.gz")

    return before, after, spam_after, mask_dilated


def realign_one_subject(skel_f, spam_vol):
    # Gets skeleton and subject name
    print(skel_f)
    sub_name = skel_f.split('_')[-1].split('.')[0]
    skel_vol = aims.read(skel_f)

    # Realign
    b, r, s, mask_dilated = realign(spam_vol, skel_vol, True, sub_name)

    # Gets volume before and after
    before = copy_volume_info(spam_vol, 'S16')
    before += b
    after = copy_volume_info(spam_vol, 'S16')
    after += r
    spam_after = copy_volume_info(spam_vol, 'S16')
    spam_after += s

    # Crop volume to mask_dilated size
    bbmin, bbmax = compute_bbox_mask(mask_dilated)
    after_cropped = aims.VolumeView(after, bbmin, bbmax - bbmin)

    return (sub_name, after_cropped, before, after)


def buckets_average_with_alignment(subject_id_list, dataset_name_list,
                                   region, side, nb_processors):
    """Computes the average bucket volumes for a list of subjects."""
    dic_vol = {}
    dim = 0
    rep = 0

    if len(subject_id_list) == 0:
        return False

    side_long = "left" if side == 'L' else 'right'
    
    spam_file = f'/neurospin/dico/data/deep_folding/current/mask/2mm/regions/{side}/{region}_{side_long}.nii.gz'
    spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
    
    before_all = copy_volume_info(spam_vol, 'S16')
    after_all = copy_volume_info(spam_vol, 'S16')
    list_after_cropped = []
    subject_l = []
    all_l = []
    
    # Find a valid volume for dimension checking
    sum_vol = aims.read(spam_file, dtype="Volume_FLOAT")
    dim = sum_vol.shape
    sum_vol.fill(0)  

    dataset_name = dataset_name_list[0]
    #dataset = 'UkBioBank' if dataset_name.lower(
    #            ) in ['ukb', 'ukbiobank', 'projected_ukb'] else 'UkBioBank40'
    skel_path = f"/neurospin/dico/data/deep_folding/current/datasets/{dataset_name}/skeletons/2mm/{side}"
        
    def realign_one(subject_id):
        file_path = f"{skel_path}/{side}resampled_skeleton_{subject_id}.nii.gz"
        if os.path.isfile(file_path):
            all_ = realign_one_subject(file_path, spam_vol)
            return all_
        else:
            print(f'FileNotFound: {file_path}')        
        
    # Process each subject
    if nb_processors > 1:
        all_l = p_map(realign_one, subject_id_list, num_cpus=nb_processors)
    else:
        for subject_id in subject_id_list:
            all_l.append(realign_one(subject_id))

    for sub_name, after_cropped, before, after in all_l:
        subject_l.append(sub_name)
        list_after_cropped.append(after_cropped)
        # Adds volume to concatenate volume
        before_all += before
        after_all += after
    
    # Normalize the accumulated volume
    sum_vol.np[:] = after_all.np.astype(float) / len(subject_id_list)
    print(f"{sum_vol.shape}, max = {sum_vol.np.max()}")
    return sum_vol


def parse_args(argv):
    """Function parsing command-line arguments
    Args:
        argv: a list containing command line arguments
    Returns:
        params: dictionary with keys
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(
        prog=basename(__file__),
        description="Process brain imaging projections with Anatomist.")
    parser.add_argument(
        "-r", "--region", type=str, default="F.Coll.-S.Rh.",
        help="Region of the brain to analyze (e.g., 'S.F.int.-F.C.M.ant.').")
    parser.add_argument(
        "-i", "--side", type=str, default="L",
        choices=["L", "R"], help="Side of the brain (left or right).")
    parser.add_argument(
        "-a", "--alignment", action="store_true",
        help="if present, performs alignment")
    parser.add_argument(
        "-b", "--nb_processors",  type=int, default=46,
        help="performs parallelization if > 1; gives number of processors.")
    parser.add_argument(
        "-d", "--dataset", type=str, default="UkBioBank40",
        help="Dataset on which are taken the crops: UkBioBank, UkBioBank40,...")
    parser.add_argument(
        "-p", "--phenotype_file", type=str,
        default="/neurospin/dico/data/deep_folding/current/models/Champollion_V0/UKB-RAP/FColl-SRh_left_predicted.csv",
        help="Path to the phenotype file.")
    parser.add_argument(
        "-s", "--subject_column", type=str, default="ID",
        help="Name of column ID")
    parser.add_argument(
        "-e", "--phenotype_column", type=str, default="predicted",
        help="Name of column phenotype")
    parser.add_argument(
        "-t", "--nb_subjects_per_average", type=int, default=200,
        help="Number of subjects per average.")
    parser.add_argument(
        "-n", "--nb_columns", type=int, default=3,
        help="Number of columns for the Anatomist windows block.")
    parser.add_argument(
        "-l", "--nb_lines", type=int, default=1,
        help="Number of lines for the Anatomist windows block.")

    args = parser.parse_args()
    params = vars(args)

    print("alignment = ", args.alignment)

    return params


def visualize_averages_along_sorted_phenotype(params):
    # anatomist objects
    global _block
    global _average_dic
    global _dic_packages
    global a

    ####################################
    # Reads params dictionary
    ####################################
    nb_lines = params['nb_lines']
    nb_columns = params['nb_columns']
    phenotype_file = params['phenotype_file']
    phenotype_column = params['phenotype_column']
    subject_column = params['subject_column']
    dataset = params['dataset']
    step = params['nb_subjects_per_average']
    region = params['region']
    side = params['side']
    alignment = params['alignment']
    nb_processors = params['nb_processors']

    ####################################
    # Initializations
    ####################################

    # Creates the block if it has not been created
    try:
        _block
    except NameError:
        _block = a.createWindowsBlock(nb_columns)

    _average_dic = {}
    _dic_packages = {}
    nb_windows = nb_lines * nb_columns

    ####################################
    # Dataframe manipulations
    ####################################

    # Load sorted projections (a .csv file)
    phenotype_df = pd.read_csv(phenotype_file)

    if subject_column == "IID":
        phenotype_df["IID"] = phenotype_df["IID"].apply(lambda x : "sub-"+str(x))

    phenotype_df = phenotype_df.sort_values(phenotype_column)
    phenotype_df = phenotype_df.set_index(subject_column)

    ####################################
    # Grouping subjects to average
    ####################################

    for i in range(0, len(phenotype_df), step):
        list_idx = phenotype_df.index[i:i + step].to_numpy()
        _dic_packages[i // step] = [f'{idx}' for idx in list_idx]

    # Ensures that last list contains the last step subjects
    list_idx = (phenotype_df.index[-step:].to_numpy())
    _dic_packages[i//step] = list_idx

    list_database = [dataset] * step
    n_pack = len(_dic_packages)

    # Process each package of subjects
    list_pack = [int(np.ceil(i*n_pack/float(nb_windows)))
                 for i in range(0, nb_windows)]
    list_pack[-1] = n_pack-1

    ####################################
    # Averaging for each group of subjects
    ####################################

    for i in list_pack:
        if alignment:
            sum_vol = buckets_average_with_alignment(_dic_packages[i], list_database,
                                                     region, side, nb_processors)
        else:
            sum_vol = buckets_average(_dic_packages[i], list_database,
                                      region, side)
        _average_dic[f'a_sum_vol{i}'] = a.toAObject(sum_vol)
        _average_dic[f'a_sum_vol{i}'].setPalette(minVal=0, absoluteMode=True)

        _average_dic[f'rvol{i}'] = a.fusionObjects(
            objects=[_average_dic[f'a_sum_vol{i}']],
            method='VolumeRenderingFusionMethod')
        _average_dic[f'rvol{i}'].releaseAppRef()

        # custom palette
        pal = a.createPalette('VR-palette')
        pal.header()['palette_gradients'] = '0;0.459574;0.497872;0.910638;1;1#0;0;0.52766;0.417021;1;1#0;0.7;1;0#0;0;0.0297872;0.00851064;0.587179;0.0666667;0.838462;0.333333;0.957447;0.808511;1;1'
        #0;0.459574;0.497872;0.910638;1;1#0;0;0.52766;0.417021;1;1#0;0.7;1;0#0;0;0.0297872;0.00851064;0.582051;0.133333;0.838462;0.333333;0.957447;0.808511;1;1
        
        build_gradient(pal)
        _average_dic[f'rvol{i}'].setPalette('VR-palette', minVal=0.05,
                                            maxVal=0.35, absoluteMode=True)
        pal2 = a.createPalette('slice-palette')
        pal2.header()['palette_gradients'] = '0;0.459574;0.497872;0.910638;1;1#0;0;0.52766;0.417021;1;1#0;0.7;1;0#0;0;0.0297872;0.00851064;0.587179;0.0666667;0.838462;0.333333;0.957447;0.808511;1;1'
        build_gradient(pal2)
        _average_dic[f'a_sum_vol{i}'].setPalette('slice-palette')

        # Create a 3D window and add the volume rendering object
        _average_dic[f'wvr{i}'] = a.createWindow('3D', block=_block)
        _average_dic[f'wvr{i}'].addObjects(_average_dic[f'rvol{i}'])


def main(argv):

    # Parsing arguments
    params = parse_args(argv)

    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication(sys.argv)

    visualize_averages_along_sorted_phenotype(params)

    app.exec_()  # Start the Qt event loop


if __name__ == "__main__":
    # This permits to call main also from another python program
    # without having to make system calls
    main(argv=sys.argv[1:])

import numpy as np
import os
from soma import aims

from sulci.registration.spam import spam_register

import anatomist.api as ana
from soma.qt_gui.qtThread import QtThreadCall
from soma.qt_gui.qt_backend import Qt

from soma.aimsalgo import MorphoGreyLevel_S16

runloop = False

# Global static variables
_AIMS_BINARY_ONE = 32767
_dilation = 5
_threshold = 0

spam_file = '/neurospin/dico/data/deep_folding/current/mask/2mm/regions/L/S.C.-sylv._left.nii.gz'
skel_f = '/neurospin/dico/data/deep_folding/current/datasets/pclean/binarized_skeletons/L/Lbinarized_skeleton_ammon.nii.gz' 

run_loop = Qt.QApplication.instance() is None

# launching anatomist
a = ana.Anatomist()


def dilate(mask, radius=_dilation):
    """Makes a dilation radius _dlation, in mm
    """
    arr = mask.np
    # Binarization of mask
    arr[arr < 1] = 0
    arr[arr >= 1] = _AIMS_BINARY_ONE
    # Dilates initial volume of 10 mm
    morpho = MorphoGreyLevel_S16()
    dilate = morpho.doDilation(mask, radius)
    arr_dilate = dilate.np
    arr_dilate[arr_dilate >= 1] = 1
    return dilate
    

def main():

    qapp = Qt.QApplication([])
    
    print("Reads spam and skeleton files")
    mask_result = aims.read(spam_file)
    skel_data = aims.read(skel_f)
    skel_data.np[:] = (skel_data.np > 0).astype(np.int16)

    print("Makes binarization and dilation on spam")
    mask_result[mask_result.np <= _threshold] = 0
    mask_result.np[:] = dilate(mask_result).np

    print("Masks skeleton data with dilated spam")
    skel_data.np[mask_result.np <= 0] = 0
    aims.write(skel_data, "/tmp/skel_before.nii.gz")

    print("Reads initial spam volume and transforms it to proba")
    spam_vol = aims.read(spam_file, dtype="Volume_FLOAT")
    spam_vol.np[:] = spam_vol.np/61

    print("Makes realignment")
    out_tr = spam_register(spam_vol,
                           skel_data,
                           do_mask=False,
                           R_angle_var=np.pi / 8,
                           t_var=5.,
                           verbose=False,
                           in_log=False,
                           calibrate_distrib=15.)
    aims.write(out_tr, '/tmp/transform.trm')

    print("Applies the realignment")
    os.system(f"AimsApplyTransform -i /tmp/skel_before.nii.gz -o /tmp/skel_realigned.nii.gz -m /tmp/transform.trm")

    print("Visualization")
    spam = a.loadObject(spam_file)
    skel = a.loadObject("/tmp/skel_before.nii.gz")
    realigned = a.loadObject("/tmp/skel_realigned.nii.gz")
    w = a.createWindow('Sagittal')
    w.addObjects(spam)
    w.addObjects(skel)
    w.addObjects(realigned)
    
    qapp.exec()
   

 





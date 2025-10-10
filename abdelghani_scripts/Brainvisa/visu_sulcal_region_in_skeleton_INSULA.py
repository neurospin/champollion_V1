"""
Multi-subject sulcal-region skeleton viewer (Anatomist)

For each subject in _SUBJECTS, this script:
- builds paths
- resamples the ICBM mask into the subject's native space
- extracts the cortical skeleton inside the region and its complement
- meshes both volumes
- displays them together with the subject's white mesh
"""

from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
from soma import aims
from soma import aimsalgo
import os
from typing import Tuple, Optional, List

# =========================
# User config
# =========================
# Examples:
# _SUBJECTS = ["sub-4270604", "sub-1134267", "sub-2106401"]
SUBJECTS_VISU ="CONTROL"
if SUBJECTS_VISU =="schiz":
    _SUBJECTS = [
        "sub-INVU4B4DLE",
        "sub-CH8853b",
        "sub-INVM3KPDY6",
        "sub-INV8317L2EY",
        "sub-INV2TA9J0N9",
        "sub-INVH170JG6K",
        "sub-bl001508",
        "sub-INV89625TBA",
        "sub-INVMG3JEL9H",
        "sub-INVHUUMFYFA"]
        #=========================================
else:        
    _SUBJECTS = [   
        "sub-CH7496a",
        "sub-pb090397",
        "sub-cb110048",
        "sub-am130237",
        "sub-INV85VZYDRM",
        "sub-INV918GKX4J",
        "sub-INV7YE3CMCL",
        "sub-CH2917",
        "sub-INVBATDPRHG",
        "sub-ab100404"
    ] 
_REGION = "INSULA_left"
_SIDE = "L"

# Bases and path segments
_DEEP_FOLDING_BASEDIR = "/neurospin/dico/data/deep_folding/current/datasets/aggregate_schizophrenia"
_MORPHOLOGIST_BASEDIR = "/home/cb283697/n4hukb/soft-brainvisa_provider-ns_morphologist"
_PATH_TO_WHITEMESH = "ses-2/anat/t1mri/default_acquisition/default_analysis/segmentation/mesh"

# Visualization toggles
SHOW_WHITE = True
SMOOTHING_STEPS = 100
SMOOTHING_FACTOR = 0.4  # 0..1

# Colors: cycles for subjects (masked/orange-ish, negative/blue-ish variants)
SUBJECT_COLORS = [
    # (inside_color RGBA, outside_color RGBA)
    ([1.00, 0.35, 0.00, 1.0], [0.00, 0.00, 1.00, 1.0]),
    ([0.90, 0.10, 0.20, 1.0], [0.10, 0.10, 0.80, 1.0]),
    ([0.95, 0.55, 0.15, 1.0], [0.15, 0.20, 0.85, 1.0]),
    ([0.85, 0.30, 0.10, 1.0], [0.20, 0.20, 0.90, 1.0]),
    ([1.00, 0.50, 0.00, 1.0], [0.00, 0.20, 0.90, 1.0]),
]

# =========================
# Anatomist init
# =========================
a = ana.Anatomist()

# =========================
# Helpers
# =========================
def subject_paths(subj: str) -> Tuple[str, str, str, str]:
    """Build all input file paths for a subject."""
    mask_icbm_file = f"{_DEEP_FOLDING_BASEDIR}/crops/2mm/{_REGION}/mask/{_SIDE}mask_skeleton.nii.gz"
    trm_raw_to_icbm_file = f"{_DEEP_FOLDING_BASEDIR}/transforms/{_SIDE}/{_SIDE}transform_to_ICBM2009c_{subj}.trm"
    skeleton_raw_file = f"{_DEEP_FOLDING_BASEDIR}/skeletons/raw/{_SIDE}/{_SIDE}skeleton_generated_{subj}.nii.gz"
    white_mesh_file = f"{_MORPHOLOGIST_BASEDIR}/{subj}/{_PATH_TO_WHITEMESH}/{subj}_{_SIDE}white.gii"
    return mask_icbm_file, trm_raw_to_icbm_file, skeleton_raw_file, white_mesh_file


def check_files(paths: List[str]) -> bool:
    ok = True
    for p in paths:
        if not os.path.exists(p):
            print(f"[WARN] Missing file: {p}")
            ok = False
    return ok


def mesh_and_merge(input_volume: aims.Volume_S16,
                   smoothing_steps: int = 100,
                   smoothing_factor: float = 0.4) -> aims.AimsTimeSurface_3_VOID:
    """Create a single merged mesh from a labeled volume."""
    mesher = aimsalgo.Mesher()
    mesher.setSmoothing(mesher.LOWPASS, smoothing_steps, smoothing_factor)
    mesh = mesher.doit(input_volume)  # dict: label -> [meshes]
    keys = list(mesh.keys())
    if not keys:
        # empty mesh: return an empty TimeSurface to avoid crashes
        return aims.AimsTimeSurface_3_VOID()

    # Start with first piece
    first_label = keys[0]
    if len(mesh[first_label]) == 0:
        return aims.AimsTimeSurface_3_VOID()
    mesh_concat = mesh[first_label][0]

    # Merge all remaining pieces
    for add in mesh[first_label][1:]:
        aims.SurfaceManip.meshMerge(mesh_concat, add)
    for k in keys[1:]:
        for add in mesh[k]:
            aims.SurfaceManip.meshMerge(mesh_concat, add)
    return mesh_concat


def compute_meshes_sulcal_region(
    skeleton_raw: aims.Volume_S16,
    mask_icbm: aims.Volume_S16,
    trm_raw_to_icbm: aims.AffineTransformation3d,
    tmp_mask_out: Optional[str] = None,
) -> Tuple[aims.AimsTimeSurface_3_VOID, aims.AimsTimeSurface_3_VOID]:
    """Return (mesh_inside_region, mesh_outside_region) in native space."""
    # Prepare an empty native-space mask volume
    hdr = skeleton_raw.header()
    dim = hdr['volume_dimension'][:3]
    mask_raw = aims.Volume(dim, dtype='S16')
    mask_raw.copyHeaderFrom(hdr)
    mask_raw.fill(0)

    # Resample ICBM mask to native
    resampler = aimsalgo.ResamplerFactory(mask_icbm).getResampler(0)  # NN
    resampler.setRef(mask_icbm)
    resampler.setDefaultValue(0)
    resampler.resample(mask_icbm, trm_raw_to_icbm.inverse(), 0, mask_raw)
    if tmp_mask_out:
        try:
            aims.write(mask_raw, tmp_mask_out)
        except Exception:
            pass

    # Close skeleton for better meshing
    mm = aimsalgo.MorphoGreyLevel_S16()
    skeleton_raw[skeleton_raw.np != 0] = 32767  # binarize as high value
    closed = mm.doClosing(skeleton_raw, 3.0).get()

    # Apply masks
    skeleton_inside = aims.Volume(closed)
    skeleton_inside[mask_raw.np == 0] = 0

    skeleton_outside = aims.Volume(closed)
    skeleton_outside[mask_raw.np != 0] = 0

    # Mesh both
    mesh_inside = mesh_and_merge(skeleton_inside, SMOOTHING_STEPS, SMOOTHING_FACTOR)
    mesh_outside = mesh_and_merge(skeleton_outside, SMOOTHING_STEPS, SMOOTHING_FACTOR)

    return mesh_inside, mesh_outside


def load_subject_objects(subj: str):
    """Read all inputs for a subject and return (mask_icbm, skeleton_raw, trm, white_mesh_a or None)."""
    mask_icbm_file, trm_file, skeleton_raw_file, white_mesh_file = subject_paths(subj)
    if not check_files([mask_icbm_file, trm_file, skeleton_raw_file]):
        return None

    try:
        mask_icbm = aims.read(mask_icbm_file)
        skeleton_raw = aims.read(skeleton_raw_file)
        trm_raw_to_icbm = aims.read(trm_file)
    except Exception as e:
        print(f"[ERROR] Reading volumes/transforms for {subj}: {e}")
        return None

    white_mesh_a = None
    if SHOW_WHITE and os.path.exists(white_mesh_file):
        try:
            white_mesh_a = a.loadObject(white_mesh_file, hidden=False)
        except Exception as e:
            print(f"[WARN] Could not load white mesh for {subj}: {e}")
            white_mesh_a = None
    else:
        if SHOW_WHITE:
            print(f"[WARN] White mesh not found for {subj}: {white_mesh_file}")

    return mask_icbm, skeleton_raw, trm_raw_to_icbm, white_mesh_a


# =========================
# Main
# =========================
if __name__ == "__main__":
    app = Qt.QApplication.instance()
    if app is None:
        app = Qt.QApplication([])

    # One 3D window for all subjects
    w = a.createWindow("3D")
    w.setHasCursor(0)

    # Optional: set a neutral background
    try:
        w.setBgColor([0.0, 0.0, 0.0, 1.0])
    except Exception:
        pass

    # Keep track of created objects for grouping
    all_objects = []

    for idx, subj in enumerate(_SUBJECTS):
        col_in, col_out = SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]

        loaded = load_subject_objects(subj)
        if loaded is None:
            print(f"[SKIP] {subj} due to missing/invalid inputs.")
            continue
        mask_icbm, skeleton_raw, trm_raw_to_icbm, white_mesh_a = loaded

        # Compute meshes (inside/outside)
        try:
            mesh_inside, mesh_outside = compute_meshes_sulcal_region(
                skeleton_raw, mask_icbm, trm_raw_to_icbm,
                tmp_mask_out=None  # or e.g., f"/tmp/{subj}_mask_raw.nii.gz"
            )
        except Exception as e:
            print(f"[ERROR] Mesh computation for {subj}: {e}")
            continue

        # Visualize subject's white mesh
        if SHOW_WHITE and white_mesh_a is not None:
            try:
                w.addObjects(white_mesh_a)
                white_mesh_a.setMaterial(a.Material(diffuse=[1.0, 1.0, 1.0, 1.0]))
                white_mesh_a.setName(f"{subj} White")
                all_objects.append(white_mesh_a)
            except Exception as e:
                print(f"[WARN] Display white mesh failed for {subj}: {e}")

        # Visualize inside/outside meshes
        try:
            inside_a = a.toAObject(mesh_inside)
            outside_a = a.toAObject(mesh_outside)

            inside_a.setMaterial(a.Material(diffuse=col_in))
            outside_a.setMaterial(a.Material(diffuse=col_out))

            inside_a.setName(f"{subj} | {_REGION}-{_SIDE} | inside")
            outside_a.setName(f"{subj} | {_REGION}-{_SIDE} | outside")

            w.addObjects([inside_a, outside_a])
            all_objects.extend([inside_a, outside_a])
        except Exception as e:
            print(f"[WARN] Display meshes failed for {subj}: {e}")
            continue

    # Optional: create a group to toggle everything at once
    if all_objects:
        try:
            group = a.createGroup(all_objects)
            group.setName(f"MultiSubject {_REGION}-{_SIDE} group")
        except Exception as e:
            print(f"[WARN] Could not create Anatomist group: {e}")

    app.exec_()

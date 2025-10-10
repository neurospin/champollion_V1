from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
import os
from soma import aims
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from soma.qt_gui.qt_backend import Qt

# -----------------------
# New path config (from your snippet)
# -----------------------
_REGION = "OCCIPITAL"
_SIDE = "L"  # 'L' or 'R'
_DEEP_FOLDING_BASEDIR = "/neurospin/dico/data/deep_folding/current/datasets/UkBioBank40"
_MORPHOLOGIST_BASEDIR = "/home/cb283697/n4hukb/soft-brainvisa_provider-ns_morphologist"
_PATH_TO_WHITEMESH = "ses-2/anat/t1mri/default_acquisition/default_analysis/segmentation/mesh"

# (These three inputs are shown for completeness; not used directly below)
# mask_icbm_file = f"{_DEEP_FOLDING_BASEDIR}/crops/2mm/{_REGION}/mask/{_SIDE}mask_skeleton.nii.gz"
# trm_raw_to_icbm_file = f"{_DEEP_FOLDING_BASEDIR}/transforms/{_SIDE}/{_SIDE}transform_to_ICBM2009c_{_SUBJECT}.trm"
# skeleton_raw_file = f"{_DEEP_FOLDING_BASEDIR}/skeletons/raw/{_SIDE}/{_SIDE}skeleton_generated_{_SUBJECT}.nii.gz"
SUBJECTS_VISU ="CONTROL"
if SUBJECTS_VISU =="MS":
    SUBJECTS = [
    
    "sub-1134267",
    "sub-4270604",
    "sub-5687536",
    "sub-2094554",
    "sub-5379026",
    
    "sub-5482063",
    "sub-1575185",
    "sub-5902976"
]

else:
    SUBJECTS = SUBJECTS = [
    "sub-1566169",
    "sub-2337969",
    "sub-4589975",
    "sub-4773726",
    "sub-2244805",
    "sub-1874228",
    "sub-1843000",
    "sub-5537302"
]



# -----------------------
# Grid helper (unchanged)
# -----------------------
def create_grid(image_files, n_cols, out_path, title=None, subject_names=None):
    imgs = [Image.open(f) for f in image_files]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    n_rows = (len(imgs) + n_cols - 1) // n_cols

    title_h = 0
    legend_h = 0
    font_size = 36
    if title or subject_names:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        if title:
            title_h = font.getbbox(title)[3] - font.getbbox(title)[1] + 10
        if subject_names:
            legend_h = font.getbbox("Test")[3] - font.getbbox("Test")[1] + 15

    grid = Image.new('RGB', (n_cols * w, n_rows * (h + legend_h) + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    if title:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        x = (grid.width - text_w) // 2
        draw.text((x, 5), title, fill=(0, 0, 0), font=font)

    for idx, im in enumerate(imgs):
        i, j = divmod(idx, n_cols)
        x0 = j * w
        y0 = title_h + i * (h + legend_h)
        grid.paste(im, (x0, y0))
        if subject_names:
            subj = subject_names[idx]
            tb = draw.textbbox((0, 0), subj, font=font)
            text_w = tb[2] - tb[0]
            draw.text((x0 + (w - text_w) // 2, y0 + h - 50), subj, fill=(0, 0, 0), font=font)

    grid.save(out_path)
    print(f"Snapshot of the block available at {out_path}")

# -----------------------
# Display / camera config
# -----------------------
HEMISPHERE = _SIDE
SULCUS_NAME = _REGION  # keep your code path consistent

CAMERA_PARAMS_FCM = {
    'view_quaternion': [-0.26836308836937, -0.323044091463089, -0.315022945404053, -0.851107776165009],
    'zoom': 1.25,
}
CAMERA_PARAMS_STS = {
    'view_quaternion': [-0.506462574005127, 0.718799889087677, 0.241088211536407, -0.410729736089706],
    'zoom': 1.25,
}
CAMERA_PARAMS_FCLP = {
    'view_quaternion': [-0.473252952098846, -0.42669266462326, 0.451164454221725, 0.624832570552826],
    'zoom': 1.25,
}
CAMERA_PARAMS_DEFAULT = {
    # 'view_quaternion': [-0.473252952098846, -0.42669266462326, 0.451164454221725, 0.624832570552826],
    # 'zoom': 1.25,
    'view_quaternion': [0.431420892477036, 0.53428453207016, 0.706584870815277, 0.170745134353638],
    'zoom': 1.25,
}
CAMERA_PARAMS_OCCIPITAL = {
    'view_quaternion':[-0.0374120064079762, -0.721375405788422, -0.691498458385468, -0.00691694254055619],

    'zoom': 1.0,
}

def quat_mul(q2, q1):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
    ]
Y180 = [0.0, 1.0, 0.0, 0.0]

# -----------------------
# Anatomist init
# -----------------------
app = Qt.QApplication.instance() or Qt.QApplication([])
a = ana.Anatomist()
windows = []
w = a.createWindow('3D')
windows.append(w)

# pick camera based on sulcus
if SULCUS_NAME == 'F.C.M.post._right':
    CAMERA_PARAMS = CAMERA_PARAMS_FCM
elif SULCUS_NAME == 'S.T.s._right':
    CAMERA_PARAMS = CAMERA_PARAMS_STS
elif SULCUS_NAME.startswith('F.C.L.p'):
    CAMERA_PARAMS = CAMERA_PARAMS_FCLP
elif SULCUS_NAME == ('OCCIPITAL'):
    CAMERA_PARAMS = CAMERA_PARAMS_OCCIPITAL
else:
    CAMERA_PARAMS = CAMERA_PARAMS_DEFAULT

view_q = CAMERA_PARAMS['view_quaternion']
# if HEMISPHERE.upper().startswith('L'):
#     view_q = quat_mul(Y180, view_q)

w.camera(
    zoom=CAMERA_PARAMS['zoom'],
    view_quaternion=view_q,
    force_redraw=True
)

# -----------------------
# New helpers to locate default_analysis and files
# -----------------------
def find_analysis_dir(subj: str):
    """
    Use the new UKB layout rooted at _MORPHOLOGIST_BASEDIR.
    Prefer ses-2; fall back to ses-1 if needed.
    Returns (da, tried_dirs)
    """
    tried = []
    # default_analysis sits here:
    # <BASE>/<subj>/ses-2/anat/t1mri/default_acquisition/default_analysis
    candidates = [
        os.path.join(_MORPHOLOGIST_BASEDIR, subj, "ses-2", "anat", "t1mri", "default_acquisition", "default_analysis"),
        os.path.join(_MORPHOLOGIST_BASEDIR, subj, "ses-1", "anat", "t1mri", "default_acquisition", "default_analysis"),
    ]
    for p in candidates:
        tried.append(p)
        if os.path.isdir(p):
            print(f"[INFO] Using analysis dir for {subj}: {p}")
            return p, tried
    return None, tried

def find_graph_arg(da: str, subj: str, hemisphere: str):
    """
    Returns (arg_path, tried_paths).
    Same logic as before, searching both dir and filename variants.
    """
    tried = []
    dir_candidates = ['3.1/deepcnn_session_auto', '3.1/default_session_auto','3.1/spam_session_auto']
    name_candidates = [
        f"{hemisphere}{subj}_deepcnn_session_auto.arg",
        f"{hemisphere}{subj}_default_session_auto.arg",
        f"{hemisphere}{subj}_spam_session_auto.arg",
        f"{hemisphere}{subj}.arg"
    ]
    for d in dir_candidates:
        for fname in name_candidates:
            p = os.path.join(da, 'folds', d, fname)
            tried.append(p)
            if os.path.isfile(p):
                return p, tried
    return None, tried

def build_white_mesh_path(subj: str) -> str:
    """
    Uses the exact value of _PATH_TO_WHITEMESH, rooted at _MORPHOLOGIST_BASEDIR/<subj>.
    """
    return os.path.join(
        _MORPHOLOGIST_BASEDIR,
        subj,
        _PATH_TO_WHITEMESH,
        f"{subj}_{_SIDE}white.gii"
    )

# -----------------------
# Loop over subjects
# -----------------------
skipped = []

for subj in SUBJECTS:
    da, tried_dirs = find_analysis_dir(subj)
    if da is None:
        print(f"[SKIP] No valid analysis folder found for {subj}. Tried:")
        for p in tried_dirs:
            print(f"       - {p}")
        skipped.append(subj)
        continue

    mesh_file  = build_white_mesh_path(subj)
    graph_file, tried_args = find_graph_arg(da, subj, _SIDE)

    # check mesh first
    if not os.path.isfile(mesh_file):
        print(f"[SKIP] Mesh missing for {subj}: {mesh_file}")
        print("       Analysis directory candidates checked (for context):")
        for p in tried_dirs:
            print(f"       - {p}")
        skipped.append(subj)
        continue

    # check graph .arg with the extended search
    if graph_file is None:
        print(f"[SKIP] Graph .arg missing for {subj}. Searched the following paths:")
        for p in tried_args:
            print(f"       - {p}")
        print("       Analysis directory used:")
        print(f"       - {da}")
        skipped.append(subj)
        continue

    # load and render
    mesh = a.loadObject(mesh_file, hidden=False)
    if mesh is None:
        print(f"[ERROR] Failed to load mesh for {subj}")
        skipped.append(subj)
        continue
    mesh.loadReferentialFromHeader()
    w.addObjects(mesh)
    mesh.setMaterial(a.Material(diffuse=[1, 1, 1, 0.5]))

    graph = a.loadObject(graph_file, hidden=False)
    if graph is None:
        print(f"[ERROR] Failed to load graph for {subj}")
        w.removeObjects([mesh])
        skipped.append(subj)
        continue
    graph.loadReferentialFromHeader()
    w.addObjects(graph, add_graph_nodes=True)
    w.setReferential(a.centralReferential())

    # color sulcus (robust match on label containing the region keyword)
    hie_path = aims.carto.Paths.findResourceFile('nomenclature/hierarchy/sulcal_root_colors.hie')
    nomenclature = a.loadObject(hie_path)
    a.execute('GraphDisplayProperties', objects=[graph], nomenclature_property='plouf')
    aims_graph = graph.graph()
    for vertex in aims_graph.vertices():
        label = vertex.get('label')
        if label and _REGION in label:
            a.execute('SetMaterial', objects=[vertex['ana_object']], diffuse=[1.0, 0.35, 0.0, 1.0])

    # snapshot
    save_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_figures"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{subj}_{SULCUS_NAME}.png"
    img_path = os.path.join(save_dir, fname)
    w.setHasCursor(0)
    w.snapshot(img_path, width=1200, height=900)

    # cleanup
    w.removeObjects([mesh, graph, nomenclature])

# report skipped subjects
if skipped:
    print("\nSubjects skipped due to missing files or load errors:")
    for s in sorted(set(skipped)):
        print(f"  - {s}")

# create montage
grid_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_figures"
create_grid(
    image_files=[os.path.join(grid_dir, f"{subj}_{SULCUS_NAME}.png") for subj in SUBJECTS if subj not in skipped],
    subject_names=[s for s in SUBJECTS if s not in skipped],
    n_cols=4,
    out_path=os.path.join(grid_dir, f"{_REGION}_{SUBJECTS_VISU}__grid_{_REGION}_10.png"),
    title=f"{_REGION}_{SUBJECTS_VISU}"
)

# optional: print camera infos every 5s
from PyQt5.QtCore import QTimer
def print_camera_infos():
    for idx, win in enumerate(windows):
        info = win.getInfos()
        print(f"Window {idx}: quat={info.get('view_quaternion')} zoom={info.get('zoom')}")
timer = QTimer()
timer.timeout.connect(print_camera_infos)
timer.start(5000)

Qt.QApplication.instance().exec()

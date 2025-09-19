from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
import os
from soma import aims
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from soma.qt_gui.qt_backend import Qt

def extract_ids(csv_path: str, n: int) -> list[str]:
    """
    Read `csv_path`, grab the first n and last n values from the 'ID' column.
    Returns them as a flat list (duplicates kept if overlap).
    """
    df = pd.read_csv(csv_path)
    if 'ID' not in df.columns:
        raise KeyError("CSV must have an 'ID' column")
    ids = df['ID']
    if len(ids) <= 2 * n:
        return ids.tolist()
    return ids.head(n).tolist() + ids.tail(n).tolist()


# SUBJECTS = extract_ids(csv_path, num_subjects)
# SUBJECTS = [
#     "sub-5420367",
#     "sub-1874228",
#     "sub-1843000",
#     "sub-5537302",
#     "sub-1874940",
#     "sub-1829795",
#     "sub-1415693",
#     "sub-4440981",
#     "sub-2498640",
#     "sub-5586284"
# ]
# [    # "sub-CH8593b",   # conf=0.838334, label=0
#     "sub-CH7098a",   # conf=0.772217, label=1
#     "sub-kb120176",  # conf=0.749500, label=1
#     "sub-CH7831a",   # conf=0.747320, label=1
#     "sub-CH7875b",   # conf=0.734446, label=1
#     # "sub-mb130153",  # conf=0.722822, label=0

#     "sub-INVZL5AW4AP",#conf=0.8240580700462996,label=1
#     "sub-INV937EPU3T",
#     "sub-INVFZZ967LD",
#     "sub-INVJJ95UD8P",
#     "sub-INVUDBY0P98",
#     "sub-INVZ3DTLE9E",
#     # "sub-INVL2D3J3RA",
#     # ------------------------
#     "sub-CH8033a",   # conf=0.284272, label=0
#     "sub-CH2917",    # conf=0.257507, label=0
#     # "sub-jf130314",  # conf=0.215812, label=1
#     "sub-ef100097",  # conf=0.205651, label=0
#     "sub-CH8095a",   # conf=0.166815, label=0
#     # "sub-ns110166",  # conf=0.155044, label=0
#     "sub-INVBX6FVFRG",
#     "sub-INV3XHCN2GH",
#     "sub-INVUA7FRWF6",
#     "sub-INV2LR63NG1",
#     "sub-INVHB7WFP9E",
#     "sub-INV7YE3CMCL",
# ]


SUBJECTS = [
    "sub-INVUA4B4DLE",
    "sub-CH8853b",
    "sub-INVM3KPDYD6",
    "sub-INV8317L2EY",
    "sub-INV2TA9J0N9",
    "sub-INVH170JGEK",
    "sub-bw100508",
    "sub-INV89625TBA",
    "sub-INVMGJ3EL9H",
    "sub-INVHUUMFYFA",
    #=========================================
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






def create_grid(image_files, n_cols, out_path, title=None, subject_names=None):
    # load all images
    imgs = [Image.open(f) for f in image_files]
    # calculate max width and height
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    # calculate number of rows
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


# ------------------------------------------------------
# Config
# ------------------------------------------------------
base_dir = '/neurospin/dico/data/human/aggregate_schizophrenia/derivatives/morphologist-2021/subjects2'
HEMISPHERE = 'L'
SULCUS_NAME = 'INSULA.'  

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
CAMERA_PARAMS_INSULA = {
    'view_quaternion': [0.570089042186737, -0.545621812343597, -0.530064225196838, 0.310366123914719], 

    'zoom': 2,
}
CAMERA_PARAMS_DEFAULT = {
    # original: [-0.473252952098846, -0.42669266462326, -0.451164454221725, 0.624832570552826]
    # ↓ pitched down ~25° (around X)
    # 'view_quaternion': [-0.2805, -0.3802, -0.4901, 0.7364],
    # 'zoom': 1.75,
    # 'view_quaternion': [-0.473252952098846, -0.42669266462326, 0.451164454221725, 0.624832570552826],
    # 'zoom': 1.25,
    'view_quaternion': [0.570089042186737, -0.545621812343597, -0.530064225196838, 0.310366123914719], 

    'zoom': 2.5,
}


# ------------------------------------------------------
# Camera flip logic
# ------------------------------------------------------
def quat_mul(q2, q1):
    # quaternion multiplication q = q2 ⊗ q1   (format [x,y,z,w])
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
        w2*w1 - x2*x1 - y2*y1 - z2*z1,
    ]

Y180 = [0.0, 1.0, 0.0, 0.0]  # 180° rotation around Y


# ------------------------------------------------------
# Anatomist init
# ------------------------------------------------------
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
elif SULCUS_NAME.startswith('INSULA_left'):
    CAMERA_PARAMS = CAMERA_PARAMS_INSULA    
else:
    CAMERA_PARAMS = CAMERA_PARAMS_DEFAULT

view_q = CAMERA_PARAMS['view_quaternion']
if HEMISPHERE.upper().startswith('L'):
    view_q = quat_mul(Y180, view_q)

w.camera(
    zoom=CAMERA_PARAMS['zoom'],
    view_quaternion=view_q,
    force_redraw=True
)


# ------------------------------------------------------
# Helpers to locate analysis dir and graph file (with debug)
# ------------------------------------------------------
def find_analysis_dir(subj: str):
    """
    Returns (da, tried_dirs)
      - da: a valid 'default_analysis' directory for the subject, or None.
      - tried_dirs: list of all candidate directories that were checked.
    For sub-INV, try multiple t1mri layouts; for others, use the legacy 'ses-v1' path.
    """
    tried = []
    if subj.startswith('sub-INV'):
        candidates = [
            os.path.join(base_dir, subj, 't1mri', 'default_acquisition', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'default_acquisation', 'default_analysis'),  # typo fallback
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-102_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-109_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-202_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-104_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-103_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-V1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-103_run-1', 'default_analysis'), 
        ]
        for p in candidates:
            tried.append(p)
            if os.path.isdir(p):
                print(f"[INFO] Using analysis dir for {subj}: {p}")
                return p, tried
        return None, tried
    else:
        p = os.path.join(base_dir, subj, 't1mri', 'ses-v1', 'default_analysis')
        tried.append(p)
        if os.path.isdir(p):
            return p, tried
        return None, tried




def find_graph_arg(da: str, subj: str, hemisphere: str):
    """
    Returns (arg_path, tried_paths)
      - arg_path: full path to the .arg file, or None
      - tried_paths: list of all .arg paths that were checked.
    Tries both dir and filename variants:
      - dirs: deepcnn_session_auto, default_session_auto
      - names: L{subj}_deepcnn_session_auto.arg, L{subj}_default_session_auto.arg
    """
    tried = []
    dir_candidates = ['deepcnn_session_auto', 'default_session_auto']
    name_candidates = [
        f"{hemisphere}{subj}_deepcnn_session_auto.arg",
        f"{hemisphere}{subj}_default_session_auto.arg",
    ]
    for d in dir_candidates:
        for fname in name_candidates:
            p = os.path.join(da, 'folds', '3.1', d, fname)
            tried.append(p)
            if os.path.isfile(p):
                return p, tried
    return None, tried


# ------------------------------------------------------
# Loop over subjects
# ------------------------------------------------------
skipped = []

for subj in SUBJECTS:
    da, tried_dirs = find_analysis_dir(subj)
    if da is None:
        print(f"[SKIP] No valid analysis folder found for {subj}. Tried:")
        for p in tried_dirs:
            print(f"       - {p}")
        skipped.append(subj)
        continue

    mesh_file  = os.path.join(da, 'segmentation', 'mesh', f"{subj}_{HEMISPHERE}white.gii")
    graph_file, tried_args = find_graph_arg(da, subj, HEMISPHERE)

    # check mesh first
    if not os.path.isfile(mesh_file):
        print(f"[SKIP] Mesh missing for {subj}: {mesh_file}")
        print("       Analysis directory used:")
        print(f"       - {da}")
        print("       All candidate analysis dirs (in order):")
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

    # color sulcus
    hie_path = aims.carto.Paths.findResourceFile('nomenclature/hierarchy/sulcal_root_colors.hie')
    nomenclature = a.loadObject(hie_path)
    a.execute(
        'GraphDisplayProperties',
        objects=[graph],
        nomenclature_property='plouf'
    )
    aims_graph = graph.graph()
    for vertex in aims_graph.vertices():
        # if vertex.get('label') in ("INSULA_left"):
        if vertex.get("label") == "INSULA_left":    
            a.execute(
                'SetMaterial',
                objects=[vertex['ana_object']],
                diffuse=[1.0, 0.35, 0.0, 1.0]
            )

    # snapshot
    save_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_figures_subjects"
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
    n_cols=10,
    out_path=os.path.join(grid_dir, f"{SULCUS_NAME}_grid_10.png"),
    title=f"{SULCUS_NAME}"
)

# # optional: print camera infos
from PyQt5.QtCore import QTimer
def print_camera_infos():
    for idx, win in enumerate(windows):
        info = win.getInfos()
        print(f"Window {idx}: quat={info.get('view_quaternion')} zoom={info.get('zoom')}")
timer = QTimer()
timer.timeout.connect(print_camera_infos)
timer.start(5000)

Qt.QApplication.instance().exec()

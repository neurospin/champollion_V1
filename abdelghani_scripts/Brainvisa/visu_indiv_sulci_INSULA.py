from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
import os
from soma import aims
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from soma.qt_gui.qt_backend import Qt

# ---------------------------------------
# Your SUBJECTS (unchanged)
# ---------------------------------------
# SUBJECTS = [
#     "sub-INV82PYLGB7",
#     "sub-INVZEB213F6",
#     "sub-INV63LUYT2C",
#     "sub-INVZL5AW4AP",
#     "sub-INV7VJB9UG1",
#     "sub-CH7353a",
#     "sub-INVVPW3T1AV",
#     "sub-INVM0H9UC11",
#     "sub-INVEUZWLV8G",
#     "sub-hb140282",
#     #=========================================
#     "sub-CH7496a",
#     "sub-pb090397",
#     "sub-cb110048",
#     "sub-am130237",
#     "sub-INV85VZYDRM",
#     "sub-INV918GKX4J",
#     "sub-INV7YE3CMCL",
#     "sub-CH2917",
#     "sub-INVBATDPRHG",
#     "sub-ab100404"
# ]

SUBJECTS = [
    
    "sub-INVZL5AW4AP",
    "sub-INVFZZ967LD",
    "sub-CH7098a",
    "sub-INVJJ95UD8P",
    "sub-INVUDBY0P98",
    "sub-kb120176",
    "sub-CH7831a",
    "sub-INVZ3DTLE9E",
    "sub-CH7875b",
    "sub-INV64AL1N24",
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

# ---------------------------------------
# Snapshot grid helper (unchanged)
# ---------------------------------------
def create_grid(image_files, n_cols, out_path, title=None, subject_names=None):
    imgs = [Image.open(f) for f in image_files]
    w = max(im.width for im in imgs)
    h = max(im.height for im in imgs)
    n_rows = (len(imgs) + n_cols - 1) // n_cols

    title_h = 0
    legend_h = 0
    font_size = 36
    font = None
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
SULCUS_NAME = 'INSULA.'   # label string used only in filenames/titles

# Camera presets (kept as-is)
CAMERA_PARAMS_INSULA = {
    #'view_quaternion': [-0.848615407943726, 0.272645741701126, -0.354154855012894, 0.283002942800522], 
    #'view_quaternion' : [0.354154855012894, -0.283002942800522, -0.848615407943726, 0.272645741701126],
    #[-0.37881138920784, -0.399040639400482, 0.596823990345001, 0.584011614322662]
    'view_quaternion' : [-0.596823990345001, -0.584011614322662, -0.37881138920784, -0.399040639400482],
    'zoom': 3.32,
    }
CAMERA_PARAMS_DEFAULT = {
    'view_quaternion' : [-0.596823990345001, -0.584011614322662, -0.37881138920784, -0.399040639400482],

    'zoom': 3.32,
    
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

Y180 = [0.0, 1.0, 0.0, 0.0]  # 180° around Y

# ------------------------------------------------------
# Anatomist init
# ------------------------------------------------------
app = Qt.QApplication.instance() or Qt.QApplication([])
a = ana.Anatomist()
windows = []
w = a.createWindow('3D')
windows.append(w)

# Choose camera (INSULA-focused)
CAMERA_PARAMS = CAMERA_PARAMS_INSULA
view_q = CAMERA_PARAMS['view_quaternion']
if HEMISPHERE.upper().startswith('L'):
    view_q = quat_mul(Y180, view_q)
w.camera(zoom=CAMERA_PARAMS['zoom'], view_quaternion=view_q, force_redraw=True)

# ------------------------------------------------------
# Helpers to locate analysis dir and graph file
# ------------------------------------------------------
def find_analysis_dir(subj: str):
    tried = []
    if subj.startswith('sub-INV'):
        candidates = [
            os.path.join(base_dir, subj, 't1mri', 'default_acquisition', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'default_acquisation', 'default_analysis'),  # fallback
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-102_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-109_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-202_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-104_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-103_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-V1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-103_run-1', 'default_analysis'),
            os.path.join(base_dir, subj, 't1mri', 'ses-1_acq-106_run-1', 'default_analysis'),
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
# Main loop — render ONLY the INSULA sulcus
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

    # --- White mesh: disabled (can re-enable later) ---
    # mesh_file = os.path.join(da, 'segmentation', 'mesh', f"{subj}_{HEMISPHERE}white.gii")
    # if os.path.isfile(mesh_file):
    #     mesh = a.loadObject(mesh_file, hidden=False)
    #     if mesh is not None:
    #         mesh.loadReferentialFromHeader()
    #         w.addObjects(mesh)
    #         mesh.setMaterial(a.Material(diffuse=[1, 1, 1, 0.2]))  # very translucent if re-enabled
    # else:
    #     print(f"[INFO] White mesh not found for {subj}: {mesh_file}")

    # Graph
    graph_file, tried_args = find_graph_arg(da, subj, HEMISPHERE)
    if graph_file is None:
        print(f"[SKIP] Graph .arg missing for {subj}. Searched:")
        for p in tried_args:
            print(f"       - {p}")
        print(f"       Analysis dir: {da}")
        skipped.append(subj)
        continue

    graph = a.loadObject(graph_file, hidden=False)
    if graph is None:
        print(f"[ERROR] Failed to load graph for {subj}")
        skipped.append(subj)
        continue
    graph.loadReferentialFromHeader()

    # IMPORTANT: do NOT add all graph nodes to the window
    # (this would show everything). We'll selectively show only INSULA nodes.
    # w.addObjects(graph, add_graph_nodes=True)  # <-- DISABLED

    # Optional nomenclature load (kept but not used to color everything)
    # hie_path = aims.carto.Paths.findResourceFile('nomenclature/hierarchy/sulcal_root_colors.hie')
    # nomenclature = a.loadObject(hie_path)
    # a.execute('GraphDisplayProperties', objects=[graph], nomenclature_property='plouf')

    aims_graph = graph.graph()

    # First: collect all node display objects, hide them (alpha=0).
    node_objs = []
    for vertex in aims_graph.vertices():
        if 'ana_object' in vertex:
            node_objs.append(vertex['ana_object'])
    if node_objs:
        try:
            a.execute(
                'SetMaterial',
                objects=node_objs,
                diffuse=[0.8, 0.8, 0.8, 0.0]  # alpha 0 => fully hidden
            )
        except Exception as e:
            print(f"[WARN] Could not hide all nodes for {subj}: {e}")

    # Now: enable & color ONLY INSULA_left
    insula_objs = []
    for vertex in aims_graph.vertices():
        label = vertex.get("label")
        # robust: handle None safely
        if isinstance(label, str) and label == "INSULA_left":
            if 'ana_object' in vertex:
                insula_objs.append(vertex['ana_object'])

    if not insula_objs:
        print(f"[SKIP] No INSULA_left node found for {subj}")
        skipped.append(subj)
        # Remove the graph to keep the view clean
        w.removeObjects([graph])
        continue

    # Add just INSULA nodes to the window and make them visible & orange
    try:
        w.addObjects(insula_objs)
        a.execute(
            'SetMaterial',
            objects=insula_objs,
            diffuse=[1.0, 0.35, 0.0, 1.0]  # orange, opaque
        )
    except Exception as e:
        print(f"[WARN] Failed to display INSULA for {subj}: {e}")
        w.removeObjects([graph])
        skipped.append(subj)
        continue

    # Keep graph object minimally (optional), but do NOT add it to the window
    # so only INSULA node(s) remain visible.

    # Snapshot
    save_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_figures"
    os.makedirs(save_dir, exist_ok=True)
    fname = f"{subj}_{SULCUS_NAME}_ONLY.png"
    img_path = os.path.join(save_dir, fname)
    w.setHasCursor(0)
    w.snapshot(img_path, width=1200, height=900)

    # Cleanup the visible INSULA nodes (keep window tidy for next subject)
    w.removeObjects(insula_objs)
    # If you ever want to fully drop the graph object too:
    w.removeObjects([graph])

# Report skipped subjects
if skipped:
    print("\nSubjects skipped due to missing files or load issues:")
    for s in sorted(set(skipped)):
        print(f"  - {s}")

# Montage of all INSULA-only snapshots
grid_dir = "/neurospin/dico/babdelghani/Runs/02_champollion_v1/Program/2023_jlaval_STSbabies/abdelghani_figures"
create_grid(
    image_files=[os.path.join(grid_dir, f"{subj}_{SULCUS_NAME}_ONLY.png") for subj in SUBJECTS if subj not in skipped],
    subject_names=[s for s in SUBJECTS if s not in skipped],
    n_cols=10,
    out_path=os.path.join(grid_dir, f"{SULCUS_NAME}_ONLY_grid_DC_OF_FCLP_Diretion_10.png"),
    title=f"{SULCUS_NAME} (only)"
)

# Optional: camera print (kept)
from PyQt5.QtCore import QTimer
def print_camera_infos():
    for idx, win in enumerate(windows):
        info = win.getInfos()
        print(f"Window {idx}: quat={info.get('view_quaternion')} zoom={info.get('zoom')}")
timer = QTimer()
timer.timeout.connect(print_camera_infos)
timer.start(5000)

Qt.QApplication.instance().exec()

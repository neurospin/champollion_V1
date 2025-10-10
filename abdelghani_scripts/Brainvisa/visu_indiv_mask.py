from soma.qt_gui.qt_backend import Qt
import anatomist.api as ana
import os
from soma import aims
from soma import aimsalgo
from PIL import Image, ImageDraw, ImageFont


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
            title_h = font.getbbox(title)[3] - font.getbbox(title)[1] + 10  # add margin
        if subject_names:
            legend_h = font.getbbox("Test")[3] - font.getbbox("Test")[1] + 15  # idem

    # create a new blank image with space for title and legend
    grid = Image.new('RGB', (n_cols * w, n_rows * (h + legend_h) + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Draw title
    if title:
        bbox = draw.textbbox((0, 0), title, font=font)  
        text_w = bbox[2] - bbox[0]
        x = (grid.width - text_w) // 2
        draw.text((x, 5), title, fill=(0, 0, 0), font=font)

    # Paste images and draw subject names
    for idx, im in enumerate(imgs):
        i, j = divmod(idx, n_cols)
        x0 = j * w
        y0 = title_h + i * (h + legend_h)
        grid.paste(im, (x0, y0))

        if subject_names:
            subj = subject_names[idx]
            text_bbox = draw.textbbox((0, 0), subj, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            draw.text((x0 + (w - text_w) // 2, y0 + h-50), subj, fill=(0, 0, 0), font=font)

    grid.save(out_path)
    print(f"Snapshot of the block available at {out_path}")



def mesh_and_merge(input_volume: aims.Volume_S16) \
        -> aims.AimsTimeSurface_3_VOID:
    """Creates a unique mesh from an input volume.
    
    Parameters
    ----------
    input_volume: aims.Volume, volume to mesh
    
    Returns
    -------
    mesh_concat: aims.AimsTimeSurface_3_VOID, concatenated mesh
    """
    # Creates a mesh from the input volume
    mesher = aimsalgo.Mesher()
    mesher.setSmoothing(mesher.LOWPASS, 100, 0.4)
    mesh = mesher.doit(input_volume)
    print(mesh.keys())
    keys = list(mesh.keys())
    key = keys[0]
    print(mesh[key])

    # Merges all meshes into mesh_concat
    mesh_concat = mesh[keys[0]][0]
    for mesh_add in mesh[keys[0]][1:]:
        aims.SurfaceManip.meshMerge(mesh_concat, mesh_add)
    for key in keys[1:]:
        for mesh_add in mesh[key]:
            aims.SurfaceManip.meshMerge(mesh_concat, mesh_add)

    return mesh_concat


def compute_meshes_sulcal_region(skeleton_raw: aims.Volume_S16,
                                 mask_icbm: aims.Volume_S16,
                                 trm_raw_to_icbm: aims.AffineTransformation3d)\
        -> tuple:
    """Returns two meshes: the one of skeleton inside the sulcal region,
    and the one outside the sulcal region.

    The meshes are computed in the native space of the subject.
    
    Parameters
    ----------
    skeleton_raw: aims.Volume_S16
        cortical skeleton in native space
    mask_icbm: aims.Volume_S16
        binary mask defining the sulcal region in ICBM2009c space
    trm_raw_to_icbm: aims.AffineTransformation3d
        transform from the native to the ICBM2009c space
    
    Returns
    -------
    mesh_masked: aims.AimsTimeSurface_3_VOID
        mesh of the cortical skeleton of the sulcal region
    mesh_negative_masked: aims.AimsTimeSurface_3_VOID
        mesh of the cortical skeleton outside the sulcal region
    """
    # We create a volume mask_raw filled with 0 of the same size 
    # and in the same referential as skeleton_raw
    hdr = skeleton_raw.header()
    dim = hdr['volume_dimension'][:3]
    mask_raw = aims.Volume(dim, dtype='S16')
    mask_raw.copyHeaderFrom(hdr)
    mask_raw.fill(0)

    # Makes the actual resampling of mask_icbm into mask_raw
    # 0 order (nearest neightbours) resampling
    resampler = aimsalgo.ResamplerFactory(mask_icbm).getResampler(0)
    resampler.setRef(mask_icbm)
    resampler.setDefaultValue(0)
    resampler.resample(mask_icbm, trm_raw_to_icbm.inverse(), 0, mask_raw)
    aims.write(mask_raw, "/tmp/mask_raw.nii.gz")

    # Print the headers
    print(f"skeleton_raw header = \n{skeleton_raw.header()}\n")
    print(f"mask_raw header = \n{mask_raw.header()}\n")

    # Closing
    mm = aimsalgo.MorphoGreyLevel_S16()
    skeleton_raw[skeleton_raw.np != 0] = 32767
    closed = mm.doClosing(skeleton_raw, 3.)
    closed = closed.get()

    # Masks the raw skeleton in the raw (native) space with mask_raw
    skeleton_masked = aims.Volume(closed)
    skeleton_masked[mask_raw.np == 0]= 0

    # Mask tha raw skeleton with the negative of the mask
    skeleton_negative_masked = aims.Volume(closed)
    skeleton_negative_masked[mask_raw.np != 0]= 0

    mesh_masked = mesh_and_merge(skeleton_masked)
    mesh_negative_masked = mesh_and_merge(skeleton_negative_masked)

    return mesh_masked, mesh_negative_masked



# select sulcus name, hemisphere, and subjects list
dataset = 'ABCD'  

if dataset == 'dHCP':
    base_dir = (
        '/neurospin/dico/data/human/dHCP/derivatives/release3/morphologist-2023'
    )

else : 
    base_dir = (
        '/neurospin/dico/rmenasria/Runs/03_main/Input/mnt/'
        'soft-brainvisa_ses-baselineYear1Arm1_ver-5.2_morphologist'
    )

PREMA_28_32_CLASSIF_ABCD_STs = [
    "sub-NDARINV9HVP5GZ1",
    "sub-NDARINVZ694ZZUM",
    "sub-NDARINVE75NTPTJ",
    "sub-NDARINVUV48BM08",
    "sub-NDARINVEYANDC02",
    "sub-NDARINV68V7T6T1",
    "sub-NDARINVB14AY6F8",
    "sub-NDARINV9DLWDBL9",
    "sub-NDARINV0T5DCF9K",
    "sub-NDARINV5PU4UNRW",
    "sub-NDARINVYZMY8DH5",
    "sub-NDARINVFBKHE2PK",
]

FULLTERMS_28_32_CLASSIF_ABCD_STs = [
    "sub-NDARINVCRY0YLTV",
    "sub-NDARINVGLNY79U6",
    "sub-NDARINVMXNTYPJU",
    "sub-NDARINVHE58CENN",
    "sub-NDARINVULT8KHWH",
    "sub-NDARINV1HKCTUW1",
    "sub-NDARINVNYP48E8M",
    "sub-NDARINVDMMAKV5Y",
    "sub-NDARINVWANGBMC4",
    "sub-NDARINV21VH8BF0",
    "sub-NDARINVZR16R6Y3",
    "sub-NDARINV9XZMNP0M",
]

PREMA_28_32_CLASSIF_ABCD_FCM= [
    "sub-NDARINVZGEEAXPZ",
    "sub-NDARINVGD7X7HAX",
    "sub-NDARINVLEFHD44M",
    "sub-NDARINVRZL7PGK1",
    "sub-NDARINVKPLUBC73",
    "sub-NDARINVY2DXN8WF",
    "sub-NDARINVKEDY719E",
    "sub-NDARINV7FG8NTPP",
    "sub-NDARINVH68T1XNB",
    "sub-NDARINV0V1TNU11",
    "sub-NDARINV4C12ZREL",
    "sub-NDARINVWBVTFEHK",
]

FULLTERMS_28_32_CLASSIF_ABCD_FCM = [
    "sub-NDARINV3H5U1V47",
    "sub-NDARINVUMTF6KTU",
    "sub-NDARINV2U88LX3T",
    "sub-NDARINVDHVY95MH",
    "sub-NDARINV4ET3VY02",
    "sub-NDARINV979TRHXJ",
    "sub-NDARINVJ1551F3N",
    "sub-NDARINVGNKUYWNE",
    "sub-NDARINV5HNAA4NT",
    "sub-NDARINVV18ZDFDE",
    "sub-NDARINVLABWKL63",
    "sub-NDARINVDZLD38UM",
]



BEST_RANKED_PREMAS_COGDIR_PICVOCAB = ['sub-NDARINVYFW72521', 'sub-NDARINVXBX1H8JJ', 'sub-NDARINVUFJX48LE', 'sub-NDARINVFLA7T3CM', 'sub-NDARINV3B60KLEU', 'sub-NDARINVEZXL2KXT', 'sub-NDARINVN18KZ8TG', 'sub-NDARINV898CW3V8', 'sub-NDARINVMWH0MZFX', 'sub-NDARINVXPGNE2W4', 'sub-NDARINVNMJZEZRD', 'sub-NDARINVFZ97ZA0Z', 'sub-NDARINVXXKJDH17', 'sub-NDARINVMBU1DVRD', 'sub-NDARINVRTYGC16L', 'sub-NDARINVGB26JEH6']

SUBJECTS_TEST = BEST_RANKED_PREMAS_COGDIR_PICVOCAB
HEMISPHERE = 'R'
#SULCUS_NAME = 'S.T.s.'  # Trop haut dans la nomenclature, donc colorie trop de sillons
SULCUS_NAME = 'F.C.M.post.'  
#SULCUS_NAME = 'F.C.L.p._right'  # F.C.L.p_right est le nom du sulcus dans la nomenclature
#SULCUS_NAMES = ["INSULA_right","F.C.L.p._right"]
#SULCUS_NAME = "F.Coll.-S.Rh."

CAMERA_PARAMS_FCM = {
    'view_quaternion': [-0.26836308836937, -0.323044091463089, -0.315022945404053, -0.851107776165009],
    'zoom': 1.25,
}
CAMERA_PARAMS_STS = {
    'view_quaternion': [-0.506462574005127, 0.718799889087677, 0.241088211536407, -0.410729736089706],
    'zoom': 1.25,
}

CAMERA_PARAMS_FCOLL = {
    'view_quaternion': [0.60460638999939, 0.670526087284088, 0.219961911439896, 0.3694087266922],
    'zoom': 1.25,
}

CAMERA_PARAMS_DEFAULT = {
    'view_quaternion': [0.473252952098846, -0.42669266462326, -0.451164454221725, 0.624832570552826],
    'zoom': 1.25,
}



# here select the list 
SUBJECTS =PREMA_28_32_CLASSIF_ABCD_FCM

app = Qt.QApplication.instance() or Qt.QApplication([])
a = ana.Anatomist()
windows = []
w = a.createWindow('3D')
windows.append(w)
#w.assignReferential(a.centralRef)

if SULCUS_NAME == 'F.C.M.post.':
        CAMERA_PARAMS = CAMERA_PARAMS_FCM
elif SULCUS_NAME == 'S.T.s.':
    CAMERA_PARAMS = CAMERA_PARAMS_STS

elif SULCUS_NAME == 'F.Coll.-S.Rh.':
    CAMERA_PARAMS = CAMERA_PARAMS_FCOLL
else:
    CAMERA_PARAMS = CAMERA_PARAMS_DEFAULT 
    
w.camera(
    zoom=CAMERA_PARAMS['zoom'],
    view_quaternion=CAMERA_PARAMS['view_quaternion'],
    force_redraw=True
)


for subj in SUBJECTS:

    if dataset == 'dHCP':
        da = os.path.join(base_dir, f"sub-{subj}",'anat','t1mri',
            'default_acquisition', 'default_analysis')
        mesh_file = os.path.join(
            da, 'segmentation', 'mesh', f"{subj}_{HEMISPHERE}white.gii")
        
        DEEP_FOLDING_BASEDIR = "/neurospin/dico/data/human/dHCP/derivatives/release3/deep_folding-2025"
    
    else : 
        da = os.path.join(
            base_dir, subj, 'ses-1', 'anat', 't1mri',
            'default_acquisition', 'default_analysis'
        )

        mesh_file = os.path.join(da, 'segmentation', 'mesh', f"{subj}_{HEMISPHERE}white.gii")
        
        DEEP_FOLDING_BASEDIR = "/neurospin/dico/data/human/ABCD/derivatives/deep_folding-2025"

       

    mask_icbm_file = f"{DEEP_FOLDING_BASEDIR}/crops/2mm/{SULCUS_NAME}/mask/{HEMISPHERE}mask_skeleton.nii.gz"
    trm_raw_to_icbm_file = f"{DEEP_FOLDING_BASEDIR}/transforms/{HEMISPHERE}/{HEMISPHERE}transform_to_ICBM2009c_{subj}.trm"
    skeleton_raw_file = f"{DEEP_FOLDING_BASEDIR}/skeletons/raw/{HEMISPHERE}/{HEMISPHERE}skeleton_generated_{subj}.nii.gz"


    #We read mask in ICBM 2009 referential, raw skeleton
    # and transform from raw referential to ICBM2009
    mask_icbm = aims.read(mask_icbm_file)
    skeleton_raw = aims.read(skeleton_raw_file)
    trm_raw_to_icbm = aims.read(trm_raw_to_icbm_file)

    mesh = a.loadObject(mesh_file, hidden=False)



    mesh_masked, mesh_negative_masked = \
        compute_meshes_sulcal_region(skeleton_raw, mask_icbm, trm_raw_to_icbm)
    
    mesh.loadReferentialFromHeader()
    w.addObjects(mesh)
    mesh.setMaterial(a.Material(diffuse=[1.0, 1.0, 1.0, 1.0],))

    # Visualize concatenated meshes
    mesh_masked_a = a.toAObject(mesh_masked)
    w.addObjects(mesh_masked_a)
    mesh_masked_a.setMaterial(a.Material(diffuse=[1.0, 0.35, 0.0, 1.0]))

    mesh_negative_masked_a = a.toAObject(mesh_negative_masked)
    w.addObjects(mesh_negative_masked_a)
    mesh_negative_masked_a.setMaterial(a.Material(diffuse=[0.0, 0., 1.0, 1.0]))

    



    snapshot = True


    # set the camera position

    if snapshot:

        save_dir = "/neurospin/dico/rmenasria/Runs/03_main/Output/Figures/anat_snapshots/mask/fullterms_28_32"
        w.setHasCursor(0)
        fname = f"{subj}_{SULCUS_NAME}.png"
        img_path = os.path.join(save_dir, fname)
        w.snapshot(img_path, width=1200, height=900)

        # Close the window
        #w.close()


    #Remove objects from the window
    w.removeObjects([mesh, mesh_masked_a, mesh_negative_masked_a])



grid_dir = "/neurospin/dico/rmenasria/Runs/03_main/Output/Figures/anat_snapshots/mask/fullterms_28_32"
create_grid(
    image_files=[
        os.path.join(grid_dir, f"{subj}_{SULCUS_NAME}.png") for subj in SUBJECTS
    ],
    subject_names=SUBJECTS,
    n_cols= 4,
    out_path=os.path.join(
        grid_dir, f"{SULCUS_NAME}_grid.png"),
    title=f"{SULCUS_NAME}"
)


# # Uncomment the following lines if you want to print camera infos periodically in the console. It allows to set manually the camera position in the code above.
# from PyQt5.QtCore import QTimer

# def print_camera_infos():
#     for idx, win in enumerate(windows):
#         try:
#             info = win.getInfos()
#             quat = info.get('view_quaternion', None)
#             zoom = info.get('zoom', None)
#             print(f"Fenêtre {idx} :")
#             print(f"  Quaternion : {quat}")
#             print(f"  Zoom       : {zoom}")
#         except Exception as e:
#             print(f"Erreur pour la fenêtre {idx} : {e}")

# timer = QTimer()
# timer.timeout.connect(print_camera_infos)
# timer.start(5000)



app.exec_()
"""Shared helpers for visualizing the RTOG 0617 npz files (VCU_DATA/Processed_rtog_npz_IMRT).

npz layout, as written by VCU_CLEAN/preprocess/data_preprocess_reduction_rtog_gpu_new.py:
    CT, DOSE        float32 (128, 128, 128), raw HU / Gy (dose clipped at 85 Gy, NOT normalized)
    OAR             uint8   (128, 128, 128), 1 cord, 2 esophagus, 3 heart, 4 lung ipsi, 5 lung contra
    PTV             uint8   (128, 128, 128), 0/1
    DOSE_SCALE      float,  DOSE[PTV > 0].mean() * DOSE_SCALE == 60 Gy
    HIST_SIG, HIST_CLINICAL  float32 (376, 8), cumulative DVHs as volume FRACTION (0-1)
    BINS            float32 (376,), dose bins of HIST_CLINICAL (0-80 Gy). HIST_SIG was computed on
                    linspace(0, 75, 376) (get_dvh), which is NOT saved; use sig_bins() for it.
    LUNG_IPSI_SIDE  'L' / 'R' / '' (side of the ipsilateral lung)

Array axes are (X, Y, Z) = (DICOM column: patient right -> left, DICOM row: anterior -> posterior,
slice: inferior -> superior). This is the DoseRTX convention ("AP along x, LR along y" when an axial
slice arr[:, :, z] is shown as is). arr.transpose(2, 1, 0) gives the usual SimpleITK (Z, Y, X) order.

The npz files carry no voxel spacing. The 128^3 grid comes from a ~320 x 320 pixel in-plane crop and a
slice crop around the lungs / heart / PTV (>= 128 slices) of the original CT, so the spacing is estimated
from the original DICOM CT and RTSTRUCT (see estimate_spacing). It only affects the aspect ratio of the
figures, not the DVHs (volume %).
"""
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

NPZ_DIR = '/home/grp_airtp/AI_RTP/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT'
DICOM_ROOT = '/mnt/data1/RTOG_DICOM_data/manifest-1542748232960/NSCLC-Cetuximab'
OUT_DIR = '/home/grp_airtp/AI_RTP/KBP_Project/results/visualization_examples'

# Example cases: left/right tumours, small to large PTV
EXAMPLE_CASES = ['0617-634370_0', '0617-622033_0', '0617-779653_0', '0617-761493_0']

GRID = 128
CROP_INPLANE = 320              # get_crop_settings: center +/- 160 pixels, clamped to the image
DEFAULT_SPACING = (2.5, 2.5, 2.5)
SIG_BINS_MAX = 75.0             # get_dvh (HIST_SIG) bins: linspace(0, 75, 376); BINS holds the 0-80 Gy ones
CT_WINDOW = (-1000, 1300)       # platipy window = (lower, width): lung + mediastinum

# key, OAR label (None = PTV array), HIST_* column, colour on light background, colour on dark background
# HIST_* columns come from one_hot(OAR, 8)[..., 1:] (labels 1..7) plus PTV as the last column;
# columns 5 and 6 are unused labels (NaN).
STRUCTURES = [
    ('cord',      1,    0, '#2a78d6', '#3987e5'),
    ('esophagus', 2,    1, '#008300', '#008300'),
    ('heart',     3,    2, '#1baf7a', '#199e70'),
    ('lung_ipsi', 4,    3, '#eda100', '#c98500'),
    ('lung_cntr', 5,    4, '#4a3aa7', '#9085e9'),
    ('ptv',       None, 7, '#e34948', '#e66767'),
]
DISPLAY_NAMES = {'cord': 'Spinal cord', 'esophagus': 'Esophagus', 'heart': 'Heart',
                 'lung_ipsi': 'Lung ipsi', 'lung_cntr': 'Lung contra', 'ptv': 'PTV'}


@dataclass
class RTOGCase:
    case_id: str
    ct: np.ndarray
    dose: np.ndarray            # Gy; multiplied by dose_scale when loaded with normalize=True
    oar: np.ndarray
    ptv: np.ndarray
    dose_scale: float
    normalized: bool
    lung_ipsi_side: str
    bins: np.ndarray
    hist_clinical: np.ndarray
    hist_sig: np.ndarray
    spacing: tuple              # (x, y, z) mm, estimated
    spacing_source: str

    def display_name(self, key):
        name = DISPLAY_NAMES[key]
        if key in ('lung_ipsi', 'lung_cntr') and self.lung_ipsi_side in ('L', 'R'):
            side = self.lung_ipsi_side if key == 'lung_ipsi' else {'L': 'R', 'R': 'L'}[self.lung_ipsi_side]
            name += ' ({})'.format(side)
        return name

    def masks(self):
        """{key: bool (X, Y, Z) mask} for every structure present in this case."""
        out = {}
        for key, label, _, _, _ in STRUCTURES:
            m = self.ptv > 0 if label is None else self.oar == label
            if m.any():
                out[key] = m
        return out

    def dvh_bins(self):
        # V(b) of DOSE * s equals V(b / s) of DOSE, so the stored DVH is rescaled by moving the bins
        return self.bins * self.dose_scale if self.normalized else self.bins

    def sig_bins(self):
        bins = np.linspace(0, SIG_BINS_MAX, self.hist_sig.shape[0], dtype=np.float32)
        return bins * self.dose_scale if self.normalized else bins


def resolve_npz(case_or_path, npz_dir=NPZ_DIR):
    if os.path.isfile(case_or_path):
        return case_or_path
    name = case_or_path if case_or_path.endswith('.npz') else case_or_path + '.npz'
    path = os.path.join(npz_dir, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _crop_anatomy_z_range(struct_file):
    """z range (mm) of the lung / heart / PTV contours, the structures get_crop_settings crops around."""
    from pydicom import dcmread
    ds = dcmread(struct_file)
    names = {r.ROINumber: str(r.ROIName).lower() for r in ds.StructureSetROISequence}
    zs = []
    for rc in ds.ROIContourSequence:
        name = names.get(rc.ReferencedROINumber, '')
        if re.search(r'lung|heart|ptv', name) and not re.search(r'cord|eso', name):
            zs += [float(c.ContourData[2]) for c in getattr(rc, 'ContourSequence', [])]
    return (min(zs), max(zs)) if zs else None


def estimate_spacing(case_id, dicom_root=DICOM_ROOT):
    """Approximate (x, y, z) voxel spacing in mm of the 128^3 npz grid from the original DICOM.

    Mirrors get_crop_settings / crop_resize_img of the preprocessing script:
      in-plane: min(320, n_pixels) pixels are resized to 128;
      z: a CT with < 128 slices is resized from all slices; otherwise the slices spanning the lung /
         heart / PTV contours are kept, padded to 128 if they span fewer.
    """
    patient = case_id.rsplit('_', 1)[0]
    ct_dirs = sorted(glob.glob(os.path.join(dicom_root, patient, '*', '*CT*')))
    files = sorted(glob.glob(os.path.join(ct_dirs[0], '*.dcm'))) if ct_dirs else []
    if not files:
        return DEFAULT_SPACING, 'default (no DICOM found)'
    try:
        from pydicom import dcmread
        headers = [dcmread(f, stop_before_pixels=True) for f in files]
        z = np.sort([float(h.ImagePositionPatient[2]) for h in headers])
        thickness = float(np.median(np.diff(z))) if len(z) > 1 else float(headers[0].SliceThickness)
        ps_row, ps_col = [float(v) for v in headers[0].PixelSpacing]
        sx = min(CROP_INPLANE, int(headers[0].Columns)) * ps_col / GRID
        sy = min(CROP_INPLANE, int(headers[0].Rows)) * ps_row / GRID

        n_slices, source = len(z), 'DICOM CT'
        if n_slices >= GRID:
            n_slices = GRID
            structs = sorted(glob.glob(os.path.join(dicom_root, patient, '*', '*RTStruct*', '*.dcm')))
            z_range = _crop_anatomy_z_range(structs[0]) if structs else None
            if z_range is not None:
                expanse = int(np.sum((z >= z_range[0] - 0.01) & (z <= z_range[1] + 0.01)))
                n_slices, source = max(expanse, GRID), 'DICOM CT + RTSTRUCT'
        return (sx, sy, n_slices * thickness / GRID), source + ' (approx.)'
    except Exception as exc:  # unreadable DICOM: fall back to a nominal spacing
        return DEFAULT_SPACING, 'default ({})'.format(exc)


def load_case(case_or_path, normalize=False, npz_dir=NPZ_DIR, dicom_root=DICOM_ROOT):
    """Load one npz. normalize=True multiplies the dose by DOSE_SCALE (PTV mean = 60 Gy)."""
    path = resolve_npz(case_or_path, npz_dir)
    case_id = os.path.splitext(os.path.basename(path))[0]
    d = np.load(path, allow_pickle=True)
    scale = float(d['DOSE_SCALE'])
    dose = d['DOSE'].astype(np.float32)
    if normalize:
        dose = dose * scale
    spacing, source = estimate_spacing(case_id, dicom_root) if dicom_root else (DEFAULT_SPACING, 'default')
    return RTOGCase(case_id=case_id, ct=d['CT'], dose=dose, oar=d['OAR'], ptv=d['PTV'],
                    dose_scale=scale, normalized=normalize,
                    lung_ipsi_side=str(d['LUNG_IPSI_SIDE']) if 'LUNG_IPSI_SIDE' in d.files else '',
                    bins=d['BINS'], hist_clinical=d['HIST_CLINICAL'], hist_sig=d['HIST_SIG'],
                    spacing=tuple(float(s) for s in spacing), spacing_source=source)


def to_sitk(arr, spacing, like=None):
    """(X, Y, Z) npz array -> SimpleITK image (identity direction, origin 0)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(arr, (2, 1, 0))))
    if like is not None:
        img.CopyInformation(like)
    else:
        img.SetSpacing(spacing)
    return img


def dvh_metrics(case):
    """Common RTOG 0617 dose metrics computed from the voxels (Gy and % volume, 128^3 grid)."""
    dose, masks = case.dose, case.masks()
    rows = []

    def dx(m, pct):  # dose received by the hottest pct % of the volume
        return float(np.percentile(dose[m], 100 - pct))

    def vx(m, d):
        return 100.0 * float((dose[m] >= d).mean())

    if 'ptv' in masks:
        m = masks['ptv']
        rows += [('PTV', 'Mean', dose[m].mean(), 'Gy'), ('PTV', 'D95%', dx(m, 95), 'Gy'),
                 ('PTV', 'D2%', dx(m, 2), 'Gy')]
    lungs = masks.get('lung_ipsi', False) | masks.get('lung_cntr', False)
    if np.any(lungs):
        rows += [('Lungs', 'Mean', dose[lungs].mean(), 'Gy'), ('Lungs', 'V20', vx(lungs, 20), '%'),
                 ('Lungs', 'V5', vx(lungs, 5), '%')]
    if 'heart' in masks:
        m = masks['heart']
        rows += [('Heart', 'Mean', dose[m].mean(), 'Gy'), ('Heart', 'V30', vx(m, 30), '%'),
                 ('Heart', 'V5', vx(m, 5), '%')]
    if 'esophagus' in masks:
        rows += [('Esophagus', 'Mean', dose[masks['esophagus']].mean(), 'Gy')]
    if 'cord' in masks:
        rows += [('Spinal cord', 'Max', dose[masks['cord']].max(), 'Gy')]
    return rows

#!/usr/bin/env python
# coding: utf-8
"""
project version: 3D dose prediction for lung RTOG 0617 IMRT plans with a 3D U-Net.
=====================================================================================

Task: given the planning CT, the contours (spinal cord, esophagus, heart, ipsilateral lung, contralateral lung)
and the PTV, a neural network predicts the 3D dose distribution of the clinical plan (knowledge-based planning).

The script is written like a notebook: every "# In[ ]:" block is a cell that can be run on its own in
Spyder / Jupyter, from top to bottom.  Run it from the command line with

    cd /home/Your_Project_dir/KBP_Project
    python test_notebook_rtog_imrt.py                  # full run
    python test_notebook_rtog_imrt.py --epochs 2 --max_cases 20 \
                                                                    --run_name smoke_unet3d      # quick test
    python test_notebook_rtog_imrt.py --model attn_unet3d

Workflow (one "STEP" banner per cell below)
-------------------------------------------
  STEP 0  Imports and CONFIG (everything you may want to change is in the CONFIG cell).
  STEP 1  Data pre-run / quality control (QC): open every npz once and exclude cases that would crash the
          training (missing contours, wrong shapes, NaNs, partial-course plans, duplicates).
  STEP 2  Split the  cases into train / validation / test, by patient, with a fixed random seed.
  STEP 3  Dataset: build the 7 input channels (CT, 5 organs, PTV) and the target dose.
  STEP 4  Models: 3D U-Net or attention 3D U-Net; sanity check (shape, parameters, GPU memory).
  STEP 5  Loss (MAE) and the two  evaluation scores (dose score, DVH score)
  STEP 6  Training loop with validation after every epoch, mlflow logging (loss + both scores), best checkpoint,
          learning curves (loss | dose score | DVH score).
  STEP 7  Testing on the held-out test cases: dose score, DVH score, MAE and clinical DVH-metric errors per case.
  STEP 8  Figures for a few test cases: DVH, dose maps, CT overlays (clinical vs predicted).

Results go to  KBP_Project/results/rtog_imrt_project/<run_name>/  and to mlflow (KBP_Project/mlruns, experiment
"RTOG_IMRT_project"; browse it with `mlflow ui --backend-store-uri file:mlruns`, run from KBP_Project).

Data conventions (see KBP_Project/visualization/rtog_vis_utils.py for the full description)
  CT    raw HU, DOSE raw Gy, OAR uint8 labels (1 cord, 2 esophagus, 3 heart, 4 lung_ipsi, 5 lung_cntr), PTV 0/1,
  all (128, 128, 128) with axes (X = patient right->left, Y = anterior->posterior, Z = inferior->superior).
  DOSE * DOSE_SCALE gives the dose "normalised so that the mean PTV dose is exactly 60 Gy" (the prescription).
"""

# In[ ]:
# =============================================================================================================
# STEP 0 - Imports and CONFIG
# =============================================================================================================
import argparse
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')                     # figures are written to files, no display needed
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

# --- code of this project -------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
sys.path.insert(0, HERE)                                           # so that "MHRG_files" is importable
sys.path.insert(0, os.path.join(HERE, 'visualization'))            # figures + DVH helpers
# Existing model building blocks (we do NOT use the legacy UNet3D2 / DAE / DVH / Moment losses here).
# NOTE: do not import MHRG_files.utils: it imports napari (needs a display).
from MHRG_files.model import UNet3D, Attn_UNet3D, Block, ResizeConv, init_weights, configure_optimizers
from rtog_vis_utils import load_case
import vis_prediction as vp

# --- CONFIG: edit here -----------------------------------------------------------------------------------------
DATA_DIR = '/home/Your_Project_dir/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT'   # STUDENTS: change to the folder with your npz files
RESULTS_ROOT = os.path.join(HERE, 'results', 'rtog_imrt_project')

MODEL_NAME = 'unet3d'        # 'unet3d' | 'attn_unet3d'   <- to try another model, change this
EPOCHS = 100                 # passes over the training set
LR = 5e-4                    # AdamW learning rate (divided by 10 after EPOCHS // 2 epochs, see STEP 6)
# Network settings below apply to MODEL_NAME = 'unet3d' only (STEP 4).  'attn_unet3d' has fixed widths 32..512
# (its feature_scale argument is ignored), uses BatchNorm, and has no dropout layer at all.
UNET_ENC_CHS = (32, 64, 128, 256, 512)   # number of filters (channels) of the 5 encoder levels; the feature maps
                                         # are 128^3, 64^3, 32^3, 16^3, 8^3.  
DROPOUT_P = 0.4                          # dropout probability of the U-Net.  There is exactly ONE dropout layer
                                         # (nn.Dropout3d, drops whole channels), in the deepest encoder block (8^3).
                                         # 0.0 switches it off.
BATCH_SIZE = 1               # one 128^3 volume with 7 channels already needs a lot of GPU memory
SEED = 42                    # makes the split, the weight initialisation and the augmentation reproducible
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / validation / test
N_VIS_CASES = 3              # number of test cases for which figures are drawn in STEP 8
AUGMENT_FLIP_LR = True       # random left-right flip of the training cases (see STEP 3)
NUM_WORKERS = 4              # parallel data-loading processes
MAX_CASES = None             # use only this many cases after QC (quick tests); None = all
RUN_NAME = None              # None -> "<model>_ep<epochs>"
DVH_BACKEND = 'numpy'        # how DVH metrics are computed in STEP 7: 'numpy' (6 ms/case) | 'platipy' (50-250 ms)
SELECT_BEST_BY = 'val_loss'  # which per-epoch validation metric picks the best checkpoint:
                             #   'val_loss' (MAE, all voxels) | 'val_dose_score' | 'val_dvh_score'
MIN_PTV_MEAN_GY = 55.0       # QC: raw PTV mean dose below this = only a part of the course was planned
MLFLOW_URI = 'file:' + os.path.join(HERE, 'mlruns')
MLFLOW_EXPERIMENT = 'RTOG_IMRT_project'
GRID = 128
OAR_LABELS = (1, 2, 3, 4, 5)           # cord, esophagus, heart, lung_ipsi, lung_cntr
CT_HU_RANGE = (-1000.0, 1300.0)        # HU window mapped to 0..1
BODY_HU = -500.0                       # CT above this = inside the body (used by the dose score, STEP 5)

# Command-line overrides.  parse_known_args ignores unknown arguments, so this also works inside Jupyter/Spyder.
_ap = argparse.ArgumentParser()
_ap.add_argument('--epochs', type=int)
_ap.add_argument('--model', choices=['unet3d', 'attn_unet3d'])
_ap.add_argument('--max_cases', type=int)
_ap.add_argument('--run_name')
_ap.add_argument('--lr', type=float)
_ap.add_argument('--dropout', type=float)
_args, _ = _ap.parse_known_args()
if _args.epochs is not None:
    EPOCHS = _args.epochs
if _args.model is not None:
    MODEL_NAME = _args.model
if _args.max_cases is not None:
    MAX_CASES = _args.max_cases
if _args.run_name is not None:
    RUN_NAME = _args.run_name
if _args.lr is not None:
    LR = _args.lr
if _args.dropout is not None:
    DROPOUT_P = _args.dropout
assert 0.0 <= DROPOUT_P < 1.0, DROPOUT_P
assert SELECT_BEST_BY in ('val_loss', 'val_dose_score', 'val_dvh_score'), SELECT_BEST_BY
RUN_NAME = RUN_NAME or '{}_ep{}'.format(MODEL_NAME, EPOCHS)
OUT_DIR = os.path.join(RESULTS_ROOT, RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True            # input size never changes -> let cuDNN pick the fastest kernels
torch.set_float32_matmul_precision('medium')     # same as the original notebook
print('run {} | model {} | epochs {} | device {} | results in {}'.format(RUN_NAME, MODEL_NAME, EPOCHS, device, OUT_DIR))


# In[ ]:
# =============================================================================================================
# STEP 1 - Data pre-run / quality control (QC)
# =============================================================================================================
# Why: one case with an empty contour (e.g. no esophagus) gives NaN DVHs / empty masks, and a plan that only
# covers part of the course (low dose) would teach the network inconsistent doses.  A single bad file
# would otherwise crash the training after hours.  So we open every file ONCE, record simple facts in a table
# (results/rtog_imrt_project/case_qc.csv) and exclude cases by explicit rules.  Read the CSV to see why.
#
# Exclusion rules (first match is shown in the "reason" column, all matches in "all_reasons"):
#   load_error / missing_key   the file cannot be opened or lacks CT, DOSE, OAR, PTV or DOSE_SCALE
#   bad_shape                  an array is not (128, 128, 128)
#   non_finite                 NaN / inf in CT or DOSE
#   empty_structure            any of the 5 organs (labels 1-5) or the PTV has no voxels
#   bad_dose_scale             DOSE_SCALE is not finite or <= 0
#   low_dose_plan              raw mean PTV dose < 55 Gy (partial-phase plan, DOSE_SCALE would be large)
#   duplicate_patient          the patient has several plans (same CT): keep ONE plan per patient, so the same
#                              anatomy never ends up in train and test.  Preferred: the 'total' plan (PlanID
#                              contains "total" in Cases_processing_summary.csv); otherwise the plan with
#                              the highest raw PTV mean dose, then the lowest plan index.
QC_CSV = os.path.join(RESULTS_ROOT, 'case_qc.csv')
EXPECTED_CASES_AFTER_QC = 163       # known result for the full IMRT dataset; only checked when MAX_CASES is None


def qc_one_case(path):
    """Open one npz and record the facts used by the exclusion rules. Never raises."""
    row = {'case_id': os.path.splitext(os.path.basename(path))[0], 'file': os.path.basename(path), 'load_ok': False}
    try:
        d = np.load(path, allow_pickle=True)
        keys = set(d.files)
        row['load_ok'] = True
        missing = [k for k in ('CT', 'DOSE', 'OAR', 'PTV', 'DOSE_SCALE') if k not in keys]
        row['missing_keys'] = ';'.join(missing)
        if missing:
            return row
        ct, dose, oar, ptv = d['CT'], d['DOSE'], d['OAR'], d['PTV']
        row['shapes_ok'] = all(a.shape == (GRID,) * 3 for a in (ct, dose, oar, ptv))
        row['finite_ok'] = bool(np.isfinite(ct).all() and np.isfinite(dose).all())
        for k in OAR_LABELS:
            row['n_oar{}'.format(k)] = int((oar == k).sum())
        row['n_ptv'] = int((ptv > 0).sum())
        scale = float(d['DOSE_SCALE'])
        row['dose_scale'] = scale
        row['ptv_mean_raw'] = float(dose[ptv > 0].mean()) if row['n_ptv'] > 0 and row['shapes_ok'] else np.nan
    except Exception as exc:       # corrupt zip, truncated file, ...
        row['error'] = repr(exc)[:200]
    return row


def run_qc(data_dir, qc_csv):
    """Return the QC table (one row per npz). Raw facts are cached in qc_csv and reused if the file list is the same."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.npz'))
    if os.path.isfile(qc_csv):
        cached = pd.read_csv(qc_csv)
        if sorted(cached['file']) == files and 'ptv_mean_raw' in cached.columns:
            print('QC: reusing cached raw statistics from', qc_csv)
            return cached
    t0 = time.time()
    with mp.get_context('fork').Pool(min(8, os.cpu_count() or 1)) as pool:     # one file per task, reads are independent
        rows = pool.map(qc_one_case, [os.path.join(data_dir, f) for f in files])
    print('QC: opened {} files in {:.1f} s'.format(len(files), time.time() - t0))
    return pd.DataFrame(rows)


def apply_exclusion_rules(qc, data_dir):
    """Add 'reasons' (all rules that fail), 'reason' (first one), 'keep' columns. Handles duplicate patients."""
    qc = qc.copy()
    qc['patient'] = qc['case_id'].str.rsplit('_', n=1).str[0]
    qc['plan_ind'] = qc['case_id'].str.rsplit('_', n=1).str[1].astype(int)
    summary_csv = os.path.join(data_dir, 'Cases_processing_summary.csv')
    qc['is_total_plan'] = False
    if os.path.isfile(summary_csv):
        meta = pd.read_csv(summary_csv)
        total_ids = set((meta['CaseID'] + '_' + meta['PlanInd'].astype(str))[meta['PlanID'].astype(str).str.contains('total')])
        qc['is_total_plan'] = qc['case_id'].isin(total_ids)

    reasons = []
    for _, r in qc.iterrows():
        why = []
        if not r['load_ok'] or isinstance(r.get('error'), str):
            why.append('load_error')
        elif isinstance(r.get('missing_keys'), str) and r['missing_keys']:
            why.append('missing_key')
        else:
            if not r['shapes_ok']:
                why.append('bad_shape')
            if not r['finite_ok']:
                why.append('non_finite')
            if any(r['n_oar{}'.format(k)] == 0 for k in OAR_LABELS) or r['n_ptv'] == 0:
                why.append('empty_structure')
            if not (np.isfinite(r['dose_scale']) and r['dose_scale'] > 0):
                why.append('bad_dose_scale')
            if not r['ptv_mean_raw'] >= MIN_PTV_MEAN_GY:        # also catches NaN
                why.append('low_dose_plan')
        reasons.append(';'.join(why))
    qc['all_reasons'] = reasons

    # one plan per patient among the cases that passed all other rules
    ok = qc[qc['all_reasons'] == '']
    for patient, grp in ok.groupby('patient'):
        if len(grp) > 1:
            best = grp.sort_values(['is_total_plan', 'ptv_mean_raw', 'plan_ind'], ascending=[False, False, True]).index[0]
            for idx in grp.index.difference([best]):
                qc.loc[idx, 'all_reasons'] = 'duplicate_patient (kept {})'.format(qc.loc[best, 'case_id'])
    qc['keep'] = qc['all_reasons'] == ''
    qc['reason'] = qc['all_reasons'].str.split(';').str[0].str.replace(r' \(.*\)', '', regex=True)
    return qc


qc_raw = run_qc(DATA_DIR, QC_CSV)
qc = apply_exclusion_rules(qc_raw, DATA_DIR)
qc.to_csv(QC_CSV, index=False)
print('QC table saved to', QC_CSV)
print('QC: {} files, {} kept, {} excluded'.format(len(qc), int(qc['keep'].sum()), int((~qc['keep']).sum())))
print(qc.loc[~qc['keep'], 'reason'].value_counts().to_string())
for _, r in qc[~qc['keep']].iterrows():
    print('  excluded {:<16s} {}  (PTV mean raw {:.1f} Gy)'.format(r['case_id'], r['all_reasons'], r.get('ptv_mean_raw', np.nan)))
if MAX_CASES is None:
    assert int(qc['keep'].sum()) == EXPECTED_CASES_AFTER_QC, 'expected {} cases after QC'.format(EXPECTED_CASES_AFTER_QC)
    print('QC check OK: {} cases, one per patient'.format(EXPECTED_CASES_AFTER_QC))
assert qc.loc[qc['keep'], 'patient'].is_unique


# In[ ]:
# =============================================================================================================
# STEP 2 - Split into train / validation / test (by patient, fixed seed)
# =============================================================================================================
# train: the network learns from these.  validation: monitored after every epoch to pick the best epoch and detect
# over-fitting.  test: used ONCE at the end for an honest estimate.  We split BY PATIENT (all plans of one patient
# go to the same set); after QC each patient has one plan, so every row is a different patient.
good = qc[qc['keep']].copy()
if MAX_CASES is not None:       # quick tests: random subset (seeded) of the QC-passed cases
    good = good.sample(n=min(MAX_CASES, len(good)), random_state=SEED)


def split_by_patient(table, fractions, seed):
    patients = sorted(table['patient'].unique())
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_val = max(1, int(round(fractions[1] * n)))
    n_test = max(1, int(round(fractions[2] * n)))
    n_train = n - n_val - n_test
    assert n_train >= 1, 'too few cases for a train/val/test split'
    which = {}
    for i, p in enumerate(patients):
        which[p] = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')
    return table['patient'].map(which)


good['split'] = split_by_patient(good, SPLIT_FRACTIONS, SEED)
split_csv = os.path.join(OUT_DIR, 'split.csv')
good[['case_id', 'patient', 'split']].sort_values(['split', 'case_id']).to_csv(split_csv, index=False)
train_ids = good.loc[good.split == 'train', 'case_id'].tolist()
val_ids = good.loc[good.split == 'val', 'case_id'].tolist()
test_ids = good.loc[good.split == 'test', 'case_id'].tolist()
print('split (patients): train {} / val {} / test {} -> {}'.format(len(train_ids), len(val_ids), len(test_ids), split_csv))


# In[ ]:
# =============================================================================================================
# STEP 3 - Dataset: inputs x (7 channels) and target y (dose)
# =============================================================================================================
# x = [CT, cord, esophagus, heart, lung_ipsi, lung_cntr, PTV]            shape (7, 128, 128, 128)
#   * CT is clipped to [-1000, 1300] HU and rescaled to 0..1.  Networks train much better with inputs of order 1
#     than with raw HU values (-1000 ... 3000), which would dominate the 0/1 mask channels.
#   * The 5 organs are one-hot channels (1 where the voxel belongs to the organ), the PTV is a 0/1 channel.
# y = DOSE * DOSE_SCALE                                                   shape (1, 128, 128, 128), in Gy
#   Each plan has a different prescription/normalisation.  Multiplying by DOSE_SCALE puts every plan on the same
#   scale (mean PTV dose = 60 Gy).  The network then only has to learn the SHAPE of the dose (where it is high /
#   low), not the prescription of each patient, which makes the problem much easier.  Predictions are in
#   the same normalised units, so they can be compared directly with clinical dose * DOSE_SCALE.
#
# Axes are NOT permuted: the predicted dose has exactly the same (X, Y, Z) layout as the npz arrays.
#
# Augmentation (training only): random left-right flip along X with probability 0.5.  (The dose is mirrored together with the inputs.)  Set
# AUGMENT_FLIP_LR = False in CONFIG to switch it off.
def npz_path(case_id):
    return os.path.join(DATA_DIR, case_id + '.npz')


class RTOGDoseDataset(Dataset):
    def __init__(self, case_ids, training=False, flip_lr=False):
        self.case_ids = list(case_ids)
        self.training = training
        self.flip_lr = flip_lr

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        d = np.load(npz_path(case_id))
        lo, hi = CT_HU_RANGE
        x = np.empty((1 + len(OAR_LABELS) + 1, GRID, GRID, GRID), dtype=np.float32)
        x[0] = (np.clip(d['CT'], lo, hi) - lo) / (hi - lo)
        oar = d['OAR']
        for i, k in enumerate(OAR_LABELS):
            x[1 + i] = oar == k
        x[-1] = d['PTV'] > 0
        y = (d['DOSE'] * float(d['DOSE_SCALE'])).astype(np.float32)[None]
        if self.training and self.flip_lr and torch.rand(1).item() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]              # flip the X axis (axis 1 after the channel axis)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), case_id


train_ds = RTOGDoseDataset(train_ids, training=True, flip_lr=AUGMENT_FLIP_LR)
val_ds = RTOGDoseDataset(val_ids)
test_ds = RTOGDoseDataset(test_ids)
loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)      # shuffle: new order every epoch
val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)
x0, y0, _ = train_ds[0]
print('input x', tuple(x0.shape), 'target y', tuple(y0.shape), '| target PTV mean {:.1f} Gy'.format(
    float(y0[0][x0[-1] > 0].mean())))


# In[ ]:
# =============================================================================================================
# STEP 4 - Models
# =============================================================================================================
# 'unet3d'      the standard 3D U-Net from MHRG_files/model.py: 5 resolution levels (widths UNET_ENC_CHS,
#               default 32..512 channels), instance normalisation, trilinear up-sampling + convolution, skip
#               connections, ReLU output, one dropout layer (DROPOUT_P) in the deepest encoder block.
# 'attn_unet3d' U-Net with attention gates on the skip connections (Oktay et al.) and deep supervision.
#               (Its `feature_scale` argument is ignored by the original code: the widths are fixed to 32..512.
#               It uses BatchNorm and contains no dropout, so UNET_ENC_CHS and DROPOUT_P do not apply to it.)
# To try another model: change MODEL_NAME in CONFIG (or pass --model on the command line), or add a branch here.
class NonNegative(nn.Module):
    """Wrap a network and apply ReLU to its output (dose cannot be negative). The attention U-Net has no output
    activation of its own; the plain U-Net already ends with a ReLU."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.relu(self.net(x))


def get_model(name, in_ch=7):
    if name == 'unet3d':
        # Widths.  Encoder level i has UNET_ENC_CHS[i] channels.  Every decoder level takes the up-sampled map and
        # concatenates the skip connection of the encoder level with the same resolution, so the decoder cannot
        # have widths of its own: it mirrors the encoder, dec_chs = (e4, e3, e2, e1, e0) = (512, 256, 128, 64, 32)
        # for the default.  Decoder level i is: up-conv (dec[i] -> dec[i+1]), concatenate the skip (dec[i+1]
        # channels), Block (2 * dec[i+1] -> dec[i+1]).
        enc = tuple(UNET_ENC_CHS)
        assert len(enc) == 5 and all(isinstance(c, int) and c > 0 for c in enc), enc
        dec = enc[::-1]
        model = UNet3D(in_ch=in_ch, enc_chs=enc, dec_chs=dec, out_ch=1)
        dec_net = model.decoder
        
        dec_net.upconvs[0] = ResizeConv(dec[0], dec[1], interp='trilinear')
        for i, blk in enumerate(dec_net.dec_blocks):
            if blk.conv1.in_channels != 2 * dec[i + 1]:
                dec_net.dec_blocks[i] = Block(2 * dec[i + 1], dec[i + 1])
        # Dropout.  The original Block_dropout hard-codes p = 0.4.  The only dropout layer of the network sits in
        # the deepest encoder block (8^3 resolution, after its first convolution + ReLU); set its p here.
        for m in model.modules():
            if isinstance(m, (nn.Dropout3d, nn.Dropout)):
                m.p = DROPOUT_P
        init_weights(model, init_type='xavier', init_gain=1)       # same initialisation as the original notebook
    elif name == 'attn_unet3d':
        # It uses BatchNorm: with batch size 1 its running statistics need a few dozen steps to settle, so the
        # validation MAE of the FIRST epoch(s) can be huge in eval mode (seen in 1-epoch smoke tests). This is normal.
        model = NonNegative(Attn_UNet3D(feature_scale=4, out_ch=1, in_ch=in_ch))
    else:
        raise ValueError('unknown model ' + name)
    return model


def check_model(name):
    """Forward + backward of one random volume: output shape, parameter count, peak GPU memory."""
    model = get_model(name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.rand(1, 7, GRID, GRID, GRID, device=device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    out = model(x)
    assert out.shape == (1, 1, GRID, GRID, GRID), out.shape
    assert float(out.detach().min()) >= 0, 'output must be non-negative'
    out.mean().backward()
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == 'cuda' else float('nan')
    print('{}: output {}, {:.2f} M parameters, peak GPU memory {:.1f} GiB (forward+backward, batch 1)'.format(
        name, tuple(out.shape), n_params / 1e6, peak))
    del model, x, out
    if device == 'cuda':
        torch.cuda.empty_cache()
    return n_params, peak


n_params, peak_gib = check_model(MODEL_NAME)
torch.manual_seed(SEED)                  # fresh, reproducible weights for the real model
model = get_model(MODEL_NAME).to(device)


# In[ ]:
# =============================================================================================================
# STEP 5 - Loss and evaluation scores
# =============================================================================================================
# (a) Loss = what the optimiser minimises.  Mean absolute error (L1) between predicted and clinical dose over all
# voxels.  Why L1: it is easy to interpret (average error in Gy) and, unlike MSE, it is not dominated by a few
# high-dose voxels.  Most voxels have low dose (or are air), so this MAE is small and flatters the model.
criterion = nn.L1Loss()


# (b) Evaluation scores = what we MEASURE (every epoch and in STEP 7) but do NOT optimise.  
#   dose score  mean |pred - clinical| over the voxels INSIDE THE BODY.  OpenKBP averages over its "possible dose
#               mask" (voxels where dose can be deposited); the body mask (CT > -500 HU) is our equivalent.  Air
#               around the patient is trivially predicted as 0 Gy and would only dilute the error.
#   DVH score   per case, the mean of |pred - clinical| over 13 DVH metrics that matter clinically:
#                 PTV      D1 (dose to the hottest 1 % of the volume = 99th percentile), D95 (5th percentile),
#                          D99 (1st percentile)
#                 each OAR (cord, esophagus, heart, ipsilateral lung, contralateral lung)   Dmean and Dmax#               
# Both scores are computed from detached tensors under no_grad: the DVH metrics use quantiles/max of masked voxels,
# which are not a useful training signal here, and keeping them out of the graph keeps the extra cost small.
CT_BODY_THRESHOLD = (BODY_HU - CT_HU_RANGE[0]) / (CT_HU_RANGE[1] - CT_HU_RANGE[0])   # BODY_HU on the 0..1 CT scale of channel 0
PTV_QUANTILES = (0.99, 0.05, 0.01)        # D1, D95, D99: "dose received by at least X % of the volume" = (100-X) percentile


@torch.no_grad()
def openkbp_scores(pred, y, x):
    """OpenKBP-style (dose_score, dvh_score) per sample, each a tensor of shape (B,) in Gy.

    pred, y: (B, 1, 128, 128, 128) predicted and clinical dose (same units); x: (B, 7, ...) network input
    (channel 0 = CT on the 0..1 scale, 1-5 = OAR masks, 6 = PTV mask).  Works on GPU tensors.
    """
    pred, y, x = pred.detach().float(), y.detach().float(), x.detach()
    body = (x[:, 0:1] > CT_BODY_THRESHOLD).float()
    dose_score = ((pred - y).abs() * body).sum(dim=(1, 2, 3, 4)) / body.sum(dim=(1, 2, 3, 4)).clamp(min=1)

    q_ptv = torch.tensor(PTV_QUANTILES, device=pred.device)
    dvh_score = torch.full((x.shape[0],), float('nan'), device=pred.device)
    for b in range(x.shape[0]):
        diffs = []                                   # |pred - clinical| of every DVH metric of this case
        p, c = pred[b, 0], y[b, 0]
        ptv = x[b, 6] > 0.5
        if ptv.any():                                # guard: QC already excludes empty structures
            diffs.append((torch.quantile(p[ptv], q_ptv) - torch.quantile(c[ptv], q_ptv)).abs())
        for ch in range(1, 1 + len(OAR_LABELS)):
            organ = x[b, ch] > 0.5
            if organ.any():
                diffs.append(torch.stack([(p[organ].mean() - c[organ].mean()).abs(),            # Dmean
                                          (p[organ].max() - c[organ].max()).abs()]))             # Dmax
        if diffs:
            dvh_score[b] = torch.cat(diffs).mean()
    return dose_score, dvh_score


# In[ ]:
# =============================================================================================================
# STEP 6 - Training
# =============================================================================================================
# Each epoch: (a) train phase - forward, loss, backward, optimiser step for every training case;
#             (b) validation phase - forward only (no gradients), model in eval mode (dropout off).
# The checkpoint with the LOWEST value of the validation metric chosen by SELECT_BEST_BY is kept (default: the
# validation loss; the dose score or the DVH score of STEP 5 can be used instead): later epochs may over-fit.
# Besides the loss, both OpenKBP-style scores are measured on the train and validation cases after every epoch, so
# the learning curves show the quantities the challenge ranks, not only the loss.
# Optimiser: AdamW (weight decay 0.01) via configure_optimizers; StepLR divides the learning rate by 10 after
# half of the epochs, which lets the network settle into a minimum at the end.
def run_epoch(model, loader, optimizer=None):
    """One pass over a loader; trains if an optimiser is given.
    Returns the mean (loss, dose_score, dvh_score) over the loader, all in Gy.

    In the train phase the scores are computed on the output of the SAME forward pass that is used for the loss:
    augmented (flipped) inputs, model in train mode.  They are therefore a little different from the validation
    scores, which use un-augmented inputs and eval mode; that is fine for watching the trend.
    """
    training = optimizer is not None
    model.train(training)
    total, dose_total, dvh_total, n = 0.0, 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if training:
                loss.backward()
                optimizer.step()
            dose_score, dvh_score = openkbp_scores(out.detach(), y, x)       # evaluated, not optimised
            total += loss.item() * x.shape[0]
            dose_total += dose_score.sum().item()
            dvh_total += dvh_score.sum().item()
            n += x.shape[0]
    return total / n, dose_total / n, dvh_total / n


def train_model(model, train_loader, val_loader, epochs, learning_rate, out_dir):
    mlflow.log_params({'model': MODEL_NAME, 'epochs': epochs, 'learning_rate': learning_rate,
                       'batch_size': BATCH_SIZE, 'seed': SEED, 'loss': 'L1 (MAE)', 'select_best_by': SELECT_BEST_BY,
                       'optimizer': 'AdamW wd=0.01',
                       'unet_enc_chs': '-'.join(str(c) for c in UNET_ENC_CHS) if MODEL_NAME == 'unet3d' else 'fixed 32-64-128-256-512',
                       'dropout_p': DROPOUT_P if MODEL_NAME == 'unet3d' else 'none (no dropout layer)',
                       'scheduler': 'StepLR(step=epochs//2, gamma=0.1)', 'flip_lr_augmentation': AUGMENT_FLIP_LR,
                       'n_train': len(train_loader.dataset), 'n_val': len(val_loader.dataset),
                       'n_test': len(test_ids), 'n_params_M': round(n_params / 1e6, 2),
                       'peak_gpu_gib': round(peak_gib, 1), 'dose_units': 'Gy, PTV mean = 60 Gy'})
    optimizer = configure_optimizers(model.parameters(), weight_decay=0.01, learning_rate=learning_rate)
    scheduler = StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)
    cols = ['epoch', 'train_mae', 'val_mae', 'train_dose_score', 'val_dose_score', 'train_dvh_score', 'val_dvh_score',
            'lr', 'seconds']
    hist = {k: [] for k in cols}
    best_value, best_epoch, best_val_loss = float('inf'), 0, float('nan')
    ckpt = os.path.join(out_dir, 'best_model.pt')
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_mae, train_dose, train_dvh = run_epoch(model, train_loader, optimizer)
        val_mae, val_dose, val_dvh = run_epoch(model, val_loader)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        sec = time.time() - t0
        for k, v in zip(cols, (epoch, train_mae, val_mae, train_dose, val_dose, train_dvh, val_dvh, lr, sec)):
            hist[k].append(v)
        for name, v in (('train_loss', train_mae), ('val_loss', val_mae), ('train_dose_score', train_dose),
                        ('val_dose_score', val_dose), ('train_dvh_score', train_dvh), ('val_dvh_score', val_dvh),
                        ('lr', lr)):
            mlflow.log_metric(name, v, step=epoch)
        selected = {'val_loss': val_mae, 'val_dose_score': val_dose, 'val_dvh_score': val_dvh}[SELECT_BEST_BY]
        flag = ''
        if selected < best_value:
            best_value, best_epoch, best_val_loss, flag = selected, epoch, val_mae, '  <- best'
            torch.save(model.state_dict(), ckpt)
        n_cases = len(train_loader.dataset) + len(val_loader.dataset)
        print('Epoch [{}/{}] train MAE {:.3f} | val MAE {:.3f} | val dose score {:.3f} | val DVH score {:.3f} Gy | '
              '{:.1f} s ({:.2f} s/case) | lr {:.1e}{}'.format(
                  epoch, epochs, train_mae, val_mae, val_dose, val_dvh, sec, sec / n_cases, lr, flag))
    pd.DataFrame(hist).to_csv(os.path.join(out_dir, 'history.csv'), index=False)
    mlflow.log_metric('best_val_loss', best_val_loss)               # val loss AT the best epoch (= its minimum with the default selection)
    if SELECT_BEST_BY != 'val_loss':
        mlflow.log_metric('best_' + SELECT_BEST_BY, best_value)     # the selected metric itself (best_val_loss is logged above)
    mlflow.log_metric('best_epoch', best_epoch)
    return hist, best_epoch, ckpt


mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
run = mlflow.start_run(run_name=RUN_NAME)       # closed with mlflow.end_run() at the end of STEP 8
print('mlflow run id:', run.info.run_id)
t_train = time.time()
history, best_epoch, ckpt_path = train_model(model, train_loader, val_loader, EPOCHS, LR, OUT_DIR)
print('training took {:.1f} min; best epoch {} by {} (val MAE {:.3f} | dose score {:.3f} | DVH score {:.3f} Gy)'.format(
    (time.time() - t_train) / 60, best_epoch, SELECT_BEST_BY, history['val_mae'][best_epoch - 1],
    history['val_dose_score'][best_epoch - 1], history['val_dvh_score'][best_epoch - 1]))
mlflow.log_artifact(ckpt_path)
curves_png = vp.plot_training_curves(history, os.path.join(OUT_DIR, 'learning_curves.png'), best_epoch=best_epoch,
                                     select_by=SELECT_BEST_BY)        # 3 panels: loss | dose score | DVH score
mlflow.log_artifact(curves_png)


# In[ ]:
# =============================================================================================================
# STEP 7 - Testing on the held-out test cases
# =============================================================================================================
# Reload the BEST checkpoint (not the last epoch) and predict every test case.  Per case we report
#   dose_score  OpenKBP dose score: mean |pred - clinical| inside the body (CT > -500 HU), see STEP 5
#   DVH_score   OpenKBP-style DVH score: mean |pred - clinical| over 13 DVH metrics (PTV D1/D95/D99, Dmean and
#               Dmax of 5 organs), see STEP 5.  Both come from openkbp_scores, the SAME function as in the training
#               loop, applied to the un-augmented input of RTOGDoseDataset, so the numbers are comparable.
#   MAE_all     mean |pred - clinical| over the whole 128^3 volume (the training loss)
#   MAE_ptv     same, only inside the PTV            (accuracy where the dose is highest)
#   DVH metric differences (predicted - clinical): PTV Mean/D95/D2, Lungs Mean/V20/V5, Heart Mean/V30/V5,
#   Esophagus Mean, cord Max - the quantities that matter clinically for RTOG 0617 plans (rtog_vis_utils.dvh_metrics).
# All doses are in the normalised units (PTV mean 60 Gy).  The clinical DVH metrics are computed here, after
# training, with fast numpy/platipy code (DVH_BACKEND in CONFIG); during training only the two scores are tracked.
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()


@torch.no_grad()
def predict_case(model, case_id):
    """Predicted dose (128, 128, 128) numpy array, same axis order as the npz."""
    x, _, _ = RTOGDoseDataset([case_id])[0]
    return model(x[None].to(device))[0, 0].float().cpu().numpy()


@torch.no_grad()
def predict_with_scores(model, case_id):
    """Predicted dose (numpy, as predict_case) plus the OpenKBP scores (dose_score, dvh_score) of that case."""
    x, y, _ = RTOGDoseDataset([case_id])[0]                  # un-flipped: training=False
    x, y = x[None].to(device), y[None].to(device)
    out = model(x)
    dose_score, dvh_score = openkbp_scores(out, y, x)
    return out[0, 0].float().cpu().numpy(), dose_score.item(), dvh_score.item()


test_rows = []
t_test = time.time()
for case_id in test_ids:
    case = load_case(case_id, normalize=True, npz_dir=DATA_DIR, dicom_root=None)       # clinical dose x DOSE_SCALE = same units as pred
    pred, dose_score, dvh_score = predict_with_scores(model, case_id)
    err = np.abs(pred - case.dose)
    row = {'case_id': case_id, 'dose_score': dose_score, 'DVH_score': dvh_score,
           'MAE_all': err.mean(), 'MAE_ptv': err[case.ptv > 0].mean()}
    for (s, m, vc, u), (_, _, vp_, _) in zip(vp.case_metrics(case, None, DVH_BACKEND),
                                             vp.case_metrics(case, pred, DVH_BACKEND)):
        name = '{}_{}'.format(s.replace(' ', ''), m.replace('%', ''))
        row[name + '_clin'], row[name + '_pred'], row[name + '_diff'] = vc, vp_, vp_ - vc
    test_rows.append(row)
    print('  {}: dose score {:.3f} | DVH score {:.3f} | MAE all {:.3f} | PTV {:.3f} Gy'.format(
        case_id, row['dose_score'], row['DVH_score'], row['MAE_all'], row['MAE_ptv']))
test_df = pd.DataFrame(test_rows)
test_csv = os.path.join(OUT_DIR, 'test_metrics.csv')
test_df.to_csv(test_csv, index=False)

summary = {}
for col in ('dose_score', 'DVH_score', 'MAE_all', 'MAE_ptv'):
    summary[col] = (test_df[col].mean(), test_df[col].std(ddof=0))
for col in [c for c in test_df.columns if c.endswith('_diff')]:
    summary[col + '_signed'] = (test_df[col].mean(), test_df[col].std(ddof=0))                # bias of the prediction
    summary[col.replace('_diff', '_absdiff')] = (test_df[col].abs().mean(), test_df[col].abs().std(ddof=0))
summary_df = pd.DataFrame(summary, index=['mean', 'std']).T
summary_csv = os.path.join(OUT_DIR, 'test_summary.csv')
summary_df.to_csv(summary_csv)
print('\nTest results on {} cases (mean +- std; diff = predicted - clinical; Gy or % volume):'.format(len(test_df)))
for name, (mean, std) in summary.items():
    print('  {:<28s} {:8.3f} +- {:.3f}'.format(name, mean, std))
    mlflow.log_metric('test_' + name + '_mean', float(mean))
mlflow.log_artifact(test_csv)
mlflow.log_artifact(summary_csv)
print('inference + metrics: {:.1f} s for {} cases'.format(time.time() - t_test, len(test_ids)))


# In[ ]:
# =============================================================================================================
# STEP 8 - Figures for a few test cases
# =============================================================================================================
# For N_VIS_CASES test cases: DVH comparison (solid = clinical from HIST_CLINICAL, dashed = predicted, computed with
# platipy), dose maps through the PTV centre (clinical | predicted | difference) and the two platipy CT overlays on a
# shared colour scale.  The case is loaded with normalize=True so that the clinical dose is in the same units as the
# prediction.  The 3D napari view is skipped (needs a display).
fig_dir = os.path.join(OUT_DIR, 'figures')
os.makedirs(fig_dir, exist_ok=True)
for case_id in test_ids[:N_VIS_CASES]:
    case = load_case(case_id, normalize=True, npz_dir=DATA_DIR)   # with voxel spacing estimated from DICOM (correct aspect ratio)
    pred = predict_case(model, case_id)
    pngs = [vp.plot_dvh_comparison(case, pred, os.path.join(fig_dir, case_id + '_dvh_comparison.png'),
                                   metrics_backend=DVH_BACKEND),
            vp.plot_dose_comparison(case, pred, os.path.join(fig_dir, case_id + '_dose_comparison.png'))]
    pngs += list(vp.plot_overlay_pair(case, pred, fig_dir))
    for p in pngs:
        mlflow.log_artifact(p, artifact_path='figures/' + case_id)
    print(case_id, '->', ', '.join(os.path.basename(p) for p in pngs))
mlflow.end_run()
print('done. results in', OUT_DIR, '| mlflow run', run.info.run_id)

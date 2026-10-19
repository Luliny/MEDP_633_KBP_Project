# Dose prediction on RTOG 0617 IMRT plans: how to run the workflow

You train a 3D U-Net to predict the dose distribution of a lung IMRT plan from the CT and the contours,
then compare predicted and clinical dose and DVHs. Everything is done by one script,
`test_notebook_rtog_imrt.py`. Read this page once, then run the quick test in Step D.

All commands are run from the `KBP_Project` folder, on the course server:

```bash
cd /home/grp_airtp/AI_RTP/KBP_Project
```

---

## 1. Requirements

| Item | Value (checked on the server) |
|---|---|
| Conda environment | `AI_RTP`, `/home/grp_airtp/anaconda3/envs/AI_RTP` |
| Python | 3.13.5 |
| torch | 2.9.0+cu128 (CUDA 12.8) |
| mlflow | 3.3.2 |
| platipy | 0.7.2 |
| napari | 0.6.6 (only needed for the 3D example figures) |
| SimpleITK | 2.5.2 |
| matplotlib / pandas / numpy | 3.10.0 / 2.3.3 / 2.2.6 |
| Also imported by `MHRG_files/model.py` | torchio 0.20.23, albumentations 2.0.8, lightning 2.5.6 |
| GPU | NVIDIA RTX 4500 Ada, 24 GB, **shared** with other users |
| GPU memory for training | about 11 GiB (the script measures 11.3 GiB for the U-Net and 11.0 GiB for the attention U-Net, batch size 1). Have at least 12 GiB free. |

The training uses the GPU if `torch.cuda.is_available()`, otherwise the CPU. CPU-only was not tested and would be very slow.

Activate the environment (either way works):

```bash
source /home/grp_airtp/anaconda3/etc/profile.d/conda.sh && conda activate AI_RTP
python --version        # Python 3.13.5
```

or call `/home/grp_airtp/anaconda3/envs/AI_RTP/bin/python` directly. The examples below write `python`, meaning the `AI_RTP` one.

Do not install or upgrade packages in `AI_RTP`.

## 2. Project layout

```
KBP_Project/
|-- test_notebook_rtog_imrt.py     <- the training script (you work here)
|-- RTOG_IMRT_WORKFLOW.md          <- this page
|-- MLFLOW_GUIDE.md                <- monitor and compare runs
|-- mlflow_tools.py                <- terminal tool for mlflow runs (list / watch / compare / artifacts)
|-- IMRT_identification_findings.md  <- how the IMRT plans were identified
|-- MHRG_files/model.py            <- UNet3D, Attn_UNet3D, optimiser helper (original code, do not edit)
|-- preprocess/identify_imrt_plans.py   <- IMRT / 3D-CRT classification (instructor)
|-- visualization/
|   |-- make_visualization_examples.py  <- inspect the data (overlay, DVH, 3D)
|   |-- rtog_vis_utils.py          <- load_case(), structure table, dose metrics
|   |-- vis_prediction.py          <- predicted-vs-clinical figures, learning curves (3 panels)
|   |-- vis_dvh.py, vis_platipy_overlay.py, vis_napari_3d.py
|-- results/
|   |-- imrt_identification/       <- rtplan_imrt_classification.csv, rtplan_beams.csv
|   |-- visualization_examples/    <- figures of Step B
|   `-- rtog_imrt_teaching/        <- everything the training script writes
|       |-- case_qc.csv            <- QC table (Step E)
|       |-- <run_name>/            <- one folder per run (Step G)
|       |-- <run_name>.log         <- console output, if you redirect it (Step D)
|       `-- compare_runs.csv/.png  <- from `mlflow_tools.py compare`
`-- mlruns/                        <- mlflow store (experiment RTOG_IMRT_teaching)

Data (outside the project):
/home/grp_airtp/AI_RTP/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT/   179 npz files, 3.6 GB
```

---

## 3. Step by step

### Step A (instructor only): from DICOM to npz

Students normally **skip this step** and start from the provided npz files. It needs the raw DICOM
(`/mnt/data1/RTOG_DICOM_data/...`), a GPU and hours of run time.

**A1. Identify the IMRT plans** (about 1.5 min). The defaults of every argument are the server paths, so this is enough:

```bash
python preprocess/identify_imrt_plans.py --out_dir results/imrt_identification
```

Arguments (from `--help`): `--dicom_root`, `--plan_map` (CSV that maps plans to npz names), `--npz_dir`,
`--out_dir` (default `./results/imrt_identification`), `--no_dose` (skip the dose-based check, faster).
It writes `rtplan_imrt_classification.csv` (one row per plan) and `rtplan_beams.csv` (one row per beam).
Re-running it into a scratch folder gave the same label and confidence for all 568 plans as the stored CSV. Method and results: `IMRT_identification_findings.md`. In short, the RTOG RT Plans contain no MLC data, so
the technique is inferred from beam names, wedges and field arrangement.

**A2. Convert DICOM to npz.** The script is `/home/grp_airtp/AI_RTP/VCU_CLEAN/preprocess/data_preprocess_reduction_rtog_gpu_new.py`
(not the older `..._doseRTX_rtog_gpu.py`). It has **no command-line arguments**, and its `__main__` block is set up for a different
run (all plans, other output folder). Call the function from Python instead, with an **empty new output folder**:

```bash
cd /home/grp_airtp/AI_RTP/VCU_CLEAN/preprocess
QT_QPA_PLATFORM=offscreen python - <<'EOF'
import data_preprocess_reduction_rtog_gpu_new as dp
dp.batch_process_cases(dp.DICOM_folder_rtog, '/path/to/NEW_output_folder',
                       case_list=None,            # or e.g. ['0617-259694'] for a test
                       imrt_only=True, min_imrt_confidence='medium')
EOF
```

- `imrt_only=True, min_imrt_confidence='medium'` keeps the plans labelled IMRT with confidence medium or high: 179 plans.
- The function rewrites `Cases_processing_summary.csv` in the output folder, so never point it at an existing finished folder.
- Checked: the module imports and the signature is as above. The conversion itself was not re-run for this guide.
- What it does: crops around lungs, heart and PTV, resamples to 128 x 128 x 128, labels the structures, computes the DVHs, and saves `DOSE_SCALE` (details in the instructor report).

The npz keys:

| Key | Content |
|---|---|
| `CT`, `DOSE` | (128,128,128) float32, raw HU and raw Gy (dose clipped at 85 Gy) |
| `OAR` | uint8 labels: 1 cord, 2 esophagus, 3 heart, 4 lung_ipsi, 5 lung_cntr |
| `PTV` | uint8 0/1 |
| `DOSE_SCALE` | `DOSE[PTV>0].mean() * DOSE_SCALE = 60 Gy` |
| `HIST_CLINICAL`, `HIST_SIG`, `BINS` | (376, 8) DVHs (volume fraction), dose bins (0-80 Gy for `HIST_CLINICAL`; `HIST_SIG` used 0-75 Gy) |
| `LUNG_IPSI_SIDE` | `'L'` or `'R'` |

Axes are (X right to left, Y anterior to posterior, Z inferior to superior). Columns 5 and 6 of the DVH arrays are always NaN (unused labels).

### Step B: look at the data

```bash
python visualization/make_visualization_examples.py --normalize --skip_3d
python visualization/make_visualization_examples.py 0617-259694_0 0617-634370_0 --normalize --skip_3d
python visualization/make_visualization_examples.py 0617-259694_0               # with the 3D napari view (needs a display)
```

| Flag | Meaning |
|---|---|
| `cases` (positional) | case ids such as `0617-259694_0`, or paths to npz files. Default: 4 example cases. |
| `--normalize` | multiply the dose by `DOSE_SCALE` (PTV mean = 60 Gy). This is the dose the network is trained on. Output names get the suffix `_norm`. |
| `--skip_3d` | no napari 3D figure. **Use it without a display**: without `DISPLAY`, napari aborts the whole process (checked), after the 2D figures are written. |
| `--isodose 40` | add an isodose surface (Gy) to the 3D view |
| `--out_dir` | output folder, default `results/visualization_examples/` |

Output per case: `<case>_overlay[_norm].png` (CT + dose + contours, three views), `<case>_dvh[_norm].png` (DVHs + metric table),
`<case>_napari3d[_norm].png` (3D surfaces). Existing examples are already in `results/visualization_examples/`.
Voxel spacing is estimated from the DICOM if present, otherwise 2.5 mm; this only changes the aspect ratio of the figures.

### Step C: configure `test_notebook_rtog_imrt.py`

Open the file and find the block `# --- CONFIG: edit here` (STEP 0). Defaults:

| Name | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `/home/grp_airtp/AI_RTP/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT` | Folder with the npz files. **Absolute path: set it to your copy of the data** if you are not using this server folder. |
| `RESULTS_ROOT` | `<script folder>/results/rtog_imrt_teaching` | Where results are written. |
| `MODEL_NAME` | `'unet3d'` | `'unet3d'` (standard 3D U-Net) or `'attn_unet3d'` (U-Net with attention gates). |
| `EPOCHS` | `100` | Passes over the training set. |
| `LR` | `5e-4` | AdamW learning rate. It is divided by 10 after `EPOCHS // 2` epochs (StepLR). |
| `UNET_ENC_CHS` | `(32, 64, 128, 256, 512)` | **`'unet3d'` only.** Number of filters (channels) of the 5 encoder levels, at resolutions 128, 64, 32, 16 and 8 cubed. The decoder widths are not a separate setting: each decoder level concatenates the skip connection of the encoder level with the same resolution, so the decoder mirrors the encoder (for the default `(512, 256, 128, 64, 32)`). Any 5 positive integers work, they need not double. Fewer filters mean fewer parameters and less GPU memory (measured below). `'attn_unet3d'` has fixed widths 32 to 512 (its `feature_scale` argument is ignored) and is not affected. |
| `DROPOUT_P` | `0.4` | **`'unet3d'` only.** Dropout probability. The network has exactly one dropout layer (`nn.Dropout3d`, drops whole channels), in the deepest encoder block at 8 cubed resolution, active in the training pass only. `0.0` switches it off. `'attn_unet3d'` has no dropout layer (it uses BatchNorm). |
| `BATCH_SIZE` | `1` | One 128 cubed volume with 7 channels already needs about 11 GiB. Do not raise it without checking the GPU memory. |
| `SEED` | `42` | Seeds the split, the weight initialisation and the flip augmentation. |
| `SPLIT_FRACTIONS` | `(0.70, 0.15, 0.15)` | Train / validation / test, split by patient. With 163 cases this gives 115 / 24 / 24. |
| `N_VIS_CASES` | `3` | Number of test cases drawn in STEP 8 (the first ones, sorted by id). |
| `AUGMENT_FLIP_LR` | `True` | Random left-right flip of the training cases. Valid because the lungs are labelled ipsilateral / contralateral. |
| `NUM_WORKERS` | `4` | Data-loading processes. |
| `MAX_CASES` | `None` | Use only this many cases after QC (random, seeded). For quick tests. `None` = all. The "163 cases" check is skipped when this is set. |
| `RUN_NAME` | `None` | Name of the run. `None` gives `<model>_ep<epochs>`, for example `unet3d_ep100`. |
| `DVH_BACKEND` | `'numpy'` | How the DVH metrics are computed in STEP 7: `'numpy'` (about 6 ms per case) or `'platipy'` (about 50-250 ms). In the DVH figures the predicted (dashed) curves always use platipy; the clinical (solid) curves are the stored `HIST_CLINICAL`. |
| `MIN_PTV_MEAN_GY` | `55.0` | QC: a plan whose raw PTV mean dose is below this only covers part of the course, and is excluded. |
| `BODY_HU` | `-500.0` | CT value (HU) above which a voxel counts as inside the body. Used only by the dose score (STEP 5, see "Evaluation scores"). |
| `SELECT_BEST_BY` | `'val_loss'` | Which validation number picks the best checkpoint: `'val_loss'` (MAE over all voxels), `'val_dose_score'` or `'val_dvh_score'`. Logged to mlflow as the param `select_best_by`. |

Further settings in the same block (usually left alone): `MLFLOW_URI`, `MLFLOW_EXPERIMENT` (`RTOG_IMRT_teaching`), `GRID` (128), `OAR_LABELS`, `CT_HU_RANGE` (-1000 to 1300 HU, scaled to 0-1). The script stops at start-up if `SELECT_BEST_BY` is not one of the three allowed names.

**Command-line overrides.** Six exist: `--epochs`, `--model` (`unet3d` or `attn_unet3d`), `--max_cases`, `--run_name`, `--lr`, `--dropout`. They override the CONFIG values. For example `python -u test_notebook_rtog_imrt.py --lr 1e-3 --dropout 0.2 --run_name unet_lr1e-3_do0.2`.
Unknown arguments are ignored (`parse_known_args`). Everything else (for example `UNET_ENC_CHS`) is changed by editing CONFIG.

Measured with the script's own check (forward + backward, batch 1, GPU), for the three width settings tried when the knobs were added:

| `UNET_ENC_CHS` | Parameters | Peak GPU memory |
|---|---|---|
| `(32, 64, 128, 256, 512)` (default) | 25.89 M | 11.3 GiB |
| `(16, 32, 64, 128, 256)` | 6.47 M | 3.4 GiB |
| `(32, 64, 128, 256, 256)` | 17.04 M | 4.7 GiB |

The default is unchanged, so the existing 10-epoch results stay valid (the stored `best_model.pt` of `unet3d_ep10` still loads into the default network). A checkpoint only loads into a network with the same `UNET_ENC_CHS`; `DROPOUT_P` does not change the weights.

**Running it as cells.** The script is split into cells marked `# In[ ]:` (STEP 0 to STEP 8). Run the cells from top to bottom:

- **Spyder:** `# In[` is recognised as a cell marker (checked in the installed Spyder 6.0.7 source). Run a cell with Ctrl+Enter.
- **VS Code:** the Python extension's native cell marker is `# %%`. Whether `# In[ ]:` is also recognised was not tested; if not, replace the markers with `# %%`.
- **Jupyter / IPython:** to run the whole script, use `%run test_notebook_rtog_imrt.py --epochs 1 --max_cases 12 --run_name quick_test`. To run it cell by cell, paste the cells in order into a notebook. Start the kernel **in `KBP_Project`**: the script uses its own folder when `__file__` exists, and the current folder otherwise.
- In all three, edit CONFIG values directly in the file (the command-line flags are not available for a cell-by-cell run).

### Step D: run it

Quick test (1 epoch, 12 cases, about one minute). This exact command was run with the current script and finishes without errors:

```bash
python -u test_notebook_rtog_imrt.py --epochs 1 --max_cases 12 --run_name quick_test
```

It makes a run folder `results/rtog_imrt_teaching/quick_test/` and an mlflow run. Delete both afterwards if you do not want them.
With so few cases the numbers are meaningless. With `--model attn_unet3d` in such a tiny test the validation MAE of the first epoch can be huge
(102.9 Gy was seen in a 1-epoch smoke run): the BatchNorm statistics have not settled with batch size 1. In the full 10-epoch run the first epoch was already 1.950 Gy.

Full runs, in the background, with a log. Create the log folder first, then **run the models one after the other**
(each needs about 11 GiB, and the GPU is shared):

```bash
mkdir -p results/rtog_imrt_teaching

nohup python -u test_notebook_rtog_imrt.py --model unet3d      --epochs 100 --run_name unet3d_ep100      > results/rtog_imrt_teaching/unet3d_ep100.log      2>&1 &
# when it has finished:
nohup python -u test_notebook_rtog_imrt.py --model attn_unet3d --epochs 100 --run_name attn_unet3d_ep100 > results/rtog_imrt_teaching/attn_unet3d_ep100.log 2>&1 &
```

Timing (measured in the 10-epoch runs, all 163 cases, including the scores): about 36 s per epoch for the U-Net and 33.5 s for the attention U-Net,
so a 100-epoch run takes about 1 hour (extrapolated, not run). Follow the log:

```bash
tail -f results/rtog_imrt_teaching/unet3d_ep100.log | grep --line-buffered Epoch
nvidia-smi                      # GPU memory in use
pgrep -af test_notebook_rtog_imrt   # is it still running?
```

`-u` (unbuffered output) is needed so that the log fills while the run progresses.

### Step E: what the script does, and what you see

| STEP | What happens | Console output |
|---|---|---|
| 0 | Imports, CONFIG, command-line overrides, seeds | `run <name> \| model ... \| epochs ... \| device cuda \| results in ...` |
| 1 | **QC.** Opens each npz once, writes `results/rtog_imrt_teaching/case_qc.csv`, excludes cases by rule: `load_error`, `missing_key`, `bad_shape`, `non_finite`, `empty_structure` (any of the 5 organs or the PTV has no voxels), `bad_dose_scale`, `low_dose_plan` (raw PTV mean < `MIN_PTV_MEAN_GY`), `duplicate_patient` (one plan per patient is kept). The raw statistics are cached in the CSV and reused while the list of files is the same. Asserts 163 cases (only if `MAX_CASES` is `None`). | QC summary, one line per excluded case |
| 2 | **Split** by patient (seeded), writes `<run>/split.csv` | `split (patients): train 115 / val 24 / test 24` |
| 3 | **Dataset**: 7 input channels (CT windowed to 0-1, 5 organ masks, PTV) and target `DOSE * DOSE_SCALE`; random left-right flip for training | `input x (7, 128, 128, 128) target y (1, 128, 128, 128)` |
| 4 | **Model** check: one forward and backward pass of random data (shape, parameters, GPU memory), then the real model | `unet3d: output ..., 25.89 M parameters, peak GPU memory 11.3 GiB` |
| 5 | **Loss and evaluation scores**: the loss is MAE (`nn.L1Loss`). The function `openkbp_scores(pred, y, x)` computes the **dose score** and the **DVH score** (see "Evaluation scores" in Step G). They are measured, not optimised. | none |
| 6 | **Training**: train and validation pass each epoch (the MAE is optimised; both scores are measured on train and validation), logs 7 metrics per epoch to mlflow, keeps the checkpoint with the lowest value of the `SELECT_BEST_BY` metric (default: validation MAE), writes `history.csv` and `learning_curves.png` (3 panels) | one `Epoch [...]` line per epoch |
| 7 | **Testing** on the test cases with the best checkpoint: dose score, DVH score, MAE and DVH metric differences, writes `test_metrics.csv` and `test_summary.csv` | per-case scores, then the summary |
| 8 | **Figures** for `N_VIS_CASES` test cases | one line per case, then `done.` |

Real output of the 10-epoch U-Net run (`results/rtog_imrt_teaching/unet3d_ep10.log`; lines marked `...` are omitted here):

```
run unet3d_ep10 | model unet3d | epochs 10 | device cuda | results in /home/grp_airtp/AI_RTP/KBP_Project/results/rtog_imrt_teaching/unet3d_ep10
QC: reusing cached raw statistics from /home/grp_airtp/AI_RTP/KBP_Project/results/rtog_imrt_teaching/case_qc.csv
QC table saved to /home/grp_airtp/AI_RTP/KBP_Project/results/rtog_imrt_teaching/case_qc.csv
QC: 179 files, 163 kept, 16 excluded
reason
low_dose_plan        12
empty_structure       2
duplicate_patient     2
  excluded 0617-385185_1    low_dose_plan  (PTV mean raw 24.2 Gy)
  excluded 0617-657744_0    empty_structure  (PTV mean raw 66.9 Gy)
  excluded 0617-673893_0    duplicate_patient (kept 0617-673893_2)  (PTV mean raw 76.9 Gy)
  ...
QC check OK: 163 cases, one per patient
split (patients): train 115 / val 24 / test 24 -> .../results/rtog_imrt_teaching/unet3d_ep10/split.csv
input x (7, 128, 128, 128) target y (1, 128, 128, 128) | target PTV mean 60.0 Gy
initialize network with xavier
unet3d: output (1, 1, 128, 128, 128), 25.89 M parameters, peak GPU memory 11.3 GiB (forward+backward, batch 1)
initialize network with xavier
mlflow run id: 5fdb2c913cec4c9dad7d5d7fd8f80813
Epoch [1/10] train MAE 3.783 | val MAE 3.232 | val dose score 4.890 | val DVH score 21.758 Gy | 35.9 s (0.26 s/case) | lr 5.0e-04  <- best
Epoch [2/10] train MAE 3.303 | val MAE 2.885 | val dose score 4.343 | val DVH score 18.997 Gy | 35.9 s (0.26 s/case) | lr 5.0e-04  <- best
  ...
Epoch [6/10] train MAE 2.293 | val MAE 1.974 | val dose score 3.048 | val DVH score 8.769 Gy | 36.0 s (0.26 s/case) | lr 5.0e-05  <- best
Epoch [7/10] train MAE 2.225 | val MAE 2.025 | val dose score 3.133 | val DVH score 9.773 Gy | 35.8 s (0.26 s/case) | lr 5.0e-05
  ...
Epoch [10/10] train MAE 2.135 | val MAE 1.843 | val dose score 2.851 | val DVH score 8.674 Gy | 35.9 s (0.26 s/case) | lr 5.0e-05  <- best
training took 6.0 min; best epoch 10 by val_loss (val MAE 1.843 | dose score 2.851 | DVH score 8.674 Gy)
  0617-410508_0: dose score 4.353 | DVH score 12.845 | MAE all 3.153 | PTV 19.503 Gy
  0617-415627_0: dose score 1.283 | DVH score 2.500 | MAE all 1.166 | PTV 2.811 Gy
  ...
Test results on 24 cases (mean +- std; diff = predicted - clinical; Gy or % volume):
  dose_score                      2.748 +- 1.119
  DVH_score                       8.205 +- 3.905
  MAE_all                         1.953 +- 0.805
  MAE_ptv                        12.085 +- 6.709
  PTV_Mean_diff_signed           -5.976 +- 12.348
  PTV_Mean_absdiff               11.693 +- 7.173
  ...
  Spinalcord_Max_absdiff          7.680 +- 5.814
inference + metrics: 3.5 s for 24 cases
0617-410508_0 -> 0617-410508_0_dvh_comparison.png, 0617-410508_0_dose_comparison.png, 0617-410508_0_overlay_clinical.png, 0617-410508_0_overlay_predicted.png
...
done. results in /home/grp_airtp/AI_RTP/KBP_Project/results/rtog_imrt_teaching/unet3d_ep10 | mlflow run 5fdb2c913cec4c9dad7d5d7fd8f80813
```

How to read it: all numbers are in Gy **in normalised units** (PTV mean = 60 Gy), lower is better. "MAE" is the mean absolute error between predicted and clinical dose over the whole volume (this is the loss). "dose score" is the same error inside the body only, and "DVH score" is the mean error of 13 DVH numbers per case (both defined under "Evaluation scores" in Step G). `<- best` marks a new lowest value of the metric chosen by `SELECT_BEST_BY` (default: validation MAE); that checkpoint is saved. The learning rate drops to `5.0e-05` after `EPOCHS // 2` epochs. `train MAE` is computed during the training pass (flipped cases, dropout on), the `val` numbers after it, so they are not directly comparable. A PTV MAE of 12 Gy is bad even though the whole-volume MAE looks small, because most voxels have low dose. The DVH score shows this at once (8.2 Gy for this U-Net, 2.8 Gy for the attention U-Net in the same setting). Always look at the PTV and DVH numbers, not only at the loss.

Neither run has converged after 10 epochs: the U-Net's best epoch is the last one. With `--model attn_unet3d` the log of the attention U-Net contains about 200 lines `initialize network with kaiming` (see Troubleshooting); they are harmless.

On the first run on a new dataset, line 2 reads `QC: opened 179 files in 0.5 s` instead of "reusing cached".

### Step F: monitor and compare with mlflow

Every run is logged to `mlruns/` (experiment `RTOG_IMRT_teaching`): parameters, per epoch `train_loss`, `val_loss`, `train_dose_score`, `val_dose_score`,
`train_dvh_score`, `val_dvh_score` and `lr`, the test metrics and the files of the run. Without a browser:

```bash
python mlflow_tools.py list        # all runs and their status
python mlflow_tools.py watch       # follow the newest run until it ends
python mlflow_tools.py compare     # table + curves of all finished runs
```

`compare` writes `results/rtog_imrt_teaching/compare_runs.csv` and `.png` (use `--csv` and `--out` for other names).
The web UI (`mlflow ui`) does **not** work in `AI_RTP` on this server; how to use it from a separate virtual environment, how to reach it from a laptop and all other options are in **`MLFLOW_GUIDE.md`**.

### Step G: outputs of a run

Folder `results/rtog_imrt_teaching/<run_name>/`:

| File | Content and how to read it |
|---|---|
| `split.csv` | `case_id, patient, split` (train / val / test). The same seed gives the same split for every run. |
| `history.csv` | One row per epoch: `epoch, train_mae, val_mae, train_dose_score, val_dose_score, train_dvh_score, val_dvh_score, lr, seconds` (all in Gy; `train_mae` and `val_mae` are the loss, called `train_loss` / `val_loss` in mlflow). |
| `best_model.pt` | Weights (a `state_dict`) of the epoch with the lowest value of the `SELECT_BEST_BY` metric (default `val_mae`), about 100 MB. Load it as in Step H. |
| `learning_curves.png` | **Three panels** side by side: loss (MAE over all voxels) \| dose score \| DVH score. Each shows train (blue) and validation (red) per epoch, with the best epoch marked by a dotted line and a dot and its validation value written in the corner. The title names the metric that chose the best epoch. How to read it: see below the table. |
| `test_metrics.csv` | One row per test case, columns below. |
| `test_summary.csv` | Mean and standard deviation over the test cases of `dose_score`, `DVH_score`, `MAE_all`, `MAE_ptv` and of the DVH metric differences (see below). |
| `figures/` | Four figures for each of the first `N_VIS_CASES` test cases (below). |

Console log: `results/rtog_imrt_teaching/<run_name>.log` if you redirected it (Step D). Also `results/rtog_imrt_teaching/case_qc.csv` (shared by all runs).

**How to read `learning_curves.png`.**

- All three panels should fall. The loss panel is what the optimiser minimises. The dose-score panel looks similar, shifted up (it ignores the air around the patient). The DVH-score panel is the one to watch for the PTV and organ maxima.
- Validation below training is normal here (train values are averaged over the epoch, with flipped cases and dropout on). A curve that is still falling at the last epoch means the model needs more epochs. Over-fitting would show as a training curve that keeps falling while the validation curve flattens or rises.
- The kink in the middle is the learning-rate drop after `EPOCHS // 2` epochs.
- The DVH score is noisier than the other two, because it depends on quantiles and maxima of few voxels. A jump in one epoch (the attention U-Net run has one at epoch 4) is not necessarily a real change; look at the neighbouring epochs. If the loss and the DVH score disagree about the best epoch, try `SELECT_BEST_BY = 'val_dvh_score'` (Exercises). In the two 10-epoch runs all three validation numbers have their minimum at the same epoch (10 for the U-Net, 9 for the attention U-Net).

**`test_metrics.csv` columns.** `case_id`, then:

- `dose_score`: mean absolute error inside the body (CT > `BODY_HU`); `DVH_score`: mean absolute error of 13 DVH metrics (both defined below); `MAE_all`: mean absolute error over the whole volume (the loss); `MAE_ptv`: inside the PTV. In Gy. (`dose_score` replaces the column `MAE_body` of the earlier version of the script; the body mask is the same.)
- For each metric `<Structure>_<Metric>`: `_clin` (clinical value), `_pred` (predicted value) and `_diff` = **predicted minus clinical**.
  Negative = the network under-doses. Example: `PTV_D95_diff = PTV_D95_pred - PTV_D95_clin`.
- Metrics (structure name without spaces, `%` removed): `PTV_Mean`, `PTV_D95`, `PTV_D2` (Gy); `Lungs_Mean` (Gy), `Lungs_V20`, `Lungs_V5` (% of lung volume);
  `Heart_Mean` (Gy), `Heart_V30`, `Heart_V5` (%); `Esophagus_Mean` (Gy); `Spinalcord_Max` (Gy). "Lungs" is ipsilateral plus contralateral lung.
  `D95` is the dose received by 95% of the volume, `V20` the percentage of the volume receiving at least 20 Gy.
- All doses are in the normalised units (PTV mean 60 Gy). Divide by the case's `DOSE_SCALE` to get the plan's own units.

**`test_summary.csv`.** For each `*_diff` column there are two rows: `<name>_diff_signed` (mean of the signed difference = the bias) and `<name>_absdiff` (mean of the absolute difference = the typical error); plus `dose_score`, `DVH_score`, `MAE_all`, `MAE_ptv` (26 rows in all). Columns: mean, std. mlflow logs the means as `test_<name>_mean`, for example `test_dose_score_mean` and `test_DVH_score_mean`.

#### Evaluation scores: dose score and DVH score

The loss (MAE over all 128 cubed voxels) is dominated by low-dose and air voxels and flatters the model. For scale: predicting zero dose everywhere gives a loss of 5.238 Gy on the 24 test cases, a dose score of 8.144 Gy and a DVH score of 40.672 Gy (computed with a scratch script with the same definitions, not part of the project). The trained models reach 1.5 to 2.0 on the loss but 2.8 to 8.2 on the DVH score, so the DVH score is the more demanding number. The script therefore also reports two numbers in the style of the OpenKBP challenge (Babier et al., Med Phys 2021), which ranks dose predictions by a dose score and a DVH score. Both are in Gy (normalised units), lower is better, and both come from one function, `openkbp_scores`, in STEP 5. The same function is used every epoch (train and validation) and in the test (STEP 7).

| Score | Definition |
|---|---|
| **Dose score** | Mean of \|predicted - clinical dose\| over the voxels **inside the body** (CT > `BODY_HU` = -500 HU). |
| **DVH score** | Per case, the mean of \|predicted - clinical\| over **13 DVH numbers** (below), in Gy. Then the mean over the cases. |

The 13 DVH numbers are computed from the voxels of each structure only:

| Structure | Numbers | Meaning |
|---|---|---|
| PTV | D1, D95, D99 | **D95** = the dose that 95% of the PTV volume receives at least; in voxel terms the 5th percentile of the PTV voxel doses (the near-minimum dose, "is the target covered?"). **D99** = the dose that 99% of the volume receives at least (1st percentile, even closer to the minimum). **D1** = the dose that only the hottest 1% of the volume reaches (99th percentile, "how hot is the hot spot?"). |
| cord, esophagus, heart, ipsilateral lung, contralateral lung (5 organs) | Dmean, Dmax | **Dmean** = mean dose of the organ, **Dmax** = highest voxel dose in the organ. 2 numbers x 5 organs = 10. |

3 PTV numbers + 10 organ numbers = 13. A perfect prediction gives 0 for both scores (checked: with the clinical dose as prediction both scores are 0).

Differences from OpenKBP (the scores are "OpenKBP-style", not the official ones):

1. **Body mask instead of the "possible dose mask".** OpenKBP averages over the voxels where dose can be deposited. We have no such mask, so we use the body (CT > -500 HU), which is our equivalent.
2. **Dmax instead of D_0.1cc.** OpenKBP uses the dose to the hottest 0.1 cc of an organ, which is robust against one hot voxel. A volume in cc needs the voxel size, and our npz files carry none, so we use the plain maximum voxel dose. It is more sensitive to single voxels, which makes the DVH score noisier.
3. **Our own structure list for lung plans.** The five organs and the PTV are the RTOG 0617 structures in our npz files; OpenKBP was built for head-and-neck plans with a different list.
4. **Normalised units.** All doses have PTV mean = 60 Gy, so the values cannot be compared with published OpenKBP numbers.

Two things that are easy to misread:

- The PTV D95, D99 and D1 here are percentiles of the voxels **inside the PTV**. (The older function `compute_eval_metrics` in `MHRG_files/model.py` takes quantiles over the whole volume including the zeros outside the PTV, which gives about 0 for D95; it is not used.)
- The train scores are computed on the output of the training pass (flipped cases, train mode), the validation and test scores on un-flipped cases in eval mode.

**Why the scores are tracked but not optimised.** The loss stays the plain MAE. The DVH score is built from quantiles and maxima of masked voxels: a handful of voxels decide each number, so as a training signal it is sparse and noisy. `openkbp_scores` is therefore computed on detached tensors under `torch.no_grad()`, purely as a measurement; adding a DVH-type loss term is one of the exercises (section 4). Tracking the scores costs little: about 0.15 s per epoch (0.4%), measured when they were added. The DVH metrics in `test_metrics.csv` (PTV Mean/D95/D2, V20, ...) are a different list of clinical metrics, computed after training; only PTV D95 is in both.

Sanity checks done when the scores were written: the torch function agrees with an independent numpy implementation to 4 decimals on 3 cases, and the Dmean/PTV numbers agree with `rtog_vis_utils.dvh_metrics` on a clinical dose (case 0617-634370_0: PTV D95 difference 0, mean differences at most 4e-6 Gy, cord maximum difference 0).


**Figures** in `figures/`, per test case `<case>`:

| File | Shows |
|---|---|
| `<case>_dvh_comparison.png` | Clinical (solid) vs predicted (dashed) cumulative DVH of each structure, and a table of the dose metrics with the difference |
| `<case>_dose_comparison.png` | Axial, coronal and sagittal slices through the PTV centre: clinical dose, predicted dose, predicted minus clinical |
| `<case>_overlay_clinical.png` | CT + clinical dose + contours (platipy), same colour scale as the next one |
| `<case>_overlay_predicted.png` | CT + predicted dose + contours |

The same figures are stored in mlflow under `figures/<case>/`.

### Step H: reuse a trained model

Importing `test_notebook_rtog_imrt.py` runs the whole training, so do not import it. The snippet below rebuilds the model
from `MHRG_files/model.py` (same architecture and patch as STEP 4), loads `best_model.pt` and predicts one case.
Tested with `unet3d_ep10` and `attn_unet3d_ep10` (run again after the two models were re-trained): the MAE over the volume for case `0617-415627_0` matched the `MAE all` of the logs (1.166 and 0.989 Gy). It works from any folder.

```python
import sys
import numpy as np
import torch
import torch.nn as nn

KBP = '/home/grp_airtp/AI_RTP/KBP_Project'
sys.path.insert(0, KBP)                                  # makes MHRG_files importable
from MHRG_files.model import UNet3D, Attn_UNet3D, Block, ResizeConv   # does not import napari

DATA_DIR = '/home/grp_airtp/AI_RTP/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT'


class NonNegative(nn.Module):                            # same wrapper as in STEP 4
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.relu(self.net(x))


ENC_CHS = (32, 64, 128, 256, 512)                        # = UNET_ENC_CHS used for the run; change it if you changed CONFIG


def get_model(name, in_ch=7, enc=ENC_CHS):               # same architecture as STEP 4 (no weight init needed)
    if name == 'unet3d':
        dec = tuple(enc)[::-1]                           # the decoder mirrors the encoder
        model = UNet3D(in_ch=in_ch, enc_chs=tuple(enc), dec_chs=dec, out_ch=1)
        model.decoder.upconvs[0] = ResizeConv(dec[0], dec[1], interp='trilinear')   # fix of the hard-coded 515 channels
        for i, blk in enumerate(model.decoder.dec_blocks):
            if blk.conv1.in_channels != 2 * dec[i + 1]:                             # only for non-doubling widths
                model.decoder.dec_blocks[i] = Block(2 * dec[i + 1], dec[i + 1])
        return model                                     # dropout needs no setting here: it is inactive in eval mode
    if name == 'attn_unet3d':
        return NonNegative(Attn_UNet3D(feature_scale=4, out_ch=1, in_ch=in_ch))
    raise ValueError(name)


def build_input(case_id, data_dir=DATA_DIR):             # same 7 channels as STEP 3
    d = np.load('{}/{}.npz'.format(data_dir, case_id))
    lo, hi = -1000.0, 1300.0
    x = np.empty((7, 128, 128, 128), dtype=np.float32)
    x[0] = (np.clip(d['CT'], lo, hi) - lo) / (hi - lo)
    for i, k in enumerate((1, 2, 3, 4, 5)):
        x[1 + i] = d['OAR'] == k
    x[6] = d['PTV'] > 0
    clinical = (d['DOSE'] * float(d['DOSE_SCALE'])).astype(np.float32)      # Gy, PTV mean = 60 Gy
    return torch.from_numpy(x)[None], clinical, d['PTV'] > 0, float(d['DOSE_SCALE'])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
run, model_name, case_id = 'unet3d_ep10', 'unet3d', '0617-415627_0'      # or 'attn_unet3d_ep10', 'attn_unet3d'
model = get_model(model_name).to(device)
model.load_state_dict(torch.load('{}/results/rtog_imrt_teaching/{}/best_model.pt'.format(KBP, run), map_location=device))
model.eval()

x, clinical, ptv, scale = build_input(case_id)
with torch.no_grad():
    pred = model(x.to(device))[0, 0].cpu().numpy()       # (128, 128, 128), same axes as the npz, normalised Gy
print('predicted PTV mean {:.1f} Gy | clinical {:.1f} Gy | MAE over volume {:.3f} Gy | DOSE_SCALE {:.3f}'.format(
    pred[ptv].mean(), clinical[ptv].mean(), np.abs(pred - clinical).mean(), scale))
```

Expected output (real output of the snippet; the last digit can differ by a hair because GPU kernels are not bit-exact). For `unet3d_ep10`:
`predicted PTV mean 61.1 Gy | clinical 60.0 Gy | MAE over volume 1.166 Gy | DOSE_SCALE 0.957`.
For `attn_unet3d_ep10` (set `run, model_name = 'attn_unet3d_ep10', 'attn_unet3d'`) the last line is
`predicted PTV mean 57.2 Gy | clinical 60.0 Gy | MAE over volume 0.989 Gy | DOSE_SCALE 0.957`, preceded by about 100 lines `initialize network with kaiming`.

Notes:

- `pred` is in normalised units. `pred / scale` is the dose in the plan's own units.
- `pred` has the npz axis order. Use `pred` with `visualization/vis_prediction.py` (`load_case(case_id, normalize=True)`, then `plot_dvh_comparison(case, pred, 'out.png')`) to draw figures.
- If you change the model in `test_notebook_rtog_imrt.py` (`get_model`, or `UNET_ENC_CHS` in CONFIG), change the copy in your snippet too (`ENC_CHS`), otherwise `load_state_dict` fails. `DROPOUT_P` does not matter for prediction: dropout is off in `model.eval()`.
- Building `Attn_UNet3D` prints `initialize network with kaiming` many times (102 times in this snippet; 204 in the training log, because the script builds the model twice: the check in STEP 4 and the real model). The lines come from the original model code and are harmless noise.

---

## 4. Exercises: how to modify

Line numbers refer to the current file; search for the names if they move.

| Goal | What to do |
|---|---|
| **Change the model** | Run with `--model attn_unet3d`, or add a new `elif name == '...'` branch in `get_model` (STEP 4, about line 388) returning an `nn.Module` that maps (1, 7, 128, 128, 128) to (1, 1, 128, 128, 128) with a non-negative output. Add the name to the `--model` choices (about line 113). Do not edit `MHRG_files/model.py`. `UNet3D2` is not usable: it needs latent codes that RTOG does not have. |
| **Add a loss term** | STEP 5 defines `criterion = nn.L1Loss()` (about line 460). The loss is used in `run_epoch` (STEP 6, the two lines `out = model(x)` and `loss = criterion(out, y)`, about lines 537-538), which is shared by training **and** validation. To add a term for training only, replace those two lines by the following (the PTV channel of `x` is channel 6; tested for one epoch on 12 cases with the earlier version of the script, the lines it replaces have not changed). Keep the validation pass as plain MAE so that `val_loss` and the best-checkpoint choice stay comparable. The logged `train_loss` then includes the extra term and is no longer comparable with `val_loss` or with other runs. The two scores are unaffected: they are computed from `out` afterwards. |

```python
            out = model(x)
            loss = criterion(out, y)
            if training:                                    # extra term: MAE inside the PTV only
                loss = loss + 0.1 * (out - y).abs()[x[:, 6:7] > 0].mean()
```

| Goal | What to do |
|---|---|
| **Change the augmentation** | STEP 3, `RTOGDoseDataset.__getitem__` (about lines 348-349): the flip is applied to `x` and `y` together, only when `self.training`. Add other transforms there and always apply the same transform to input and target. Rotations or shifts also need the same transform on `x` and `y`. Do not flip along any axis other than X (the lungs are labelled ipsi / contra, and the table side and gravity are not symmetric). `AUGMENT_FLIP_LR = False` switches the flip off. |
| **Change the number of filters (`unet3d`)** | In CONFIG set `UNET_ENC_CHS`, for example `(16, 32, 64, 128, 256)` (a quarter of the parameters, 3.4 GiB instead of 11.3 GiB) or `(48, 96, 192, 384, 768)` (more capacity, more memory: check `nvidia-smi` first). Five integers, any values; the decoder follows automatically (`get_model`, STEP 4). The model check at the start prints the parameters and the peak GPU memory. Use `--run_name` so that the run is easy to find. Not available for `attn_unet3d`. A `best_model.pt` only loads into the same widths (Step H). |
| **Change the dropout rate (`unet3d`)** | `DROPOUT_P` in CONFIG, or `--dropout 0.2` on the command line. `0.0` removes the dropout. There is one dropout layer only, in the deepest encoder block, so the effect is small and mostly shows in the gap between train and validation curves. `attn_unet3d` has no dropout. |
| **Change the learning rate** | `LR` in CONFIG, or `--lr 1e-3`. The StepLR drop after `EPOCHS // 2` epochs stays. |
| **Change the QC threshold** | Change `MIN_PTV_MEAN_GY` in CONFIG. The rules are re-applied at every run (only the raw statistics are cached in `case_qc.csv`). The assertion `EXPECTED_CASES_AFTER_QC = 163` (about line 167) then fails: set it to the number the QC prints, or run with `--max_cases` (which skips the check). If you replace the npz files by different files with the same names, delete `case_qc.csv` so that the statistics are recomputed. |
| **Add a new metric** | Metrics come from `dvh_metrics()` in `visualization/rtog_vis_utils.py` (backend `numpy`, the default). Append a tuple `(structure, name, value, unit)` to the list it returns. STEP 7 builds the column names from these rows, so `test_metrics.csv`, `test_summary.csv`, mlflow and the table in the DVH figure pick the new metric up automatically. If you use `DVH_BACKEND = 'platipy'`, add the same row, in the same position, to `platipy_dvh_metrics()` in `visualization/vis_prediction.py`, because clinical and predicted rows are paired by position. Do not edit the files for the other students: copy the folder first. |
| **Change the best-checkpoint rule** | In CONFIG set `SELECT_BEST_BY = 'val_dose_score'` or `'val_dvh_score'` (default `'val_loss'`). The `<- best` flags, `best_model.pt`, the best epoch in `learning_curves.png` and the test (STEP 7, which loads that checkpoint) then follow that metric. mlflow logs the param `select_best_by`, and for the two scores also `best_val_dose_score` / `best_val_dvh_score`; `best_val_loss` stays the validation MAE at the chosen epoch. The DVH score is noisy (see Step G), so look at the curve before trusting its minimum. In the two 10-epoch runs the three choices give the same epoch (10 and 9), so nothing changes there; compare them on longer runs. |
| **Train on the DVH score?** | The scores are not part of the loss. `torch.quantile` and `.max()` can pass gradients (to the voxels that sit at the quantile or are the maximum), so a DVH-type loss is possible in principle, but `openkbp_scores` is decorated with `@torch.no_grad()` and works on detached tensors, so using it as a loss term gives no gradient. To try it, write a second function without `no_grad` (for example PTV D95 and organ Dmean/Dmax differences), add it to the loss in `run_epoch` for training only as in "Add a loss term", and weigh it against the MAE. This is an exercise and has not been tested. Expect a sparse, noisy signal (a few voxels per number). Note that `val_dvh_score` is then no longer an independent test of what you trained on. |
| **Change the 13 DVH numbers of the score** | Edit `openkbp_scores` in STEP 5 (PTV quantiles in `PTV_QUANTILES`, organ numbers in the loop). The training loop and STEP 7 both call it, so train, validation and test stay consistent. `compute_eval_metrics` in `MHRG_files/model.py` is not a drop-in replacement (it takes quantiles over the whole volume). |

---

## 5. Troubleshooting

| Problem | Cause and fix |
|---|---|
| `torch.cuda.OutOfMemoryError` | The GPU is shared. Check `nvidia-smi` and wait until you have about 12 GiB free; do not start two runs at once. Batch size is already 1. Close notebooks that still hold GPU memory (restart the kernel). (OOM itself was not provoked in testing; the memory figures come from the script's own measurement.) |
| `AssertionError: expected 163 cases after QC` | The dataset or `MIN_PTV_MEAN_GY` changed. Look at the QC table printed above the error and `results/rtog_imrt_teaching/case_qc.csv`, then update `EXPECTED_CASES_AFTER_QC` (about line 167) or use `--max_cases`. |
| `FileNotFoundError` / empty QC | `DATA_DIR` is wrong. It is an absolute path; set it to the folder with the `.npz` files. |
| Napari / OpenGL / Qt error, or a core dump, in Step B | Napari needs a display. Use `--skip_3d`. The training script never opens napari (figures are saved with the `Agg` backend). |
| `mlflow ui` crashes with `Router.__init__() got an unexpected keyword argument 'on_startup'` | Known: `mlflow ui` is broken in `AI_RTP` (version clash of starlette and fastapi). Do not try to fix `AI_RTP`. Use `python mlflow_tools.py ...`, or the separate virtual environment described in `MLFLOW_GUIDE.md`. Training and logging are not affected. |
| A run shows RUNNING forever in mlflow | The job was killed (`kill`, closed SSH session without `nohup`, crash), so mlflow never closed the run. `mlflow_tools.py list` shows it as `RUNNING?` after 30 minutes without new metrics. Check `pgrep -af test_notebook_rtog_imrt`, then close it as shown in `MLFLOW_GUIDE.md` (`set_terminated`). |
| `Run with UUID ... is already active` when you re-run STEP 6 in Spyder or Jupyter | An earlier attempt died between `mlflow.start_run` (STEP 6) and `mlflow.end_run()` (end of STEP 8). Run `mlflow.end_run()` once in the console and start again. |
| Figures have a strange aspect ratio, or the console says `default (no DICOM found)` | `load_case` estimates the voxel spacing from the DICOM. If the DICOM folder is not available, it falls back to 2.5 mm cubic voxels. The npz files carry no spacing. This only changes how the figures look. The DVH values and metrics are not affected. STEP 7 never reads DICOM. |
| Relative paths: `results/...log: No such file or directory`, or mlflow shows no runs | Run the commands from `KBP_Project`. The training script itself writes to `results/` and `mlruns/` next to the script, wherever you start it, but the shell redirect `> results/...log` and `mlflow ... file:mlruns` are relative to the current folder. In Jupyter, start the kernel in `KBP_Project`. Create `results/rtog_imrt_teaching/` before the first `nohup ... > results/rtog_imrt_teaching/<run>.log`. |
| The attention U-Net has a huge validation MAE in epoch 1 (about 100 Gy) | Seen in a tiny 1-epoch smoke run (102.9 Gy), not in the full run (epoch 1: 1.950 Gy). Normal with batch size 1 and BatchNorm; the statistics settle after a few dozen steps. |
| About 200 lines `initialize network with kaiming` in the log of the attention U-Net | Harmless. `Attn_UNet3D` (original model code) prints one line per initialised block when it is constructed, about 100 per construction (204 lines in `attn_unet3d_ep10.log`: the script builds the model twice). Filter them with `grep -v "initialize network"`. The U-Net prints two lines `initialize network with xavier`. |
| The DVH score jumps up in one epoch (attention U-Net, epoch 4) | Not a bug. The DVH score uses quantiles and maxima of few voxels (Dmax, D1), so a small shift in the PTV dose level or one hot voxel moves it a lot, while the MAE hardly changes. Look at the neighbouring epochs; use the loss to choose the checkpoint unless you have checked the curve. |
| `ModuleNotFoundError: No module named 'MHRG_files'` in your own snippet | Add `sys.path.insert(0, '/home/grp_airtp/AI_RTP/KBP_Project')` first (Step H). |
| Do not | import `MHRG_files.utils` (it imports napari and needs a display), edit `MHRG_files/model.py`, or install packages into `AI_RTP`. |

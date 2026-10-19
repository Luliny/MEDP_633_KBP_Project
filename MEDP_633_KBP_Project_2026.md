**MEDP 633 Knowledge-based Planning and Machine Learning Project**

**Option B: PyTorch platform**

**(Updated 10/19/2026)**

# 1. Introduction

In this project, we are going to train a deep learning neural network to predict 3D dose distributions and DVHs, taking input data of patient CT images and target and OAR contours. The network architecture we are using is U-net, which is a version of the Convolutional Neural Network commonly used for image segmentation. The data are lung radiation therapy plans of the RTOG 0617 trial. The DVHs are calculated from the predicted dose.

Some of the code for this project are adapted from MSK doseRTX project (<https://github.com/nadeemlab/DoseRTX>). You can find more physics background on this project in the publication: Jhanwar, G., Dahiya, N., Ghahremani, P., Zarepisheh, M., & Nadeem, S. (2022). Domain knowledge driven 3D dose prediction using moment-based loss function. _Physics in Medicine and Biology_, _67_(18). <https://doi.org/10.1088/1361-6560/ac8d45>.

The prediction accuracy is measured with two scores, the dose score and the DVH score, which follow the OpenKBP challenge.

# 2. Model Training Code Environment

Deep learning models are trained on Python PyTorch platform. The data are provided already preprocessed (npz files). Data visulization (`visualization/`) can run on a PC without GPU, in any Python IDE such as Spyder or Jupyter notebook. The model training code needs a computer with a GPU.

**Google Colab: These are the steps it would need to setup on Colab:

1. Choose a GPU runtime. The training needs about 11.3 GiB of GPU memory at batch size 1; a GPU with 16 GB may be enough.
2. Upload the code package and the npz data (3.6 GB) to your Google Drive or to the Colab session, and mount the Drive.
3. Install the packages: `pip install -r requirements_rtog_imrt.txt` (see below). Colab comes with its own Python and torch versions; the pinned versions were tested only with Python 3.13, so you may have to relax some pins .
4. Set `DATA_DIR` in the CONFIG block of `test_notebook_rtog_imrt.py` to the folder with the npz files.
5. Run the script from the `KBP_Project` folder (`!python -u test_notebook_rtog_imrt.py ...`), or run the script cell by cell. 

**Required Python modules.** These are the modules that the training script, `MHRG_files/model.py`, the `visualization/` folder and `mlflow_tools.py` import :

| Module | Version | Used for |
|---|---|---|
| torch | 2.9.0 (CUDA 12.8 build) | network, training |
| torchio, albumentations, lightning | 0.20.23, 2.0.8, 2.5.6 | imported by `MHRG_files/model.py` |
| mlflow | 3.3.2 | run tracking |
| numpy, pandas, scipy | 2.2.6, 2.3.3, 1.16.0 | data, tables |
| scikit-image | 0.25.2 | image functions of the visualisation code |
| matplotlib | 3.10.0 | all figures |
| SimpleITK, pydicom | 2.5.2, 2.4.4 | image conversion, voxel spacing from DICOM (visualisation) |
| platipy | 0.7.2 | DVH curves and CT/dose overlays |
| napari, PyQt5, vispy (optional) | 0.6.6, 5.15.10, 0.15.2 | interactive 3D figure only; needs a display |

**Configuration files** (in the root of the package, next to `test_notebook_rtog_imrt.py`):

- `requirements_rtog_imrt.txt` for pip, `environment_rtog_imrt.yml` for conda. 

```bash
# pip (a virtual environment is recommended)
python -m venv kbp_env && source kbp_env/bin/activate
pip install -r requirements_rtog_imrt.txt

# or conda
conda env create -f environment_rtog_imrt.yml
conda activate kbp_rtog_imrt
```

- The torch line installs the CUDA 12.8 build on Linux. To choose the CUDA build explicitly, see the comment in the file (`--index-url https://download.pytorch.org/whl/cu128`).

# 3. Description of the Data Set

The data set is shared with you on the VCU FileLocker. The link to download the dataset will be shared with you in a separate email. Copy it to your own folder and set `DATA_DIR` in `test_notebook_rtog_imrt.py` to that folder.

**Source.** The dataset contains the **179 anonymized IMRT lung plans** of the RTOG 0617 trial. The plans were extracted from the DICOM files, cropped around the lungs, heart and PTV, and resampled to 128 x 128 x 128 voxels. The files are in npz format, one file per plan, named `<patient>_<plan index>.npz` (about 21 MB each, 3.6 GB in total). 
**Quality control and split.** The training script checks every file first (STEP 1) and excludes cases by rule. Of the 179 files, **163 are kept** and 16 are excluded. The 163 cases are split **by patient** with a fixed random seed (42) into **115 training, 24 validation and 24 test cases** (70/15/15 %), see `split.csv` in each run folder. The same seed gives the same split in every run. The validation cases are used after every epoch to monitor training and to choose the best epoch; the test cases are used once, at the end.

**Contents of each npz file.** Each file contains a python dictionary of data arrays, with the keys:

CT=ct, DOSE=dose, OAR=oar, PTV=ptv, DOSE_SCALE=dose_scale, HIST_SIG=hist_sig, HIST_CLINICAL=hist_clinical, BINS=bins, LUNG_IPSI_SIDE=lung_ipsi_side

The ct is the patient CT image, dose is the 3D dose array, oar and ptv are the structure masks for the organs and the target. The axes are (X right to left, Y anterior to posterior, Z inferior to superior). Hist_sig and hist_clinical are the Dose Volume Histograms (DVH) calculated using the sigmoid function or the Heaviside function. The hist_clinical are the DVHs normally used in clinics. Note: the bins of `HIST_SIG` are 0 to 75 Gy (376 values), the `BINS` array belongs to `HIST_CLINICAL` (0 to 80 Gy); do not plot `HIST_SIG` against `BINS`. In both DVH arrays the columns are 0 cord, 1 esophagus, 2 heart, 3 lung_ipsi, 4 lung_cntr and 7 PTV; **columns 5 and 6 are unused** (they are always NaN). The training script does not use the DVH arrays: it computes DVH numbers from the dose directly.

We are going to use 5 organs in the models: 'cord': 1, 'esophagus': 2, 'heart': 3, 'lung_ipsi': 4, 'lung_cntr': 5. The lungs are labelled ipsilateral / contralateral to the tumour (not left / right).

**What the network sees (STEP 3 of the script).**

- Input: **7 channels**, each 128 x 128 x 128: the CT (clipped to -1000 to 1300 HU and scaled to 0-1), the 5 organ masks (cord, esophagus, heart, lung_ipsi, lung_cntr) and the PTV mask.
- Target: **the dose multiplied by `DOSE_SCALE`**, so that the mean PTV dose is 60 Gy in every case (the 5 organs plus the PTV define the structures for the scores). 

# 4. Description of the Code

The codes and setup file for the whole project is on github and shared with you at: <https://github.com/Luliny/AI_RTP>. (`git remote -v` in the working copy shows `origin https://github.com/Luliny/AI_RTP` for fetch and push.) The instructor will upload the cleaned code package there **[instructor: state the branch name once uploaded]**. You can create a local copy of the codes using the git command:

```
git clone https://github.com/Luliny/AI_RTP "$YOUR_PROJECT_ROOT"
```

The project folder is `KBP_Project`. All commands in the documents are run from this folder. The package is arranged like this:

```
KBP_Project/
|-- test_notebook_rtog_imrt.py     the training script (STEP 0 to STEP 8): the file you work with
|-- MHRG_files/model.py            network classes (UNet3D, Attn_UNet3D), optimiser helper; do not edit
|-- visualization/                 figures: rtog_vis_utils.py (load a case, DVH metrics), vis_prediction.py (predicted vs clinical
|                                  figures, learning curves), make_visualization_examples.py (look at the data), others
|-- mlflow_tools.py                terminal tool to list, watch and compare runs
|-- results/                       outputs: results/rtog_imrt_teaching/<run_name>/ (one folder per run), visualization_examples/
|-- mlruns/                        mlflow store of all runs
|-- RTOG_IMRT_WORKFLOW.md          student guide: run, outputs, evaluation scores, how to modify (read this first)
|-- MLFLOW_GUIDE.md                how to monitor and compare runs
|-- requirements_rtog_imrt.txt, environment_rtog_imrt.yml   install files (Section 2)
`-- preprocess/identify_imrt_plans.py, IMRT_identification_findings.md   how the IMRT plans were selected (instructor)
```

Folders such as `data`, `models`, `options`, `util` and `scripts`, and the files `MEDP633_Project1_AI_RTP.ipynb`, `train.py`, `models/doseprediction3d_model.py` and `models/networks3d.py`, belong to the earlier DoseRTX code and are **not used** by the workflow of this project.

**Where to look.** The purpose of the table is to guide you where to look when you work on the assignment (Section 7). The training script is written like a notebook, with one "STEP" banner per cell. Line numbers refer to the files of the package (`test_notebook_rtog_imrt.py`, abbreviated TNR; search for the names if the lines moved).

| # | What | File:line | Code snippet | Comment |
|---|---|---|---|---|
| 1a | Location of the data sets | `test_notebook_rtog_imrt.py:76` | `DATA_DIR = '/home/grp_airtp/AI_RTP/VCU_CLEAN/VCU_DATA/Processed_rtog_npz_IMRT'` | Folder with the npz files; change it to your own copy. |
| 1b | Split into train / val / test | `test_notebook_rtog_imrt.py:295` | `good['split'] = split_by_patient(good, SPLIT_FRACTIONS, SEED)` | By patient, seed 42, fractions `SPLIT_FRACTIONS` (line 93, 0.70 / 0.15 / 0.15) give 115 / 24 / 24; the lists are written to `<run>/split.csv`. |
| 2 | Load one case for training | `test_notebook_rtog_imrt.py:337` | `def __getitem__(self, index):` | `RTOGDoseDataset.__getitem__` (lines 337-350) loads one npz and builds the 7-channel input `x` and the target `y = DOSE * DOSE_SCALE`. |
| 3 | Data augmentation | `test_notebook_rtog_imrt.py:348` | `if self.training and self.flip_lr and torch.rand(1).item() < 0.5:` | Random left-right flip (X axis) of input and dose together, training cases only. Switch: `AUGMENT_FLIP_LR` (line 95). |
| 4a | Optimiser and its parameters | `test_notebook_rtog_imrt.py:560` | `optimizer = configure_optimizers(model.parameters(), weight_decay=0.01, learning_rate=learning_rate)` | AdamW (the helper is in `MHRG_files/model.py:58`), weight decay 0.01; the learning rate is `LR` (line 81, 5e-4). |
| 4b | Learning-rate schedule | `test_notebook_rtog_imrt.py:561` | `scheduler = StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.1)` | The learning rate is divided by 10 after half of the epochs. |
| 4c | Loss function | `test_notebook_rtog_imrt.py:460` | `criterion = nn.L1Loss()` | Mean absolute error (MAE) over all voxels, in Gy. It is applied in `run_epoch` (line 521, `loss = criterion(out, y)`). |
| 4d | Dose score and DVH score | `test_notebook_rtog_imrt.py:481` | `def openkbp_scores(pred, y, x):` | Both scores come from this one function (STEP 5); definition below. They are measured, not optimised. |
| 5 | Choose the type of U-Net | `test_notebook_rtog_imrt.py:79` | `MODEL_NAME = 'unet3d'` | `'unet3d'` or `'attn_unet3d'` (also `--model` on the command line); the models are built in `get_model` (line 388, `def get_model(name, in_ch=7):`). |
| 6a | Number of filters (down- and up-sampling) | `test_notebook_rtog_imrt.py:84` | `UNET_ENC_CHS = (32, 64, 128, 256, 512)` | Filters of the 5 encoder levels; the decoder mirrors it. Passed to `UNet3D` in `get_model` (line 398). |
| 6b | Where the network is defined | `MHRG_files/model.py:307` | `def __init__(self, in_ch=7, enc_chs=(32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), out_ch=1):` | `UNet3D` (line 306) with `Encoder` (line 246), `Decoder` (line 277), `Block` (line 211) and `Block_dropout` (line 227). The attention U-Net has fixed widths: `MHRG_files/model.py:652`. |
| 7a | Plot the learning curves (call) | `test_notebook_rtog_imrt.py:607` | `curves_png = vp.plot_training_curves(history, os.path.join(OUT_DIR, 'learning_curves.png'), best_epoch=best_epoch,` | Called at the end of STEP 6, after training. |
| 7b | Plot the learning curves (definition) | `visualization/vis_prediction.py:315` | `def plot_training_curves(history, out_png, best_epoch=None, select_by='val_loss'):` | Three panels: loss, dose score, DVH score; train and validation; best epoch marked. |
| 8a | Visualisation of predictions (STEP 8) | `test_notebook_rtog_imrt.py:695` | `for case_id in test_ids[:N_VIS_CASES]:` | STEP 8 (line 687) draws figures for the first `N_VIS_CASES` test cases (line 94). |
| 8b | DVH comparison figure | `visualization/vis_prediction.py:148` | `def plot_dvh_comparison(case, pred_dose, out_png, dpi=150, metrics_backend='numpy'):` | Clinical (solid) vs predicted (dashed) DVH and a metric table. |
| 8c | Dose map figure | `visualization/vis_prediction.py:214` | `def plot_dose_comparison(case, pred_dose, out_png, diff_range=10, dpi=150, dose_min_frac=0.1):` | Clinical, predicted and difference in three views. |
| 8d | CT overlay figures | `visualization/vis_prediction.py:276` | `def plot_overlay_pair(case, pred_dose, out_dir, dpi=150):` | CT + dose + contours, clinical and predicted. |
| 9a | Number of epochs | `test_notebook_rtog_imrt.py:80` | `EPOCHS = 100` | Also `--epochs N` on the command line. |
| 9b | The epoch loop | `test_notebook_rtog_imrt.py:567` | `for epoch in range(1, epochs + 1):` | Inside `train_model` (line 550). |
| 10 | Example outputs | `results/rtog_imrt_teaching/unet3d_ep10/` | `learning_curves.png`, `figures/0617-410508_0_dvh_comparison.png` | Real output of a 10-epoch run, see the list below. |

**Item 4: how the loss and the scores are calculated.**

- **Loss (what is optimised).** The MAE between predicted and clinical dose over all 128 x 128 x 128 voxels, in Gy. It is easy to read, and it is not dominated by a few high-dose voxels (as the MSE would be). Most voxels are air or low dose, so this MAE looks small even if the target is badly predicted: predicting zero dose everywhere gives 5.24 Gy on the 24 test cases. Always look at the two scores below, not only at the loss.
- **Dose score (measured, not optimised).** The mean of |predicted - clinical dose| over the voxels **inside the body**. The body mask is the CT above -500 HU (`BODY_HU`). Zero dose everywhere gives 8.14 Gy.
- **DVH score (measured, not optimised).** For each case, the mean of |predicted - clinical| over **13 DVH numbers**, in Gy; then the mean over the cases. Zero dose everywhere gives 40.67 Gy. Each number is computed from the voxels of one structure only:
  - PTV: **D1, D95, D99**. D95 is the dose that 95% of the PTV volume receives at least (the 5th percentile of the PTV voxel doses, "is the target covered?"), D99 the dose that 99% of the volume receives at least (1st percentile), D1 the dose of the hottest 1% (99th percentile, the hot spot).
  - Each of the 5 organs (cord, esophagus, heart, lung_ipsi, lung_cntr): **Dmean** (mean dose) and **Dmax** (highest voxel dose). 5 x 2 = 10 numbers. In total 3 + 10 = 13.
- Both scores are computed every epoch on the training and the validation cases, and once at the end on the test cases, by the same function (`openkbp_scores`), so the curves and the final numbers are comparable. A perfect prediction gives 0 for both. A score is lower = better.
- **Deviations from OpenKBP.** (1) A **body mask** (CT > -500 HU) replaces the "possible dose mask" of OpenKBP, which we do not have. (2) **Dmax instead of D0.1cc** for the organs: OpenKBP uses the dose to the hottest 0.1 cc, which needs the voxel size, and our npz files carry none; Dmax is the plain maximum voxel dose, so it is more sensitive to single voxels and the DVH score is noisier. (3) Our own structure list for lung plans (OpenKBP is head-and-neck). (4) **Normalised units** (PTV mean = 60 Gy), so the numbers cannot be compared with published OpenKBP values.
- In addition, after training, the test step (STEP 7) writes the differences (predicted minus clinical) of clinical DVH metrics, for example PTV D95, lung V20 and heart mean dose, to `test_metrics.csv` and `test_summary.csv`. The full definitions are in "Evaluation scores" of `RTOG_IMRT_WORKFLOW.md` (Step G).

**Item 10: where the example outputs are.** All outputs of one run are in `results/rtog_imrt_teaching/<run_name>/`. The package contains two 10-epoch demonstration runs, `unet3d_ep10` and `attn_unet3d_ep10` (they are not converged; use them to see what the outputs look like):

| File | Content |
|---|---|
| `results/rtog_imrt_teaching/unet3d_ep10/learning_curves.png` | Learning curves: loss, dose score, DVH score vs epoch, train and validation, best epoch marked |
| `results/rtog_imrt_teaching/unet3d_ep10/history.csv` | The numbers behind the curves, one row per epoch |
| `results/rtog_imrt_teaching/unet3d_ep10/figures/0617-410508_0_dvh_comparison.png` | DVH comparison, clinical vs predicted (also `0617-415627_0_...`, `0617-429991_0_...`) |
| `results/rtog_imrt_teaching/unet3d_ep10/figures/0617-410508_0_dose_comparison.png` | Dose maps: clinical, predicted, difference |
| `results/rtog_imrt_teaching/unet3d_ep10/figures/0617-410508_0_overlay_clinical.png`, `..._overlay_predicted.png` | CT overlay with clinical and predicted dose |
| `results/rtog_imrt_teaching/unet3d_ep10/test_metrics.csv`, `test_summary.csv` | Test scores and DVH metric differences, per case and summary |
| `results/rtog_imrt_teaching/unet3d_ep10/best_model.pt` | Trained weights (Section 5) |
| `results/rtog_imrt_teaching/compare_runs.png` | Curves of the two demo runs together (made by `mlflow_tools.py compare`) |

The same files exist in `results/rtog_imrt_teaching/attn_unet3d_ep10/`. For orientation only (10 epochs, one seed, test set of 24 cases): the U-Net reaches a test dose score of 2.75 and a DVH score of 8.21; the attention U-Net 2.02 and 2.80.

**Monitoring with mlflow.** Every run is logged automatically to `mlruns/` (experiment `RTOG_IMRT_teaching`): the settings of the run (model, epochs, learning rate, filters, dropout, ...), the loss and both scores for train and validation after every epoch, the test scores, and the output files. You can watch a run, and compare runs, from the terminal: `python mlflow_tools.py list`, `watch` and `compare` (the web interface `mlflow ui` does not work in the `AI_RTP` environment). See `MLFLOW_GUIDE.md`.

**Student documents.** `RTOG_IMRT_WORKFLOW.md` explains how to run the workflow step by step, what every output file contains, how the scores are defined, how to reuse a trained model, and how to modify the code (exercises: change the model, the filters, the dropout, the loss, the augmentation). `MLFLOW_GUIDE.md` explains the monitoring.

# 5. Pretrained model

The pretrained model of the earlier version of this project (doseRTX, MSK data), `checkpoints/MAE_Moment_loss/saved_latest_net_G.pth`, is still in the package folder, but it **cannot be used with this code**: it was saved from a network with BatchNorm layers (its file contains `BN.*` weights), while the current U-Net uses InstanceNorm, and it was trained on the MSK data. Loading it into the current network fails.

Instead, the package provides trained checkpoints of this workflow:

- `results/rtog_imrt_teaching/unet3d_ep10/best_model.pt` (U-Net) and `results/rtog_imrt_teaching/attn_unet3d_ep10/best_model.pt` (attention U-Net), each about 100 MB. They are **10-epoch demonstration runs**, not converged.
- **[Placeholder: the instructor may provide a 100-epoch reference model. Location and link to be added.]**

To use a checkpoint, see the snippet in `RTOG_IMRT_WORKFLOW.md`, Step H ("reuse a trained model"): it rebuilds the network, loads `best_model.pt` and predicts one case. If you changed `UNET_ENC_CHS`, use the same widths there.

# 6. Work to do

Our work to do are:

1. Get access to a computer with a GPU: an account on the course server (or, as an option, Google Colab, Section 2). **[instructor: add how to get the server account and access]**
2. Install the coding environment (on the server: activate `AI_RTP`; elsewhere: the install files of Section 2) and clone the project code (Section 4).
3. Get the data set (Section 3) and set `DATA_DIR`.
4. Run the quick test to check that everything works (about 1 minute; it makes a run called `quick_test`, which you can delete afterwards):

   ```
   python -u test_notebook_rtog_imrt.py --epochs 1 --max_cases 12 --run_name quick_test
   ```

5. Run a trained checkpoint on a test case with the snippet of Step H of `RTOG_IMRT_WORKFLOW.md` and look at the example figures (Section 5 and Section 4, item 10).
6. Train the initial network with the default settings (about 1 hour for 100 epochs for the U-Net), and monitor it with mlflow (`python mlflow_tools.py watch`).
7. Then tune: change the settings (Section 7, item 3), one change at a time, and compare the runs.

# 7. Project Assignment

You are going to submit a project report with 4 sections in the format of a paper on Medical Physics: Introduction, Methods and materials, Results and Discussion. This project will account for 30 points in your final grades. **Due by [date to be set by instructor]**.

1. **Train the initial network.** Your methods section should include the following specifics about the network you have trained: a diagram of the architecture like the example shown in Figure 4 on which the dimensions for each feature map, the number of the filters (features) for each layer are annotated; the type of the network; the type of the loss function and optimizer; the type of data augmentation transformations. **14 points**.

2. **Obtain results from the initial network.** You should include the following results: plots of training loss as a function of epochs; and plots of the dose and DVH scores from the training and validation vs. epochs; when do you stop training (at which epoch); the final training and validation dose and DVH scores. **10 points**.

3. **Model refinement and hyperparameter tuning.** You should tune the network architecture and hyperparameters to optimize the prediction accuracy of your network in terms of the final validation dose and DVH scores (smaller is better). At least one of the following hyperparameters should be fine-tuned: number of filters, dropout rate, learning rate, type of data augmentation, or adjust network structure, etc. Present the results from your own network similarly as in item 2 above. **6 points**.

Figure 4. An example network architecture diagram **[instructor: insert the image here]**

## Hints from the code

**Item 1: where to read each specific.**

| Specific | Where |
|---|---|
| Type of the network | `MODEL_NAME` (line 79) and `get_model` (line 388) of `test_notebook_rtog_imrt.py`; classes in `MHRG_files/model.py` (`UNet3D` line 306; `Attn_UNet3D` line 640). The default is `unet3d`: a 3D U-Net with 5 levels, 3x3x3 convolutions, ReLU, InstanceNorm, max-pooling for down-sampling, trilinear up-sampling followed by a convolution, skip connections (concatenation), a final 1x1x1 convolution and a ReLU (the dose cannot be negative). Output: 25.89 M parameters. |
| Loss function | `criterion = nn.L1Loss()` (line 460): MAE. |
| Optimizer | `configure_optimizers(...)` (line 560): AdamW, weight decay 0.01, learning rate `LR` = 5e-4, divided by 10 after half of the epochs (line 561). Batch size 1. |
| Data augmentation | line 348: random left-right flip, probability 0.5, training cases only. |
| Numbers for the Methods section | The check line printed at the start of a run (`unet3d: output ..., 25.89 M parameters, ...`), the split line (`train 115 / val 24 / test 24`), and the settings of each run stored in mlflow (model, epochs, learning rate, filters, dropout; `python mlflow_tools.py list`). |

**Item 1: feature-map sizes for the diagram** (default `unet3d`, batch size 1; the volume is 128 x 128 x 128; each "block" is Conv3x3x3 - ReLU - InstanceNorm - Conv3x3x3 - ReLU - InstanceNorm):

| Level | Encoder (after the block) | Decoder (after the block) |
|---|---|---|
| Input | 7 x 128³ | |
| 1 | 32 x 128³ | up-conv 64 -> 32, concatenate the skip (32): 64 -> block -> **32 x 128³** |
| 2 | 64 x 64³ (after max-pool 2) | up-conv 128 -> 64, concatenate (64): 128 -> block -> **64 x 64³** |
| 3 | 128 x 32³ | up-conv 256 -> 128, concatenate (128): 256 -> block -> **128 x 32³** |
| 4 | 256 x 16³ | up-conv 512 -> 256, concatenate (256): 512 -> block -> **256 x 16³** |
| 5 (bottom) | 512 x 8³ (this block contains the dropout layer, p = 0.4) | |
| Output | | 1x1x1 convolution, 32 -> 1, then ReLU: 1 x 128³ |

Each up-conv doubles the size of the map (8³ -> 16³ -> 32³ -> 64³ -> 128³) and then applies a 3x3x3 convolution. If you change `UNET_ENC_CHS`, replace the channel numbers (the decoder mirrors the encoder). The attention U-Net has the same widths but adds attention gates on the skip connections, and deep supervision (see `MHRG_files/model.py` line 640).

**Item 2: curves, scores and when to stop.**

- After a run, `results/rtog_imrt_teaching/<run>/history.csv` has one row per epoch with the columns `epoch, train_mae, val_mae, train_dose_score, val_dose_score, train_dvh_score, val_dvh_score, lr, seconds`. Plot from it, or use `learning_curves.png` (loss, dose score, DVH score; train and validation). `train_mae` / `val_mae` are the loss (called `train_loss` / `val_loss` in mlflow). The same six series are logged to mlflow every epoch (`python mlflow_tools.py compare` makes a table and a figure with the curves of several runs).
- The train numbers are averaged over the epoch with flipped cases and dropout on, so the train curve is not directly comparable with the validation curve; validation below training is normal here.
- **When to stop.** The script keeps the checkpoint of the epoch with the lowest value of the validation metric chosen by `SELECT_BEST_BY` (line 100; default `'val_loss'`, alternatives `'val_dose_score'` and `'val_dvh_score'`). That epoch is marked `<- best` in the console, written as `best_epoch` in mlflow, and drawn as a dotted line in `learning_curves.png`; the test step uses that checkpoint. "Stop at which epoch" is this best epoch, if you give a reason (for example the validation curve stops falling, or starts to rise while the training curve keeps falling). A curve that is still falling at the last epoch means that the model needs more epochs. The DVH score is noisier than the loss, so look at the curve before trusting its minimum.
- The final scores to report are the values of `history.csv` at the best epoch (train and validation dose and DVH scores). The test scores in `test_summary.csv` are for the held-out test cases (use them only once, for the final model).
- All numbers are in normalised Gy (PTV mean = 60 Gy); lower is better.

**Item 3: settings you can change** (CONFIG block of `test_notebook_rtog_imrt.py`, lines 76-108; other settings are possible by editing the code). Give every run its own name with `--run_name`, change one thing at a time, and compare the runs with `python mlflow_tools.py compare`.

| Setting | Line | What it does |
|---|---|---|
| `UNET_ENC_CHS` | 84 | Number of filters per level, for example `(16, 32, 64, 128, 256)` (6.5 M parameters, 3.4 GiB) or `(32, 64, 128, 256, 256)` (17.0 M, 4.7 GiB). `unet3d` only. |
| `DROPOUT_P` | 88 | Dropout probability (default 0.4; 0.0 = off). `unet3d` only; the network has one dropout layer, in the deepest encoder block. Also `--dropout`. |
| `LR` | 81 | Learning rate (default 5e-4). Also `--lr`. |
| `EPOCHS` | 80 | Number of epochs. Also `--epochs`. |
| `AUGMENT_FLIP_LR` | 95 | Left-right flip on / off. To add another augmentation, edit `RTOGDoseDataset.__getitem__` (line 337); apply the same transform to input and target. |
| `MODEL_NAME` | 79 | `'unet3d'` or `'attn_unet3d'` (the attention U-Net has fixed widths and no dropout). Also `--model`. |
| `SELECT_BEST_BY` | 100 | Which validation metric chooses the best epoch. |
| Add a loss term | 537-538 | For example an extra MAE inside the PTV. A tested example is in the exercises of `RTOG_IMRT_WORKFLOW.md` (section 4). |

Tuning hints. The result of one run changes a little from run to run (GPU computation is not exactly reproducible), so do not over-interpret small differences between two runs. Compare runs by the **validation** dose and DVH scores, as the assignment says. Use the same split (it is the same for all runs by default).

**Using the shared GPU fairly.** The GPU of the course server is shared by all students, and one run needs about 11 GiB of the 24 GB (less with fewer filters).

- Run **one training at a time**; do not start two runs in parallel.
- Before you start, check `nvidia-smi` and wait until about 12 GiB are free. If you get `torch.cuda.OutOfMemoryError`, wait; do not retry in a loop.
- For long runs use `nohup` with unbuffered output, so that the run survives a closed SSH session and the log fills while it runs:

  ```
  mkdir -p results/rtog_imrt_teaching
  nohup python -u test_notebook_rtog_imrt.py --run_name my_run > results/rtog_imrt_teaching/my_run.log 2>&1 &
  ```

- Close notebooks that still hold GPU memory when you are finished (restart the kernel).
- Test your changes with the quick test (`--epochs 1 --max_cases 12`) before you start a long run.

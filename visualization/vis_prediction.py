"""Figures and DVH helpers for comparing a PREDICTED dose with the CLINICAL dose of an RTOG 0617 npz case.

Used by KBP_Project/test_notebook_rtog_imrt.py (teaching training script), but usable on its own.

    plot_dvh_comparison(case, pred_dose, out_png)   clinical (solid) vs predicted (dashed) DVHs + metric table
    plot_dose_comparison(case, pred_dose, out_png)  axial/coronal/sagittal: clinical | predicted | predicted - clinical
    plot_training_curves(history, out_png, ...)     3 panels per epoch: loss | dose score | DVH score (train / val)
    plot_learning_curves(train, val, out_png)       train / validation MAE per epoch (single panel, older figure;
                                                    kept for backward compatibility, the training script no longer uses it)
    plot_overlay_pair(case, pred_dose, out_dir)     the two platipy CT + dose + contour overlays on ONE colour scale

DVH helpers (all fast, nothing here is meant for the training loop):
    compute_dvh(dose, mask, bins)           numpy cumulative DVH (volume fraction, "dose >= bin")
    platipy_dvh(case, dose)                 DVH of every structure with platipy.imaging.dose.dvh
    platipy_dvh_metrics(case, dose)         RTOG 0617 metrics (same rows as rtog_vis_utils.dvh_metrics) via platipy
    case_metrics(case, dose, backend)       rows of (structure, metric, value, unit); backend 'numpy' (default,
                                            fastest) or 'platipy'

IMPORTANT: `case` must be loaded with load_case(id, normalize=True) so that case.dose is in the SAME units as the
network prediction (clinical dose * DOSE_SCALE, i.e. PTV mean = 60 Gy). pred_dose is a (128, 128, 128) array
in the npz axis order (X, Y, Z); no transposition is needed.

Example (fake prediction = blurred clinical dose):
    conda run -n AI_RTP python visualization/vis_prediction.py 0617-634370_0 --out_dir /tmp/demo
"""
import argparse
import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtog_vis_utils import CT_WINDOW, STRUCTURES, dvh_metrics, load_case, to_sitk

INK, INK_2, GRID_INK = '#0b0b0b', '#52514e', '#e4e3df'
PLATIPY_BIN_WIDTH = 0.05      # Gy; with 0.05 Gy bins platipy matches HIST_CLINICAL to ~0.2 % volume
_DOSE_CMAP = 'inferno_dosewash'   # registered by vis_platipy_overlay (inferno without its black end)


# --------------------------------------------------------------------------------------------------------------
# DVH computation
# --------------------------------------------------------------------------------------------------------------
def compute_dvh(dose, mask, bins):
    """Cumulative DVH as volume FRACTION (0-1): fraction of the structure voxels with dose >= each bin.

    Same definition as HIST_CLINICAL (Heaviside). Sorting the structure voxels once and using searchsorted is
    ~1000x faster than looping over the bins on the whole volume.
    """
    vals = np.sort(dose[mask > 0].ravel())
    if vals.size == 0:
        return np.full(len(bins), np.nan)
    return 1.0 - np.searchsorted(vals, bins, side='left') / vals.size


def _platipy_labels(case, dose):
    """sitk dose image + {name: binary sitk label} for every structure (+ the union of both lungs)."""
    dose_img = to_sitk(np.asarray(dose, dtype=np.float32), case.spacing)
    masks = case.masks()
    labels = {k: to_sitk(m.astype(np.uint8), None, like=dose_img) for k, m in masks.items()}
    if 'lung_ipsi' in masks and 'lung_cntr' in masks:
        labels['lungs'] = to_sitk((masks['lung_ipsi'] | masks['lung_cntr']).astype(np.uint8), None, like=dose_img)
    return dose_img, labels


def platipy_dvh(case, dose, bin_width=PLATIPY_BIN_WIDTH, max_dose=None):
    """platipy.imaging.dose.dvh.calculate_dvh_for_labels for all structures of `case` (+ 'lungs').

    Returns the platipy DataFrame: columns label, cc (volume in cm^3 on the estimated spacing), mean (Gy), then
    one float column per dose bin CENTRE holding the volume fraction with dose >= bin lower edge.
    """
    from platipy.imaging.dose.dvh import calculate_dvh_for_labels
    dose_img, labels = _platipy_labels(case, dose)
    if max_dose is None:
        max_dose = float(np.max(dose)) + 1.0
    return calculate_dvh_for_labels(dose_img, labels, bin_width=bin_width, max_dose=max_dose)


def platipy_curves(dvh, bin_width=PLATIPY_BIN_WIDTH):
    """(dose_edges, {structure: volume fraction}) from a platipy DVH frame.

    platipy labels each bin by its centre but counts voxels with dose >= the bin's lower edge, so the x axis is
    shifted by half a bin to give the usual 'V(dose >= x)' curve.
    """
    cols = [c for c in dvh.columns if isinstance(c, float)]
    x = np.array(cols) - bin_width / 2.0
    return x, {row['label']: np.asarray(row[cols], dtype=float) for _, row in dvh.iterrows()}


def platipy_dvh_metrics(case, dose=None, dvh=None, bin_width=PLATIPY_BIN_WIDTH):
    """Same rows as rtog_vis_utils.dvh_metrics, but D95/D2/V20/V5/V30 read from a platipy DVH.

    Mean doses come from platipy's 'mean' column, volumes (V_x, platipy returns cc) are converted to %.
    """
    from platipy.imaging.dose.dvh import calculate_d_x, calculate_v_x
    dose = case.dose if dose is None else dose
    if dvh is None:
        dvh = platipy_dvh(case, dose, bin_width)
    present = set(dvh.label)
    mean = dict(zip(dvh.label, dvh['mean']))
    cc = dict(zip(dvh.label, dvh['cc']))
    rows = []
    if 'ptv' in present:
        d = calculate_d_x(dvh, [95, 2], label='ptv').iloc[0]
        rows += [('PTV', 'Mean', mean['ptv'], 'Gy'), ('PTV', 'D95%', d['D95'], 'Gy'), ('PTV', 'D2%', d['D2'], 'Gy')]
    if 'lungs' in present:
        v = calculate_v_x(dvh, [20, 5], label='lungs').iloc[0]
        rows += [('Lungs', 'Mean', mean['lungs'], 'Gy'), ('Lungs', 'V20', 100 * v['V20'] / cc['lungs'], '%'),
                 ('Lungs', 'V5', 100 * v['V5'] / cc['lungs'], '%')]
    if 'heart' in present:
        v = calculate_v_x(dvh, [30, 5], label='heart').iloc[0]
        rows += [('Heart', 'Mean', mean['heart'], 'Gy'), ('Heart', 'V30', 100 * v['V30'] / cc['heart'], '%'),
                 ('Heart', 'V5', 100 * v['V5'] / cc['heart'], '%')]
    if 'esophagus' in present:
        rows += [('Esophagus', 'Mean', mean['esophagus'], 'Gy')]
    if 'cord' in present:
        m = case.masks()['cord']
        rows += [('Spinal cord', 'Max', float(np.max(np.asarray(dose)[m])), 'Gy')]   # platipy has no D_max
    return rows


def case_metrics(case, dose=None, backend='numpy'):
    """RTOG 0617 metric rows (structure, metric, value, unit) of `dose` (default: case.dose) on case's contours.

    backend 'numpy'   -> rtog_vis_utils.dvh_metrics (exact, ~6 ms per case, the default)
            'platipy' -> platipy DVH (~50 ms per case, binned at PLATIPY_BIN_WIDTH)
    """
    if backend == 'platipy':
        return platipy_dvh_metrics(case, dose)
    return dvh_metrics(case if dose is None else replace(case, dose=np.asarray(dose, dtype=np.float32)))


# --------------------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------------------
def _style_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)


def plot_dvh_comparison(case, pred_dose, out_png, dpi=150, metrics_backend='numpy'):
    """Left: clinical (solid, HIST_CLINICAL) vs predicted (dashed, platipy DVH of pred_dose). Right: metric table."""
    pred_dose = np.asarray(pred_dose, dtype=np.float32)
    bins = case.dvh_bins()
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={'width_ratios': [1.9, 1.25]})

    dvh_pred = platipy_dvh(case, pred_dose, max_dose=max(float(pred_dose.max()), float(case.dose.max())) + 1.0)
    x_pred, curves = platipy_curves(dvh_pred)
    for key, label, col, color, _ in STRUCTURES:
        ref = case.hist_clinical[:, col]
        if not np.isfinite(ref).all() or ref.max() <= 1e-6:        # structure not present (volume 0 -> NaN)
            continue
        ax.plot(bins, 100 * ref, color=color, lw=1.8, label=case.display_name(key))
        if key in curves:
            ax.plot(x_pred, 100 * curves[key], color=color, lw=1.6, ls='--')
    ax.add_line(Line2D([], [], color=INK_2, lw=1.8, label='solid: clinical'))
    ax.add_line(Line2D([], [], color=INK_2, lw=1.6, ls='--', label='dashed: predicted'))
    ax.set_xlim(0, max(85.0, float(bins.max())) if case.normalized else 85.0)
    ax.set_ylim(0, 102)
    ax.set_xlabel('Dose (Gy)', color=INK)
    ax.set_ylabel('Volume (%)', color=INK)
    ax.grid(True, color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.set_title('{}: cumulative DVH, clinical vs predicted\n(dose x DOSE_SCALE {:.3f}, PTV mean 60 Gy)'.format(
        case.case_id, case.dose_scale), fontsize=11, color=INK, loc='left')

    clin = case_metrics(case, None, metrics_backend)
    pred = case_metrics(case, pred_dose, metrics_backend)
    rows = []
    for (s, m, vc, u), (_, _, vp, _) in zip(clin, pred):
        rows.append([s, m, '{:.1f}'.format(vc), '{:.1f}'.format(vp), '{:+.1f} {}'.format(vp - vc, u)])
    ax_t.axis('off')
    tab = ax_t.table(cellText=rows, colLabels=['Structure', 'Metric', 'Clinical', 'Predicted', 'Diff'],
                     loc='upper center', cellLoc='left', colLoc='left', colWidths=[0.28, 0.16, 0.18, 0.2, 0.2])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID_INK)
        cell.get_text().set_color(INK if r else INK_2)
        if r == 0:
            cell.get_text().set_weight('bold')
    ax_t.set_title('Dose metrics (Gy, % volume; Diff = predicted - clinical)', fontsize=10, color=INK, loc='left')

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def _views(case, com):
    """Three slice extractors through `com` = (x, y, z) voxel index. Each returns a 2D array with
    origin='lower' display orientation matching platipy: axial anterior up, coronal/sagittal superior up,
    patient left on the screen right (axial/coronal), anterior on the screen left (sagittal).
    Returns [(name, fn(arr) -> 2D, aspect, (col_label, row_label))]."""
    x, y, z = com
    sx, sy, sz = case.spacing
    return [
        ('Axial', lambda a: a[:, :, z].T[::-1], sy / sx),        # rows: y (anterior at top), cols: x
        ('Coronal', lambda a: a[:, y, :].T, sz / sx),            # rows: z (superior at top), cols: x
        ('Sagittal', lambda a: a[x, :, :].T, sz / sy),           # rows: z (superior at top), cols: y (anterior left)
    ]


def plot_dose_comparison(case, pred_dose, out_png, diff_range=10, dpi=150, dose_min_frac=0.1):
    """Rows axial / coronal / sagittal through the PTV centre of mass; columns clinical | predicted | predicted-clinical.

    Clinical and predicted dose share one colour scale; the difference uses a symmetric +-diff_range Gy RdBu_r scale.
    """
    import vis_platipy_overlay  # noqa: F401  (registers the 'inferno_dosewash' colormap)
    from matplotlib.colors import Normalize
    pred_dose = np.asarray(pred_dose, dtype=np.float32)
    ref = 60.0 if case.normalized else float(case.dose[case.ptv > 0].mean())
    vmax = float(max(case.dose.max(), pred_dose.max()))
    vmin = dose_min_frac * ref
    diff = pred_dose - case.dose
    shown = np.maximum(case.dose, pred_dose) >= vmin          # only show the difference where there is dose

    ptv = case.ptv > 0
    com = tuple(int(round(float(c.mean()))) for c in np.nonzero(ptv)) if ptv.any() else (64, 64, 64)
    views = _views(case, com)
    masks = case.masks()
    colors = {s[0]: s[4] for s in STRUCTURES}
    dose_cmap = plt.get_cmap(_DOSE_CMAP)
    ct_kw = dict(cmap='gray', vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], origin='lower')

    h = [case.ct.shape[2] * case.spacing[2] if v[0] != 'Axial' else case.ct.shape[1] * case.spacing[1] for v in views]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13.5), gridspec_kw={'height_ratios': h}, layout='constrained')
    titles = ['Clinical dose (Gy)', 'Predicted dose (Gy)', 'Predicted - clinical (Gy)']
    ims = [None] * 3
    for r, (name, fn, aspect) in enumerate(views):
        ct2d = fn(case.ct)
        for c, vol in enumerate([case.dose, pred_dose, diff]):
            ax = axes[r, c]
            ax.imshow(ct2d, aspect=aspect, **ct_kw)
            v2d = fn(vol)
            if c < 2:
                ims[c] = ax.imshow(np.ma.masked_less(v2d, vmin), aspect=aspect, origin='lower', cmap=dose_cmap,
                                   vmin=vmin, vmax=vmax, alpha=0.55)
            else:
                ims[c] = ax.imshow(np.ma.masked_where(~fn(shown), v2d), aspect=aspect, origin='lower', cmap='RdBu_r',
                                   norm=Normalize(-diff_range, diff_range), alpha=0.8)
            for key, m in masks.items():
                m2d = fn(m.astype(np.uint8))
                if m2d.any():
                    ax.contour(m2d, levels=[0.5], colors=[colors[key]], linewidths=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[c], fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(name, fontsize=11, color=INK)
    for c in range(3):
        cb = fig.colorbar(ims[c], ax=axes[:, c], shrink=0.5, pad=0.02, location='bottom', aspect=30)
        cb.set_label('Gy', color=INK_2)
    handles = [Line2D([], [], color=colors[k], lw=1.5, label=case.display_name(k)) for k in masks]
    fig.legend(handles=handles, loc='upper right', ncol=3, frameon=False, fontsize=9)
    fig.suptitle('{}: clinical vs predicted dose\nslices through the PTV centre; shared dose scale up to {:.0f} Gy, '
                 'difference +-{:g} Gy'.format(case.case_id, vmax, diff_range), fontsize=11, x=0.01, ha='left')
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def plot_overlay_pair(case, pred_dose, out_dir, dpi=150):
    """Two platipy overlays (clinical, predicted) with the SAME dose colour scale. Returns (png_clin, png_pred)."""
    from vis_platipy_overlay import plot_overlay
    pred_dose = np.asarray(pred_dose, dtype=np.float32)
    dmax = float(max(case.dose.max(), pred_dose.max()))
    a = plot_overlay(case, os.path.join(out_dir, '{}_overlay_clinical.png'.format(case.case_id)), dpi=dpi, dose_max=dmax)
    b = plot_overlay(replace(case, dose=pred_dose),
                     os.path.join(out_dir, '{}_overlay_predicted.png'.format(case.case_id)), dpi=dpi, dose_max=dmax)
    return a, b


def plot_learning_curves(train_loss, val_loss, out_png, best_epoch=None, ylabel='MAE (Gy, PTV mean = 60 Gy)'):
    """Train and validation MAE vs epoch (epochs counted from 1); best_epoch is 1-based and marked."""
    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(epochs, train_loss, color='#2a78d6', lw=1.8, label='train')
    ax.plot(epochs, val_loss, color='#e34948', lw=1.8, label='validation')
    if best_epoch is not None:
        ax.axvline(best_epoch, color=INK_2, lw=1, ls=':')
        ax.plot([best_epoch], [val_loss[best_epoch - 1]], 'o', color='#e34948', ms=7, mec=INK)
        ax.annotate('best val {:.3f} (epoch {})'.format(val_loss[best_epoch - 1], best_epoch),
                    (best_epoch, val_loss[best_epoch - 1]), textcoords='offset points', xytext=(8, 12),
                    fontsize=9, color=INK)
    ax.set_xlabel('Epoch', color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    _style_axes(ax)
    ax.legend(frameon=False)
    ax.set_title('Learning curves', loc='left', fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


TRAIN_COLOR, VAL_COLOR = '#2a78d6', '#e34948'


def plot_training_curves(history, out_png, best_epoch=None, select_by='val_loss'):
    """Three panels side by side, train and validation per epoch (all in Gy, lower is better):
        (1) Loss        MAE over all voxels (what is optimised)
        (2) Dose score  MAE inside the body (OpenKBP dose score)
        (3) DVH score   OpenKBP-style DVH score (mean |pred - clinical| over 13 DVH metrics)
    `history` is a dict or DataFrame with the columns epoch, train_mae, val_mae, train_dose_score, val_dose_score,
    train_dvh_score, val_dvh_score (the history of the training script).  best_epoch (1-based) is marked on every
    panel; select_by names the validation metric that chose it and appears in the title.
    """
    h = {k: np.asarray(history[k], dtype=float) for k in ('train_mae', 'val_mae', 'train_dose_score', 'val_dose_score',
                                                         'train_dvh_score', 'val_dvh_score')}
    epochs = np.asarray(history['epoch']) if 'epoch' in history else np.arange(1, len(h['val_mae']) + 1)
    panels = [('Loss: MAE over all voxels', 'train_mae', 'val_mae'),
              ('Dose score: MAE inside the body', 'train_dose_score', 'val_dose_score'),
              ('DVH score: OpenKBP-style', 'train_dvh_score', 'val_dvh_score')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (title, tr, va) in zip(axes, panels):
        ax.plot(epochs, h[tr], color=TRAIN_COLOR, lw=1.8, label='train')
        ax.plot(epochs, h[va], color=VAL_COLOR, lw=1.8, label='validation')
        if best_epoch is not None and 1 <= best_epoch <= len(epochs):
            v = h[va][best_epoch - 1]
            ax.axvline(best_epoch, color=INK_2, lw=1, ls=':')
            ax.plot([best_epoch], [v], 'o', color=VAL_COLOR, ms=7, mec=INK)
            ax.text(0.97, 0.96, 'best epoch {}: val {:.3f}'.format(best_epoch, v), transform=ax.transAxes,
                    ha='right', va='top', fontsize=9, color=INK)       # top right: the curves fall, so it is empty there
        ax.set_xlabel('Epoch', color=INK)
        ax.set_ylabel('Gy (PTV mean = 60 Gy)', color=INK)
        ax.grid(True, color=GRID_INK, lw=0.8)
        ax.set_axisbelow(True)
        _style_axes(ax)
        ax.set_title(title, loc='left', fontsize=11, color=INK)
    axes[0].legend(frameon=False)
    best = '' if best_epoch is None else '; best epoch {} (selected by {})'.format(best_epoch, select_by)
    fig.suptitle('Training curves: loss and OpenKBP-style scores (lower is better){}'.format(best), fontsize=12,
                 color=INK, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------------------------------------------
# Demo: fake prediction
# --------------------------------------------------------------------------------------------------------------
def main():
    from scipy.ndimage import gaussian_filter
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', nargs='?', default='0617-634370_0')
    ap.add_argument('--out_dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      'results', 'prediction_demo'))
    ap.add_argument('--sigma', type=float, default=3.0, help='gaussian blur (voxels) used as the fake prediction')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    case = load_case(args.case, normalize=True)
    rng = np.random.default_rng(0)
    pred = gaussian_filter(case.dose, args.sigma) * 0.95 + rng.normal(0, 0.3, case.dose.shape).astype(np.float32)
    pred = np.clip(pred, 0, None).astype(np.float32)          # a real network ends with ReLU: no negative dose
    print('fake prediction MAE vs clinical: {:.3f} Gy'.format(float(np.abs(pred - case.dose).mean())))

    out = lambda kind: os.path.join(args.out_dir, '{}_{}.png'.format(case.case_id, kind))
    print(plot_dvh_comparison(case, pred, out('dvh_comparison')))
    print(plot_dose_comparison(case, pred, out('dose_comparison')))
    print(plot_overlay_pair(case, pred, args.out_dir))
    tr = 6 * np.exp(-np.arange(30) / 8) + 2
    va = tr + 0.5 + 0.2 * np.sin(np.arange(30))
    fake = {'epoch': np.arange(1, 31), 'train_mae': tr, 'val_mae': va, 'train_dose_score': 1.5 * tr,
            'val_dose_score': 1.5 * va, 'train_dvh_score': 3 * tr, 'val_dvh_score': 3 * va}
    print(plot_training_curves(fake, out('learning_curves'), best_epoch=22))


if __name__ == '__main__':
    main()

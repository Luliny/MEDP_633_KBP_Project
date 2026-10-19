"""Cumulative DVHs (HIST_CLINICAL stored in the npz) with a table of RTOG 0617 dose metrics, saved as PNG.

Example:
    conda run -n AI_RTP python visualization/vis_dvh.py 0617-634370_0 0617-622033_0
    conda run -n AI_RTP python visualization/vis_dvh.py 0617-634370_0 --normalize --show_sig
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtog_vis_utils import EXAMPLE_CASES, OUT_DIR, STRUCTURES, dvh_metrics, load_case

INK, INK_2, GRID_INK = '#0b0b0b', '#52514e', '#e4e3df'


def plot_dvh(case, out_png, show_sig=False, dpi=150):
    bins = case.dvh_bins()
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={'width_ratios': [2.1, 1]})

    for key, label, col, color, _ in STRUCTURES:
        dvh = case.hist_clinical[:, col]
        if not np.isfinite(dvh).all() or dvh.max() <= 1e-6:   # structure not present (volume 0 -> NaN)
            continue
        ax.plot(bins, 100 * dvh, color=color, lw=1.8, label=case.display_name(key))
        if show_sig:
            ax.plot(case.sig_bins(), 100 * case.hist_sig[:, col], color=color, lw=1.0, ls='--')
    if show_sig:
        ax.plot([], [], color=INK_2, lw=1.0, ls='--', label='sigmoid DVH (HIST_SIG)')

    ax.set_xlim(0, max(85.0, float(bins.max())) if case.normalized else 85.0)
    ax.set_ylim(0, 102)
    ax.set_xlabel('Dose (Gy)', color=INK)
    ax.set_ylabel('Volume (%)', color=INK)
    ax.grid(True, color=GRID_INK, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    dose_note = 'dose x DOSE_SCALE ({:.3f}), PTV mean 60 Gy'.format(case.dose_scale) if case.normalized \
        else 'planned dose (DOSE_SCALE {:.3f} not applied)'.format(case.dose_scale)
    ax.set_title('{}: cumulative DVH (HIST_CLINICAL)\n{}'.format(case.case_id, dose_note),
                 fontsize=11, color=INK, loc='left')

    # metrics table (also the text alternative to the colour-coded curves)
    rows = [[s, m, '{:.1f} {}'.format(v, u)] for s, m, v, u in dvh_metrics(case)]
    ax_t.axis('off')
    tab = ax_t.table(cellText=rows, colLabels=['Structure', 'Metric', 'Value'], loc='upper center',
                     cellLoc='left', colLoc='left', colWidths=[0.42, 0.25, 0.33])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.35)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID_INK)
        cell.get_text().set_color(INK if r else INK_2)
        if r == 0:
            cell.get_text().set_weight('bold')
    ax_t.set_title('Dose metrics (from voxels, 128$^3$ grid)', fontsize=10, color=INK, loc='left')

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='*', default=EXAMPLE_CASES, help='case ids (e.g. 0617-634370_0) or npz paths')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--normalize', action='store_true', help='scale dose by DOSE_SCALE (PTV mean 60 Gy)')
    ap.add_argument('--show_sig', action='store_true', help='also draw the sigmoid DVHs (HIST_SIG), dashed')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for c in args.cases:
        case = load_case(c, normalize=args.normalize, dicom_root=None)   # spacing not needed for DVHs
        suffix = '_norm' if args.normalize else ''
        out = os.path.join(args.out_dir, '{}_dvh{}.png'.format(case.case_id, suffix))
        plot_dvh(case, out, show_sig=args.show_sig)
        print('{} -> {}'.format(case.case_id, out))


if __name__ == '__main__':
    main()

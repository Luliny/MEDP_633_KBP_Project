"""CT with dose colour wash and structure contours (platipy ImageVisualiser), saved as PNG.

Example:
    conda run -n AI_RTP python visualization/vis_platipy_overlay.py 0617-637899_0 0617-805545_0
    conda run -n AI_RTP python visualization/vis_platipy_overlay.py path/to/case.npz --normalize --axis z
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from platipy.imaging import ImageVisualiser
from platipy.imaging.label.utils import get_com

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtog_vis_utils import (CT_WINDOW, EXAMPLE_CASES, OUT_DIR, STRUCTURES, load_case, to_sitk)

# inferno without its black end: near-black low doses at partial alpha read as grey CT artifacts.
# platipy looks the colormap up by name, so it has to be registered.
DOSE_CMAP = 'inferno_dosewash'
if DOSE_CMAP not in matplotlib.colormaps:
    matplotlib.colormaps.register(ListedColormap(plt.cm.inferno(np.linspace(0.25, 1.0, 256)), name=DOSE_CMAP))


def plot_overlay(case, out_png, axis='ortho', dose_min_frac=0.1, dpi=150, dose_max=None):
    # dose_max: upper end of the dose colour scale (Gy); None = max of this case's dose (the default).
    # Pass the same value for two cases (e.g. clinical and predicted dose) to compare them on one scale.
    ct_img = to_sitk(case.ct, case.spacing)
    dose_img = to_sitk(case.dose, None, like=ct_img)
    masks = case.masks()
    colors = {s[0]: s[4] for s in STRUCTURES}          # dark-background colours: drawn over CT

    contours, contour_colors = {}, {}
    for key, m in masks.items():
        name = case.display_name(key)
        contours[name] = to_sitk(m.astype(np.uint8), None, like=ct_img)
        contour_colors[name] = colors[key]

    # cut through the PTV centre of mass (z, y, x index order) or the image centre without a PTV
    ptv_name = case.display_name('ptv')
    cut = get_com(contours[ptv_name]) if ptv_name in contours else None
    if axis != 'ortho' and cut is not None:
        cut = cut[{'z': 0, 'y': 1, 'x': 2}[axis]]

    ref = 60.0 if case.normalized else float(case.dose[case.ptv > 0].mean()) if case.ptv.any() else case.dose.max()
    vis = ImageVisualiser(ct_img, cut=cut, axis=axis, window=CT_WINDOW, figure_size_in=10)
    vis.add_scalar_overlay(dose_img, name='Dose (Gy)', colormap=matplotlib.colormaps[DOSE_CMAP], alpha=0.45,
                           min_value=dose_min_frac * ref, max_value=float(case.dose.max() if dose_max is None else dose_max))
    vis.add_contour(contours, color=contour_colors, linewidth=1.5)
    fig = vis.show()

    dose_note = 'dose x DOSE_SCALE ({:.3f}), PTV mean 60 Gy'.format(case.dose_scale) if case.normalized \
        else 'planned dose, PTV mean {:.1f} Gy'.format(ref)
    fig.suptitle('{}: CT, dose (< {:.0f}% of PTV mean hidden) and contours\n{}'.format(
        case.case_id, 100 * dose_min_frac, dose_note), fontsize=13, y=1.01, va='bottom')
    fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='*', default=EXAMPLE_CASES, help='case ids (e.g. 0617-637899_0) or npz paths')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--axis', default='ortho', choices=['ortho', 'x', 'y', 'z'],
                    help="'ortho' = axial + coronal + sagittal; x = sagittal, y = coronal, z = axial")
    ap.add_argument('--normalize', action='store_true', help='scale dose by DOSE_SCALE (PTV mean 60 Gy)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for c in args.cases:
        case = load_case(c, normalize=args.normalize)
        suffix = '_norm' if args.normalize else ''
        out = os.path.join(args.out_dir, '{}_overlay{}.png'.format(case.case_id, suffix))
        plot_overlay(case, out, axis=args.axis)
        print('{}: spacing {} mm ({}) -> {}'.format(case.case_id, tuple(round(s, 2) for s in case.spacing),
                                                    case.spacing_source, out))


if __name__ == '__main__':
    main()

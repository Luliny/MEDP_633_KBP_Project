"""3D structure surfaces in napari: interactive viewer, or saved screenshots from several directions.

Each structure mask is lightly smoothed and turned into a surface mesh (marching cubes) at the estimated
voxel spacing; lungs are drawn translucent so the mediastinum and PTV stay visible. An isodose surface can
be added with --isodose.

Examples:
    # save <case>_napari3d.png (anterior, left lateral and oblique views) for the example cases
    conda run -n AI_RTP python visualization/vis_napari_3d.py
    # interactive viewer with CT, dose, label volumes and surfaces (needs a desktop session)
    conda run -n AI_RTP python visualization/vis_napari_3d.py 0617-634370_0 --interactive --isodose 57

Rendering needs an OpenGL context. This workstation's shell exports QT_XCB_GL_INTEGRATION=none, which
disables OpenGL in Qt and makes napari screenshots fail; the script switches it to xcb_egl (EGL on the
X display) for its own process. A hidden viewer (show=False) renders black, so screenshots briefly open a
window on the display.
"""
import argparse
import os
import sys
import warnings

if os.environ.get('DISPLAY') and os.environ.get('QT_XCB_GL_INTEGRATION', 'none') == 'none':
    os.environ['QT_XCB_GL_INTEGRATION'] = 'xcb_egl'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy import ndimage
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rtog_vis_utils import EXAMPLE_CASES, OUT_DIR, STRUCTURES, load_case

# Patient coordinates are right-handed in (X, Y, Z) order, i.e. left-handed in napari's (Z, Y, X) order, so
# napari's (right-handed) 3D camera would show a mirror image. The data is therefore passed to napari as
# (Z, Y, X) with X flipped: (inferior->superior, anterior->posterior, patient LEFT->RIGHT). Flipping the data
# rather than the camera keeps napari's surface lighting correct. In 2D the horizontal axis is shown
# right-to-left, so axial slices are in the usual radiological view (patient right on the screen left).
ORIENTATION_3D = ('towards', 'down', 'right')            # napari default
ORIENTATION_2D = ('towards', 'down', 'left')
# view_direction (camera -> scene) and screen-up direction in the flipped (Z, Y, X) frame, checked on
# rendered anatomy (anterior: heart in front of the cord, patient's left on the viewer's right, apex up)
UP = (1, 0, 0)
LATERAL = {'L': (0, 0, 1), 'R': (0, 0, -1)}              # camera on the patient's left / right


def views_for(case):
    side = case.lung_ipsi_side if case.lung_ipsi_side in LATERAL else 'L'
    lat = np.asarray(LATERAL[side], dtype=float)
    oblique = 0.75 * np.asarray((0, 1, 0)) + 0.6 * lat + 0.3 * np.asarray((-1, 0, 0))
    name = {'L': 'left', 'R': 'right'}[side]
    return {'anterior': (0, 1, 0),
            '{} lateral (ipsilateral)'.format(name): tuple(lat),
            'anterior-{} oblique'.format(name): tuple(oblique / np.linalg.norm(oblique))}


OPACITY = {'lung_ipsi': 0.3, 'lung_cntr': 0.3}
ISODOSE_COLOR = '#f2f2f2'
BACKGROUND = '#000000'


def mask_to_mesh(mask_zyx, spacing_zyx, sigma=1.0):
    """Smoothed marching-cubes surface of a (Z, Y, X) bool mask, vertices in mm."""
    vol = ndimage.gaussian_filter(mask_zyx.astype(np.float32), sigma) if sigma else mask_zyx.astype(np.float32)
    vol = np.pad(vol, 1)                                   # closes surfaces touching the grid border
    verts, faces, _, _ = measure.marching_cubes(vol, 0.5, spacing=spacing_zyx)
    return verts - np.asarray(spacing_zyx), faces


def add_surface(viewer, verts, faces, name, color, opacity):
    # 'opaque' keeps the depth test between surfaces; translucent ones are added last
    viewer.add_surface((verts, faces), name=name, vertex_colors=np.tile(to_rgba(color), (len(verts), 1)),
                       opacity=opacity, shading='smooth', blending='opaque' if opacity >= 1 else 'translucent')


def build_viewer(case, show=True, isodose=None, interactive=False):
    import napari

    sx, sy, sz = case.spacing
    scale = (sz, sy, sx)
    zyx = lambda a: np.ascontiguousarray(np.transpose(a, (2, 1, 0))[:, :, ::-1])   # see ORIENTATION_3D

    viewer = napari.Viewer(title=case.case_id, ndisplay=3, show=show)
    viewer.camera.orientation = ORIENTATION_3D
    viewer.dims.events.ndisplay.connect(
        lambda e: setattr(viewer.camera, 'orientation', ORIENTATION_3D if e.value == 3 else ORIENTATION_2D))
    if interactive:   # volumes for slicing / toggling; hidden in the initial 3D view
        viewer.add_image(zyx(case.ct), name='CT', colormap='gray', contrast_limits=(-1000, 400),
                         scale=scale, visible=False)
        viewer.add_image(zyx(case.dose), name='Dose (Gy)', colormap='inferno', blending='additive',
                         scale=scale, visible=False)
        viewer.add_labels(zyx(case.oar), name='OAR labels', scale=scale, visible=False)
        viewer.add_labels(zyx(case.ptv), name='PTV label', scale=scale, visible=False)

    masks = case.masks()
    legend = []
    # opaque surfaces first, translucent ones (lungs, isodose) last so depth blending looks right
    for key, _, _, _, color in sorted(STRUCTURES, key=lambda s: s[0] in OPACITY):
        if key not in masks:
            continue
        verts, faces = mask_to_mesh(zyx(masks[key]), scale)
        add_surface(viewer, verts, faces, case.display_name(key), color, OPACITY.get(key, 1.0))
        legend.append((case.display_name(key), color))
    if isodose is not None and (case.dose >= isodose).any():
        verts, faces = mask_to_mesh(zyx(case.dose >= isodose), scale, sigma=0.5)
        name = '{:g} Gy isodose'.format(isodose)
        add_surface(viewer, verts, faces, name, ISODOSE_COLOR, 0.25)
        legend.append((name, ISODOSE_COLOR))
    viewer.reset_view()
    return viewer, legend


def crop_to_content(images, pad=10):
    """Crop all screenshots to the union bounding box of non-background pixels."""
    h, w = min(img.shape[0] for img in images), min(img.shape[1] for img in images)
    images = [img[:h, :w] for img in images]
    occupied = np.any([img[..., :3].max(axis=-1) > 8 for img in images], axis=0)
    rows, cols = np.where(occupied.any(axis=1))[0], np.where(occupied.any(axis=0))[0]
    if len(rows) == 0:
        return images
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, occupied.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, occupied.shape[1])
    return [img[r0:r1, c0:c1] for img in images]


def save_screenshots(case, out_png, isodose=None, size=(900, 900), dpi=150):
    from qtpy.QtWidgets import QApplication

    viewer, legend = build_viewer(case, show=True, isodose=isodose)
    try:
        viewer.window._qt_window.resize(*size)
        QApplication.processEvents()
        views = views_for(case)
        shots = []
        viewer.reset_view()                                # fit zoom/centre once; it also resets the angles
        for direction in views.values():
            with warnings.catch_warnings():            # exact lateral views trigger a harmless gimbal-lock note
                warnings.simplefilter('ignore', UserWarning)
                viewer.camera.set_view_direction(direction, UP)
            QApplication.processEvents()                   # let the surface lighting follow the camera
            shots.append(viewer.screenshot(canvas_only=True))
    finally:
        viewer.close()
    if max(s[..., :3].max() for s in shots) == 0:
        raise RuntimeError('napari rendered an empty canvas (no OpenGL context?). Run on a desktop session '
                           'or try QT_XCB_GL_INTEGRATION=xcb_glx / xcb_egl.')
    shots = crop_to_content(shots)

    fig, axes = plt.subplots(1, len(shots), figsize=(4.2 * len(shots), 5.0), facecolor=BACKGROUND)
    for ax, title, img in zip(axes, views, shots):
        ax.imshow(img[..., :3])
        ax.set_title(title, color='#c3c2b7', fontsize=11)
        ax.axis('off')
    fig.legend(handles=[Patch(facecolor=c, edgecolor='none', label=n) for n, c in legend],
               loc='lower center', ncol=len(legend), frameon=False, labelcolor='#ffffff', fontsize=10)
    fig.suptitle('{}: 3D structures (napari, lungs translucent)'.format(case.case_id),
                 color='#ffffff', fontsize=13)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_png, dpi=dpi, facecolor=BACKGROUND)
    plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='*', default=EXAMPLE_CASES, help='case ids (e.g. 0617-634370_0) or npz paths')
    ap.add_argument('--out_dir', default=OUT_DIR)
    ap.add_argument('--interactive', action='store_true', help='open the napari viewer for the first case')
    ap.add_argument('--normalize', action='store_true', help='scale dose by DOSE_SCALE (PTV mean 60 Gy)')
    ap.add_argument('--isodose', type=float, default=None, help='add an isodose surface at this dose (Gy)')
    args = ap.parse_args()

    if args.interactive:
        import napari
        case = load_case(args.cases[0], normalize=args.normalize)
        build_viewer(case, show=True, isodose=args.isodose, interactive=True)
        napari.run()
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for c in args.cases:
        case = load_case(c, normalize=args.normalize)
        out = os.path.join(args.out_dir, '{}_napari3d.png'.format(case.case_id))
        save_screenshots(case, out, isodose=args.isodose)
        print('{}: spacing {} mm ({}) -> {}'.format(case.case_id, tuple(round(s, 2) for s in case.spacing),
                                                    case.spacing_source, out))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# ADD near your other imports:
import matplotlib.patches as patches
try:
    # available on most installs
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _toolkit_inset_axes
except Exception:
    _toolkit_inset_axes = None
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # <<< NEW
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
import random
import time
import h5py
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import socket
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




def create_mask_RT(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def):
    N, H, W = LS.shape
    
    # y ∈ [0, 0.3], vertical axis
    y_vals = np.linspace(0, 0.3, H).reshape(1, H, 1)  # shape: (1, H, 1)

    # Expand geometry fields to match shape (N, H, W)
    RF_geo_top_exp = np.broadcast_to(RF_geo_bottom[:, np.newaxis, :], (N, H, W))
    RF_geo_bottom_exp = np.broadcast_to(RF_geo_top[:, np.newaxis, :], (N, H, W))


    # Top region: y_vals >= (0.3 - RF_geo_top) ➜ top of domain
    mask_top = y_vals >= (0.3 - RF_geo_top_exp)

    # Bottom region: y_vals <= RF_geo_bottom ➜ bottom of domain
    mask_bottom = y_vals <= RF_geo_bottom_exp

    # Final binary mask (float32): 1 in top or bottom region, 0 elsewhere
    mask_RT = np.where(mask_top | mask_bottom, 1.0, 0.0).astype(np.float32)

    return mask_RT


def _to_t(x, device, dtype=torch.float32):
    return torch.as_tensor(x, device=device, dtype=dtype)

def _interp_2d(xNCHW, size_hw, mode="bilinear", align_corners=True):
    # x: (N,C,H,W) -> size=(H_new, W_new)
    return F.interpolate(xNCHW, size=size_hw, mode=mode, align_corners=align_corners)

def _interp_1d_width(xNCW, W_new, mode="linear", align_corners=True):
    # x: (N,C,W) -> treat as (N,C,1,W) then resize width
    x = xNCW.unsqueeze(2)  # (N,C,1,W)
    y = F.interpolate(x, size=(1, W_new), mode="bilinear" if mode=="linear" else mode,
                      align_corners=align_corners)
    return y.squeeze(2)    # (N,C,W_new)

def create_mask_highres(
    vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def,
    extent_y=(0.0, 0.3),
    ls_threshold=1.4,
    scale=4,                 # e.g., 4 -> 480x480 from 120x120
    swap_rt_assignment=True  # keep your original mapping by default
):
    """
    High-resolution re-rasterization of your masks (no smoothing of values).
    Returns: mask_perm_hr, mask_poro_hr  with shape (N, H*scale, W*scale)
    """
    assert LS.ndim == 3, "LS must be (N,H,W)"
    N, H, W = LS.shape
    Hh, Wh = H*scale, W*scale
    y0, y1 = extent_y

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) Upscale LS to high-res and threshold (crisper central defect)
    # (This refines only the BOUNDARY placement; values remain piecewise-constant later.)
    LS_t = _to_t(LS, device).unsqueeze(1)               # (N,1,H,W)
    LS_hr = _interp_2d(LS_t, (Hh, Wh), mode="bilinear", align_corners=True).squeeze(1).cpu().numpy()
    ls_mask_hr = LS_hr > ls_threshold                   # (N,Hh,Wh) boolean

    # ---- 2) Upscale RF geometry along x (W -> Wh) for smooth top/bottom curves
    # RF_geo_* are (N, W) -> (N, Wh)
    RF_top_t    = _to_t(RF_geo_top,    device).unsqueeze(1)        # (N,1,W)
    RF_bottom_t = _to_t(RF_geo_bottom, device).unsqueeze(1)        # (N,1,W)
    RF_top_hr    = _interp_1d_width(RF_top_t,    Wh, mode="linear", align_corners=True).squeeze(1).cpu().numpy()
    RF_bottom_hr = _interp_1d_width(RF_bottom_t, Wh, mode="linear", align_corners=True).squeeze(1).cpu().numpy()

    # Broadcast to (N,Hh,Wh)
    RF_top_hr_exp    = np.broadcast_to(RF_top_hr[:,    None, :], (N, Hh, Wh))
    RF_bottom_hr_exp = np.broadcast_to(RF_bottom_hr[:, None, :], (N, Hh, Wh))

    # ---- 3) High-res y grid and RT region masks
    y_vals_hr = np.linspace(y0, y1, Hh, dtype=np.float32).reshape(1, Hh, 1)
    mask_top_hr    = y_vals_hr >= (y1 - RF_top_hr_exp)     # top cap
    mask_bottom_hr = y_vals_hr <= RF_bottom_hr_exp         # bottom cap

    # ---- 4) Expand per-sample scalars/fields to high-res
    # vec columns: [perm_C, perm_def0, poro_C, poro_def, poro_RT_top, poro_RT_bottom]
    perm_C     = np.broadcast_to(vec[:, 0].reshape(N,1,1), (N, Hh, Wh))
    perm_def0  = np.broadcast_to(vec[:, 1].reshape(N,1,1), (N, Hh, Wh))  # unused unless you want
    poro_C     = np.broadcast_to(vec[:, 2].reshape(N,1,1), (N, Hh, Wh))
    poro_def   = np.broadcast_to(vec[:, 3].reshape(N,1,1), (N, Hh, Wh))
    poro_RT_top    = np.broadcast_to(vec[:, 4].reshape(N,1,1), (N, Hh, Wh))
    poro_RT_bottom = np.broadcast_to(vec[:, 5].reshape(N,1,1), (N, Hh, Wh))

    # perm_def (Input9) may be scalar per-sample (N,) or a field (N,H,W)
    if perm_def.ndim == 1:
        perm_def_hr = np.broadcast_to(perm_def.reshape(N,1,1), (N, Hh, Wh))
    elif perm_def.ndim == 3:
        # upsample to high-res (NEAREST to keep piecewise-constant blocks if it's categorical-like)
        pd_t = _to_t(perm_def, device).unsqueeze(1)  # (N,1,H,W)
        pd_hr = _interp_2d(pd_t, (Hh, Wh), mode="nearest", align_corners=None).squeeze(1).cpu().numpy()
        perm_def_hr = pd_hr
    else:
        raise ValueError("perm_def must be shape (N,) or (N,H,W)")

    # RT_field_* may be scalar per-sample, (N,H,W), or already broadcastable.
    def to_highres_like(arr):
        if np.isscalar(arr):
            return np.full((N,Hh,Wh), float(arr), dtype=np.float32)
        arr = np.asarray(arr)
        if arr.ndim == 1:        # (N,)
            return np.broadcast_to(arr.reshape(N,1,1), (N,Hh,Wh))
        if arr.shape == (N, H, W):
            t = _to_t(arr, device).unsqueeze(1)
            return _interp_2d(t, (Hh, Wh), mode="nearest", align_corners=None).squeeze(1).cpu().numpy()
        if arr.shape == (N, Hh, Wh):
            return arr
        # Last resort: broadcast if possible
        return np.broadcast_to(arr, (N,Hh,Wh))

    RT_top_hr    = to_highres_like(RT_field_top)
    RT_bottom_hr = to_highres_like(RT_field_bottom)

    # ---- 5) Build high-res base fields (piecewise-constant)
    base_perm_hr = np.where(ls_mask_hr, perm_def_hr, perm_C)  # defect vs central value
    base_poro_hr = np.where(ls_mask_hr, poro_def,   poro_C)

    # ---- 6) Apply RT overrides at high resolution
    if swap_rt_assignment:
        # your original: TOP region -> RT_bottom; BOTTOM region -> RT_top
        mask_perm_hr = np.where(mask_top_hr,    RT_bottom_hr, base_perm_hr)
        mask_perm_hr = np.where(mask_bottom_hr, RT_top_hr,    mask_perm_hr)

        mask_poro_hr = np.where(mask_top_hr,    poro_RT_bottom, base_poro_hr)
        mask_poro_hr = np.where(mask_bottom_hr, poro_RT_top,    mask_poro_hr)
    else:
        # intuitive: TOP->RT_top; BOTTOM->RT_bottom
        mask_perm_hr = np.where(mask_top_hr,    RT_top_hr,    base_perm_hr)
        mask_perm_hr = np.where(mask_bottom_hr, RT_bottom_hr, mask_perm_hr)

        mask_poro_hr = np.where(mask_top_hr,    poro_RT_top,     base_poro_hr)
        mask_poro_hr = np.where(mask_bottom_hr, poro_RT_bottom,  mask_poro_hr)

    return mask_perm_hr.astype(np.float32), mask_poro_hr.astype(np.float32), RF_top_hr ,  RF_bottom_hr,LS_hr,ls_mask_hr




seed = 42  
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # For multi-GPU
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures consistent computation

torch.set_float32_matmul_precision('high')


def get_ensemble_MATLAB(filename, highres_scale=4):
    k=30
    with h5py.File(filename, 'r') as f:
            LS              = f['/Input3'][0:k]
            RF_geo_top      = f['/Input4'][0:k]
            RF_geo_bottom   = f['/Input5'][0:k]
            RT_field_top    = f['/Input6'][0:k]
            RT_field_bottom = f['/Input7'][0:k]
            vec             = f['/Input8'][0:k]
            perm_def        = f['/Input9'][0:k]
            num_samples=LS.shape[0]
            LS=LS.reshape(num_samples,120,120).transpose(0, 2, 1)
            RT_field_bottom=RT_field_bottom.reshape(num_samples,120,120).transpose(0, 2, 1)
            RT_field_top=RT_field_top.reshape(num_samples,120,120).transpose(0, 2, 1)
            perm_def=perm_def.reshape(num_samples,120,120).transpose(0, 2, 1)

        #    mask_perm, mask_poro= create_mask(vec,LS,RT_field_top,RT_field_bottom,RF_geo_bottom,RF_geo_top,perm_def)
            mask_RT= create_mask_RT(vec,LS,RT_field_top,RT_field_bottom,RF_geo_bottom,RF_geo_top,perm_def)
            mask_perm, mask_poro, RF_top_hr ,  RF_bottom_hr,LS_hr,ls_mask_hr= create_mask_highres(
                    vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def,
                    extent_y=(0.0, 0.3),
                    ls_threshold=1.0,
                    scale=highres_scale,
                    swap_rt_assignment=True
                )
            '''
            mask_perm, mask_poro = create_mask_refined(
                vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def,
                extent_y=(0.0, 0.3),
                ls_threshold=1.0,
                morph_open_ks=3,      # 0/1 to disable; 3–5 for crisper edges
                morph_close_ks=3,
                morph_iters=1,
                swap_rt_assignment=True  # set False if you want top->top, bottom->bottom
            )
            '''


    return mask_perm, mask_poro, LS, vec,mask_RT, RF_top_hr ,  RF_bottom_hr,LS_hr,ls_mask_hr



#Case3
perm_prior, poro_prior, LS_prior, vec, RT_prior, RT_geo_top, RT_geo_bottom,LS_hr,ls_mask_hr=get_ensemble_MATLAB('prior_ensemble.h5')





from matplotlib.colors import ListedColormap
from matplotlib.ticker import FormatStrFormatter

def plot_levelset_and_mask(
    LS_hr,
    ls_mask_hr,
    extent=(0.0, 0.3, 0.0, 0.3),
    threshold=1.0,                   # for the optional contour on the level-set
    ls_cmap='jet',
    mask_colors=('white', 'black'),  # 0 -> white, 1 -> black
    use_tex=False,                   # turn on full LaTeX if you want
    title_left=r"Level set (high-res)",
    title_right=r"Thresholded mask",
    title_fontsize=14,
    annotations_right=None,          # list of dicts (see examples below)
    savepath=None
):
    """
    Show LS_hr on the left and the binary mask on the right, with optional
    annotations over the right panel.

    annotations_right: list of dicts, each can contain:
      - text:         string (MathText/LaTeX allowed if use_tex=True)
      - xy:           tuple, arrow tip (default (0.5,0.5))
      - xycoords:     'data' | 'axes fraction' (default 'data')
      - xytext:       tuple, text position (default (0.5,0.9))
      - textcoords:   'data' | 'axes fraction' (default 'axes fraction')
      - arrowprops:   dict for arrow (arrowstyle, lw, color, connectionstyle, etc.)
      - fontsize, color, bbox, ha, va, zorder
    """
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern"],
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{xcolor}"
        })

    # Ensure mask is 0/1 ints for a clean discrete colormap
    mask = np.array(ls_mask_hr)
    if mask.dtype != np.int32 and mask.dtype != np.int64:
        # treat anything >0.5 as 1
        mask = (mask > 0.5).astype(int)

    # Figure layout: 1 row × (2 images + 1 slim colorbar for LS)
    CBAR_FRAC = 0.045
    
    fig = plt.figure(figsize=(8.6, 4.2), constrained_layout=False)
    # <<< put the colorbar column in the middle >>>
    gs  = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.0, CBAR_FRAC, 1.0],   # [LEFT, CBAR, RIGHT]
        wspace=0.06, hspace=0.0
    )
    
    axL = fig.add_subplot(gs[0, 0])   # level-set (left)
    cax = fig.add_subplot(gs[0, 1])   # colorbar in the middle
    axR = fig.add_subplot(gs[0, 2])   # mask (right)
    
    # ... draw images ...
    imL = axL.imshow(LS_hr, origin='lower', extent=extent, cmap=ls_cmap, aspect='equal')
    # (optional) contour on level-set
    # axL.contour(LS_hr_2d, levels=[threshold], origin='lower', extent=extent, colors='w', linewidths=1.5)
    
    # right panel (mask) stays the same
    # imR = axR.imshow(mask_2d, ...)
    
    # <<< colorbar belongs to the level-set image >>>
    cb = fig.colorbar(imL, cax=cax)   # attaches to LEFT image
    cb.ax.tick_params(labelsize=9)
    
    plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.08)

    # Optional contour at the threshold
    if threshold is not None:
        axL.contour(
            LS_hr, levels=[threshold], origin='lower',
            extent=extent, colors='w', linewidths=1.5, alpha=0.9
        )

    # ----- Right: thresholded mask -----
    cmap_mask = ListedColormap(mask_colors)
    imR = axR.imshow(
        mask, origin='lower', extent=extent,
        cmap=cmap_mask, vmin=0, vmax=1, interpolation='nearest', aspect='equal'
    )

    # ----- titles -----
    axL.set_title(title_left,  fontsize=title_fontsize, pad=4)
    axR.set_title(title_right, fontsize=title_fontsize, pad=4)

    # ----- remove ticks/frames for a clean look -----
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('left','right','top','bottom'):
            ax.spines[s].set_visible(False)

    # ----- colorbar for the level set only -----
    cb = fig.colorbar(imL, cax=cax)
    cb.ax.tick_params(labelsize=9)
    # If you prefer formatted ticks:
    # cb.formatter = FormatStrFormatter('%.2f'); cb.update_ticks()

    # ----- apply annotations on the RIGHT panel -----
    if annotations_right:
        for spec in annotations_right:
            axR.annotate(
                spec.get("text", r"$\mathrm{Label}$"),
                xy=spec.get("xy", (0.5, 0.5)),
                xycoords=spec.get("xycoords", "data"),
                xytext=spec.get("xytext", (0.5, 0.9)),
                textcoords=spec.get("textcoords", "axes fraction"),
                ha=spec.get("ha", "left"),
                va=spec.get("va", "center"),
                fontsize=spec.get("fontsize", 12),
                color=spec.get("color", "white"),
                bbox=spec.get("bbox", dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.2)),
                arrowprops=spec.get("arrowprops", {"arrowstyle": "->", "lw": 1.4, "color": "white"}),
                clip_on=False,
                zorder=spec.get("zorder", 30)
            )

    # margins
    plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.08)
#    xmin, xmax, ymin, ymax = extent
#    axL.set_xlim(xmin, xmax)
#    axL.set_ylim(ymin, ymax)
    
    from matplotlib.patches import Rectangle

#    axL.add_patch(Rectangle(
#        (xmin, ymin), xmax - xmin, ymax - ymin,
#        fill=False, edgecolor="k", linewidth=1.5, zorder=10
#    ))

    xmin, xmax, ymin, ymax = extent
    axR.add_patch(Rectangle(
        (xmin, ymin), xmax - xmin, ymax - ymin,
        fill=False, edgecolor="k", linewidth=1.5, zorder=10
    ))
    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_levelset_and_mask2(
    LS_hr_2d, ls_mask_2d,
    extent=(0.0, 0.3, 0.0, 0.3),
    threshold=1.0,
    ls_cmap='jet',
    mask_colors=('white','black'),
    # NEW:
    panel_gap_frac=0.10,         # <- visible gap between panels (as a fraction of a data panel)
    label_fontsize=12,
    tick_labelsize=11,
    add_labels=True,
    x_label=r"$x$", y_label=r"$y$",
    title_left=r"Level set (high-res)",
    title_right=r"Thresholded mask",
    title_fontsize=18,
    annotations_right=None,
    savepath=None
):
    """
    LS_hr_2d, ls_mask_2d are 2D arrays (e.g. LS_hr[idx], ls_mask_hr[idx]).
    """
    # --- geometry from extent ---
    xmin, xmax, ymin, ymax = extent
    Lx = xmax - xmin
    Ly = ymax - ymin

    # --- discreet colormap for mask ---
    mask = (np.asarray(ls_mask_2d) > 0.5).astype(int)
    cmap_mask = ListedColormap(mask_colors)

    # --- layout: [LEFT | CBAR | GAP | RIGHT] ---
    CBAR_FRAC   = 0.045
    GAP_FRAC    = max(0.02, float(panel_gap_frac))  # avoid zero
    fig = plt.figure(figsize=(8.8, 4.3), constrained_layout=False)
    gs  = fig.add_gridspec(
        nrows=1, ncols=4,
        width_ratios=[1.0, CBAR_FRAC, GAP_FRAC, 1.0],
        wspace=0.02, hspace=0.0
    )

    axL = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])   # colorbar for left image
    axR = fig.add_subplot(gs[0, 3])   # right panel (mask)

    # --- LEFT: level set ---
    imL = axL.imshow(
        LS_hr_2d, origin='lower', extent=extent,
        cmap=ls_cmap, aspect='equal'
    )
    if threshold is not None:
        # draw contour in data coords so it matches extent
        ny, nx = np.asarray(LS_hr_2d).shape[:2]
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys, indexing='xy')
        axL.contour(X, Y, LS_hr_2d, levels=[threshold], colors='w', linewidths=1.4)

    # --- RIGHT: thresholded mask ---
    axR.imshow(
        mask, origin='lower', extent=extent,
        cmap=cmap_mask, vmin=0, vmax=1, interpolation='nearest', aspect='equal'
    )

    # --- frame, limits, ticks & labels on BOTH axes ---
    for ax in (axL, axR):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        # endpoint ticks only
        ax.set_xticks([xmin, xmax])
        ax.set_yticks([ymin, ymax])
        # LaTeX-ish tick labels for endpoints
        ax.set_xticklabels([r"$0$", r"$D_x$"], fontsize=tick_labelsize)
        ax.set_yticklabels([r"$0$", r"$D_y$"], fontsize=tick_labelsize)
        # show a neat rectangular box
        for side in ("left","right","top","bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color("k")

    # axis labels (set on both; remove on axR if you prefer only left)
    if add_labels:
        axL.set_xlabel(x_label, fontsize=label_fontsize)
        axL.set_ylabel(y_label, fontsize=label_fontsize)
        axR.set_xlabel(x_label, fontsize=label_fontsize)
        axR.set_ylabel(y_label, fontsize=label_fontsize)

    # titles
    axL.set_title(title_left,  fontsize=title_fontsize,  pad=4)
    axR.set_title(title_right, fontsize=title_fontsize, pad=4)

    # colorbar tied to LEFT image
    cb = fig.colorbar(imL, cax=cax)
    cb.ax.tick_params(labelsize=9)

    # optional annotations on RIGHT
    if annotations_right:
        for spec in annotations_right:
            axR.annotate(
                spec.get("text", r"$\mathrm{Label}$"),
                xy=spec.get("xy", (0.5,0.5)),
                xycoords=spec.get("xycoords","data"),
                xytext=spec.get("xytext",(0.5,0.9)),
                textcoords=spec.get("textcoords","axes fraction"),
                ha=spec.get("ha","left"), va=spec.get("va","center"),
                fontsize=spec.get("fontsize",12),
                color=spec.get("color","white"),
                bbox=spec.get("bbox", dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.2)),
                arrowprops=spec.get("arrowprops", {"arrowstyle":"->","lw":1.4,"color":"white"}),
                clip_on=False, zorder=30
            )

    plt.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.10)

    if savepath:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Assuming your call returned:
# mask_perm, mask_poro, RF_top_hr, RF_bottom_hr, LS_hr, ls_mask_hr = create_mask_highres(...)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
})

ann = [
    {
        "text": r"$\mathcal{C}$",
        "xy": (0.20, 0.08),            # arrow tip (in data coords)
        "xycoords": "data",
        "xytext": (0.70, 0.40),        # label position (axes fraction here)
        "textcoords": "axes fraction",
        "arrowprops": {"arrowstyle": "-|>", "lw": 1.6, "color": "k"},
        "bbox": {"facecolor": "white", "alpha": 0.0, "edgecolor": "none", "pad": 1.0},
        "color": "white", "fontsize": 16
    },
    
    {
        "text": r"$\mathcal{C}$",
        "xy": (0.40, 0.8),            # arrow tip (in data coords)
        "xycoords": "axes fraction",
        "xytext": (0.40, 0.80),        # label position (axes fraction here)
        "textcoords": "axes fraction",
        "arrowprops": {"arrowstyle": "-|>", "lw": 1.6, "color": "k"},
        "bbox": {"facecolor": "white", "alpha": 0.0, "edgecolor": "none", "pad": 1.0},
        "color": "white", "fontsize": 16
    },
    
    
    {
        "text": r"$\mathcal{C}$",
        "xy": (0.20, 0.29),            # arrow tip (in data coords)
        "xycoords": "axes fraction",
        "xytext": (0.2, 0.38),        # label position (axes fraction here)
        "textcoords": "axes fraction",
        "arrowprops": {"arrowstyle": "-|>", "lw": 1.6, "color": "k"},
        "bbox": {"facecolor": "white", "alpha": 0.0, "edgecolor": "none", "pad": 1.0},
        "color": "red", "fontsize": 16
    },



]

i = 25  # pick the sample

# If these are torch tensors, convert to NumPy first:
LS2   = np.asarray(LS_hr[i])
MASK2 = np.asarray(ls_mask_hr[i])

plot_levelset_and_mask2(
    LS2, MASK2,
    extent=(0.0, 0.3, 0.0, 0.3),
    threshold=1.0,
    panel_gap_frac=0.15,          # increase/decrease spacing between panels
    add_labels=True,
    x_label=r"$x$", y_label=r"$y$",
#    annotations_right=ann,
    title_left=r"$L(x,y)$",
    title_right=r"$\mathbb{I}_{\{L(x,y)>1\}}$",
    savepath="../Visualise/Figures/levelset_vs_mask.jpg"
)


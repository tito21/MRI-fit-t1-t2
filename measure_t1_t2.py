#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "matplotlib>=3.10.6",
#    "numpy>=2.3.3",
#    "pydicom>=3.0.1",
#    "scikit-image>=0.25.2",
#    "scipy>=1.16.2",
#    "tqdm>=4.67.1"
# ]
# ///

import pathlib

import numpy as np
import pydicom
from scipy.optimize import curve_fit
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import tqdm


T1_range = (0, 2000)  # Typical T1 values in ms
T2_range = (0, 200)  # Typical T2 values in ms


# --- Model functions ---
def t1_model(TI, S0, T1, k=-1.0):
    return np.abs(S0 * (1 - (1 - k) * np.exp(-TI / T1)))

def t1_model_jac(TI, S0, T1, k=-1.0):
    E1 = np.exp(-TI / T1)
    inner = 1 - (1 - k) * E1
    dS0 = abs(inner)
    dT1 = inner * S0 * TI * (1 - k) * E1 / (T1 ** 2 * abs(inner))
    dk = S0 * inner * E1 / abs(inner)
    return np.array([dS0, dT1, dk]).T

def t2_model(TE, S0, T2):
    return S0 * np.exp(-TE / T2)

def t2_model_jac(TE, S0, T2):
    E2 = np.exp(-TE / T2)
    dS0 = E2
    dT2 = S0 * TE * E2 / (T2 ** 2)
    return np.array([dS0, dT2]).T

# --- Load DICOM images ---
def load_dicom_series(folder):
    files = list(pathlib.Path(folder).glob('**/*.dcm'))
    images = []
    times = []
    for f in files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array.astype(np.float32)
        images.append(img)
        # For T1: use Inversion Time (TI), for T2: use Echo Time (TE)
        ti_metadata = ds.get('SharedFunctionalGroupsSequence', None)
        if ti_metadata:
            # print(metadata)
            params = ti_metadata[0].get('MRModifierSequence', None)
            ti = params[0].get('InversionTimes', 0.0) if params else 0.0

        te_metadaata = ds.get('PerFrameFunctionalGroupsSequence', None)
        if te_metadaata:
            # print(metadata)
            params = te_metadaata[0].get('MREchoSequence', None)
            te = params[0].get('EffectiveEchoTime', None) if params else None

        times.append((ti, te))
    images = np.stack(images)

    print(times)
    return images, times

def estimate_initial_t1(time_points, images):
    # Use the image with the darkest image (zero crossing) to estimate T1
    org_shape = images.shape
    images = images.reshape(images.shape[0], -1)  # Flatten spatial dimensions
    min_signal_idx = np.argmin(images, axis=0)  # Index of the minimum signal for each voxel
    time_points = time_points.reshape(-1, 1)  # Ensure time_points is a column vector
    min_signal_time = time_points[min_signal_idx]
    t1_est = min_signal_time / np.log(2)  # Rough estimate based on the zero crossing point
    return images[-1].reshape(org_shape[1:]), t1_est.reshape(org_shape[1:])  # Reshape back to original spatial dimensions


def estimate_initial_t2(time_points, images):
    # OLS estimate for T2
    origin_shape = images.shape
    images = images.reshape(images.shape[0], -1)  # Flatten spatial dimensions
    log_signal = np.log(images + 1e-6)  # Avoid log(0)
    A = np.vstack([time_points, np.ones_like(time_points)]).T
    slope, intercept = np.linalg.lstsq(A, log_signal, rcond=None)[0]
    t20 = -1 / slope
    s0 = np.exp(intercept)
    return s0.reshape(origin_shape[1:]), t20.reshape(origin_shape[1:])  # Reshape back to original spatial dimensions

# --- Fit T1 and T2 voxel-wise ---
def fit_relaxation(images, times, mode='T1'):
    shape = images.shape[1:]
    n_images = images.shape[0]
    t_map = np.zeros(shape)
    s0_map = np.zeros(shape)
    residual_map = np.zeros(shape)

    time_points = np.array([t[0] if mode == 'T1' else t[1] for t in times])

    idx = np.argsort(time_points)
    time_points = time_points[idx]
    images = images[idx, :, :]

    # Apply Otsu's thresholding to create a mask
    thresh = threshold_otsu(images[idx[0], :, :])
    mask = images[idx[0], :, :] < thresh

    # Get time points
    if mode == 'T1':
        s0, t10 = estimate_initial_t1(time_points, images)
        model = t1_model
        model_jac = t1_model_jac
        p0 = np.zeros(s0.shape + (3,))  # Initial guess: S0, T1, k
        p0[..., 0] = s0  # S0
        p0[..., 1] = t10  # T1
        p0[..., 2] = -1.0  # k
        bounds = ([0, 0, -1.0], [np.inf, np.inf, 1.0])  # S0 > 0, T1 > 0, k in [-1.0, 1.0]
        k_map = np.zeros(shape)  # Map to store k values for debugging
    else:
        s0, t20 = estimate_initial_t2(time_points, images)
        model = t2_model
        model_jac = t2_model_jac
        p0 = np.zeros(s0.shape + (2,))  # Initial guess: S0, T2
        p0[..., 0] = s0  # S0
        p0[..., 1] = t20  # T2
        bounds = ([0, 0], [np.inf, np.inf])  # S0 > 0, T2 > 0

    print("Time points:", time_points)
    print("Number of images:", n_images)
    # Fit each voxel
    bar = tqdm.tqdm(total=shape[0]*shape[1], desc=f'Fitting {mode} map')
    for i in range(shape[0]):
        for j in range(shape[1]):
            signal = images[:, i, j]
            try:
                popt, _ = curve_fit(model, time_points, signal, p0=p0[i, j], maxfev=5000, jac=model_jac, bounds=bounds)
                s0_map[i, j], t_map[i, j] = popt[0], popt[1]
                if mode == 'T1':
                    k_map[i, j] = popt[2]

            except Exception:
                s0_map[i, j], t_map[i, j] = p0[i, j][0], p0[i, j][1]  # Fallback to initial guess if fitting fails
                popt = p0[i, j]

            residual_map[i, j] = np.sum((signal - model(time_points, *popt)) ** 2)
            bar.update(1)
    bar.close()

    # Clip values to reasonable ranges
    s0_map = np.clip(s0_map, 0, np.max(images))

    if mode == 'T1':
        maps = {
            't_map': t_map,
            's0_map': s0_map,
            'k_map': k_map
        }
    else:
        maps = {
            't_map': t_map,
            's0_map': s0_map
        }
    return maps, residual_map, mask


def plot_montage(images, titles, rows=None, cols=None):
    if cols is None:
        cols = int(np.ceil(np.sqrt(len(images))))
    if rows is None:
        rows = int(np.ceil(len(images) / cols))
    vmax = np.max([img.max() for img in images])
    vmin = np.min([img.min() for img in images])
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = axes.flatten() if len(images) > 1 else [axes]
    for i in range(len(images)):
        axes[i].imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    plt.tight_layout()
    return fig

# --- Main script ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Measure T1 and T2 from DICOM MRI images')
    parser.add_argument('folder', help='Folder containing DICOM images')
    parser.add_argument('--mode', choices=['T1', 'T2'], default='T1', help='Relaxation type to measure')
    parser.add_argument('--output', type=pathlib.Path, default='.', help='Output path for the numpy arrays')
    parser.add_argument("--range", type=float, nargs=2, help="Display range for the relaxation map")
    parser.add_argument('--debug', action='store_true', help='Show intermediate results for debugging')
    args = parser.parse_args()

    images, times = load_dicom_series(args.folder)

    if args.debug:
        titles = [f'TI={t[0]} ms, TE={t[1]} ms' for t in times]
        plot_montage(images, titles)
        # plt.imshow(montage(images), cmap='gray')
        plt.show()

    maps, residual_map, mask = fit_relaxation(images, times, mode=args.mode)

    if args.range:
        display_range = tuple(args.range)
    else:
        display_range = T1_range if args.mode == 'T1' else T2_range

    # Save and show results
    if args.debug:
        plt.imshow(residual_map, cmap='hot', vmax=np.percentile(residual_map, 99))  # Clip to 99th percentile for better visualization
        plt.title('Residual map')
        plt.colorbar()
        plt.savefig(args.output / f'{args.mode}_residual_map.png', dpi=300)
        plt.show()

        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.show()

        if args.mode == 'T1':
            plt.imshow(maps['k_map'], cmap='bwr', vmin=-1, vmax=1)
            plt.title('k map')
            plt.colorbar()
            plt.show()

    np.savez(args.output / f'{args.mode}_map.npz', **maps, residual_map=residual_map, mask=mask)
    t_map = maps['t_map']
    s0_map = maps['s0_map']
    s0_map[mask] = np.nan
    t_map[mask] = np.nan
    plt.subplot(1, 2, 1)
    plt.imshow(t_map, cmap='hot')
    plt.clim(*display_range)
    plt.title(f'{args.mode} map')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(s0_map, cmap='gray')
    plt.title('S0 map')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(args.output / f'{args.mode}_map.png', dpi=300)
    plt.show()

    plt.imshow(residual_map, cmap='hot', vmax=np.percentile(residual_map, 99))  # Clip to 99th percentile for better visualization
    plt.title('Residual map')
    plt.colorbar()
    plt.savefig(args.output / f'{args.mode}_residual_map.png', dpi=300)

    print(f'{args.mode}: {np.nanmean(t_map)} +/- {np.nanstd(t_map)} ms')
    print(f'{args.mode} map saved as {args.output / f"{args.mode}_map.npz"}')

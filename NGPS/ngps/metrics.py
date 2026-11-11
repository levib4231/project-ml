from __future__ import annotations

import numpy as np
import scipy.ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, pred, data_range=1.0))


def ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    window = min(7, min(ref.shape))
    window -= 1 - window % 2
    if window < 3:
        raise ValueError("SSIM needs spatial dimensions >= 3")
    return float(structural_similarity(ref, pred, data_range=1.0, win_size=window))


def hfen(ref: np.ndarray, pred: np.ndarray, sigma: float = 1.5) -> float:
    ref_log = scipy.ndimage.gaussian_laplace(ref, sigma=sigma)
    pred_log = scipy.ndimage.gaussian_laplace(pred, sigma=sigma)
    return float(np.linalg.norm(ref_log - pred_log) / (np.linalg.norm(ref_log) + 1e-8))


def gmsd(ref: np.ndarray, pred: np.ndarray) -> float:
    hx = np.array([[1 / 3, 0, -1 / 3]] * 3)
    hy = hx.T
    ref_d = scipy.ndimage.uniform_filter(ref, size=2)[::2, ::2]
    pred_d = scipy.ndimage.uniform_filter(pred, size=2)[::2, ::2]
    grad_ref = np.sqrt(scipy.ndimage.convolve(ref_d, hx) ** 2 + scipy.ndimage.convolve(ref_d, hy) ** 2)
    grad_pred = np.sqrt(scipy.ndimage.convolve(pred_d, hx) ** 2 + scipy.ndimage.convolve(pred_d, hy) ** 2)
    c = 0.0026
    return float(np.std((2 * grad_ref * grad_pred + c) / (grad_ref**2 + grad_pred**2 + c)))


def gradient_magnitude_similarity(ref: np.ndarray, pred: np.ndarray) -> float:
    """Sobel magnitude similarity weighted by max magnitude; NOT FSIM."""

    gx_ref = scipy.ndimage.sobel(ref, axis=1)
    gy_ref = scipy.ndimage.sobel(ref, axis=0)
    gx_pred = scipy.ndimage.sobel(pred, axis=1)
    gy_pred = scipy.ndimage.sobel(pred, axis=0)
    mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
    mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    c = 1e-4
    sim = (2 * mag_ref * mag_pred + c) / (mag_ref**2 + mag_pred**2 + c)
    weight = np.maximum(mag_ref, mag_pred)
    if weight.sum() == 0:
        return 1.0
    return float((sim * weight).sum() / weight.sum())


METRICS = {"PSNR": psnr, "SSIM": ssim, "HFEN": hfen,
           "SobelGradientMagnitudeSimilarity": gradient_magnitude_similarity, "GMSD": gmsd}


def volume_metrics(ref, pred):
    from datasets.volume_folder import validate_volume
    validate_volume(ref, unit_range=True)
    validate_volume(pred)
    if ref.shape != pred.shape:
        raise ValueError(f"Shape mismatch: {ref.shape} != {pred.shape}")
    return {name: float(np.mean([fn(r, p) for r, p in zip(ref, pred)]))
            for name, fn in METRICS.items()}


def aggregate_subjects(rows):
    """Equal subject weights, not pooled slices. Call separately per sigma."""
    if not rows or len({r["subject_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty unique subject rows")
    result = {}
    for name in METRICS:
        values = np.asarray([r[name] for r in rows], dtype=np.float64)
        result[name] = {"mean": float(values.mean()),
                        "std_population": float(values.std()) if np.isfinite(values).all() else None,
                        "n_subjects": len(rows)}
    return result

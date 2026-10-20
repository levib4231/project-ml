from __future__ import annotations

import numpy as np
import scipy.ndimage


METRIC_NAMES = ("psnr", "ssim", "fsim_proxy", "hfen", "gmsd")


def _pair(ref: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(ref, dtype=np.float64)
    prediction = np.asarray(pred, dtype=np.float64)
    if reference.shape != prediction.shape:
        raise ValueError(f"Metric shape mismatch: ref={reference.shape}, pred={prediction.shape}")
    if reference.ndim != 2:
        raise ValueError(f"Slice metrics require 2D arrays, got {reference.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(prediction).all():
        raise ValueError("Metrics require finite arrays.")
    return reference, prediction


def psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    reference, prediction = _pair(ref, pred)
    mse = float(np.mean((reference - prediction) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    """Single-channel SSIM using the standard 7x7 uniform-window constants."""

    reference, prediction = _pair(ref, pred)
    win_size = min(7, reference.shape[0], reference.shape[1])
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        raise ValueError("SSIM requires both slice dimensions to be at least 3.")
    ux = scipy.ndimage.uniform_filter(reference, size=win_size)
    uy = scipy.ndimage.uniform_filter(prediction, size=win_size)
    uxx = scipy.ndimage.uniform_filter(reference * reference, size=win_size)
    uyy = scipy.ndimage.uniform_filter(prediction * prediction, size=win_size)
    uxy = scipy.ndimage.uniform_filter(reference * prediction, size=win_size)
    covariance_normalization = win_size**2 / (win_size**2 - 1)
    vx = covariance_normalization * (uxx - ux * ux)
    vy = covariance_normalization * (uyy - uy * uy)
    vxy = covariance_normalization * (uxy - ux * uy)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * ux * uy + c1) * (2 * vxy + c2)
    denominator = (ux**2 + uy**2 + c1) * (vx + vy + c2)
    score = numerator / denominator
    pad = (win_size - 1) // 2
    if pad:
        score = score[pad:-pad, pad:-pad]
    return float(score.mean())


def hfen(ref: np.ndarray, pred: np.ndarray, sigma: float = 1.5) -> float:
    reference, prediction = _pair(ref, pred)
    ref_log = scipy.ndimage.gaussian_laplace(reference, sigma=sigma)
    pred_log = scipy.ndimage.gaussian_laplace(prediction, sigma=sigma)
    return float(np.linalg.norm(ref_log - pred_log) / (np.linalg.norm(ref_log) + 1e-8))


def gmsd(ref: np.ndarray, pred: np.ndarray) -> float:
    reference, prediction = _pair(ref, pred)
    hx = np.array([[1 / 3, 0, -1 / 3]] * 3)
    hy = hx.T
    ref_d = scipy.ndimage.uniform_filter(reference, size=2)[::2, ::2]
    pred_d = scipy.ndimage.uniform_filter(prediction, size=2)[::2, ::2]
    grad_ref = np.sqrt(scipy.ndimage.convolve(ref_d, hx) ** 2 + scipy.ndimage.convolve(ref_d, hy) ** 2)
    grad_pred = np.sqrt(scipy.ndimage.convolve(pred_d, hx) ** 2 + scipy.ndimage.convolve(pred_d, hy) ** 2)
    c = 0.0026
    return float(np.std((2 * grad_ref * grad_pred + c) / (grad_ref**2 + grad_pred**2 + c)))


def fsim_proxy(ref: np.ndarray, pred: np.ndarray) -> float:
    """Project-local gradient-similarity proxy; this is not reference FSIM."""

    reference, prediction = _pair(ref, pred)
    gx_ref = scipy.ndimage.sobel(reference, axis=1)
    gy_ref = scipy.ndimage.sobel(reference, axis=0)
    gx_pred = scipy.ndimage.sobel(prediction, axis=1)
    gy_pred = scipy.ndimage.sobel(prediction, axis=0)
    mag_ref = np.sqrt(gx_ref**2 + gy_ref**2)
    mag_pred = np.sqrt(gx_pred**2 + gy_pred**2)
    c = 1e-4
    similarity = (2 * mag_ref * mag_pred + c) / (mag_ref**2 + mag_pred**2 + c)
    weight = np.maximum(mag_ref, mag_pred)
    weight_sum = float(weight.sum())
    return float((similarity * weight).sum() / weight_sum) if weight_sum else float(similarity.mean())


def compute_slice_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {
        "psnr": psnr(ref, pred),
        "ssim": ssim(ref, pred),
        "fsim_proxy": fsim_proxy(ref, pred),
        "hfen": hfen(ref, pred),
        "gmsd": gmsd(ref, pred),
    }


def compute_subject_metrics(ref: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    reference = np.asarray(ref)
    prediction = np.asarray(pred)
    if reference.shape != prediction.shape or reference.ndim != 3:
        raise ValueError(f"Subject metrics require matching (D, H, W) volumes: {reference.shape}, {prediction.shape}")
    rows = [compute_slice_metrics(reference[index], prediction[index]) for index in range(reference.shape[0])]
    return {name: float(np.mean([row[name] for row in rows])) for name in METRIC_NAMES}

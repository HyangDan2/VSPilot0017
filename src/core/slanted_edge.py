import numpy as np
import cv2

from .esf_utils import smooth_1d

def pca_line_orientation(edge_points):
    pts = np.asarray(edge_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])
    centroid = pts.mean(axis=0)
    X = pts - centroid
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    direction = Vt[0]
    direction /= (np.linalg.norm(direction) + 1e-12)
    return direction, centroid

def extract_edge_points(img_gray_float01):
    img = (img_gray_float01 * 255.0).astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thresh = np.percentile(mag, 95.0)
    mask = mag >= max(thresh, 1e-6)
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1)
    return pts

def resample_strip_along_normal(img_gray_f01, centroid, tangent, normal, half_len=40, half_width=12):
    us = np.linspace(-half_len, half_len, 2 * half_len + 1)
    vs = np.linspace(-half_width, half_width, 2 * half_width + 1)
    U, V = np.meshgrid(us, vs)
    tx, ty = tangent
    nx, ny = normal
    X = centroid[0] + U * tx + V * nx
    Y = centroid[1] + U * ty + V * ny
    strip = cv2.remap(img_gray_f01, X.astype(np.float32), Y.astype(np.float32),
                      interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    strip = np.clip(strip.astype(np.float32), 0.0, 1.0)
    return strip

def robust_esf_from_hist(
    t_vals, intens, oversample=4,
    trim_bins=2, min_frac=0.05, smooth_sigma=1.0,
    tail_guard=True
):
    s = int(max(1, oversample))
    t_min = np.percentile(t_vals, 1) - 1.0
    t_max = np.percentile(t_vals, 99) + 1.0
    nbins = int(np.ceil((t_max - t_min) * s))
    nbins = max(64, nbins)

    edges = np.linspace(t_min, t_max, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(t_vals, bins=edges)
    sums,   _ = np.histogram(t_vals, bins=edges, weights=intens)

    with np.errstate(divide='ignore', invalid='ignore'):
        esf = sums / counts

    mask_valid = np.isfinite(esf)
    if np.count_nonzero(mask_valid) >= 4:
        if not np.all(mask_valid):
            esf = np.interp(centers, centers[mask_valid], esf[mask_valid])
            counts = np.where(mask_valid, counts, 0)
    else:
        raise ValueError("ESF accumulation failed: insufficient valid samples.")

    radius_px = max(1, int(round(3 * smooth_sigma)))
    trim_edge_bins = max(trim_bins, radius_px * s)
    keep = np.ones_like(esf, dtype=bool)
    keep[:trim_edge_bins] = False
    keep[-trim_edge_bins] = False

    medc = np.median(counts[counts > 0]) if np.any(counts > 0) else 0
    thresh = max(1, int(medc * min_frac))
    keep &= counts >= thresh

    if np.any(keep):
        valid_idx = np.where(keep)[0]
        last_good = valid_idx[-1]
        ker = np.array([1, 2, 3, 2, 1], dtype=float); ker /= ker.sum()
        counts_sm = np.convolve(counts.astype(float), ker, mode='same')
        r_idx = np.where(counts_sm >= thresh)[0]
        if r_idx.size > 0:
            last_good = min(last_good, r_idx[-1])
        last_good = max(last_good - 2, 0)
        keep[(last_good + 1):] = False

    if np.count_nonzero(keep) < 16:
        n = esf.size
        left = int(0.2 * n); right = int(0.8 * n)
        keep = np.zeros(n, dtype=bool); keep[left:right] = True

    centers_k = centers[keep]
    esf_k = esf[keep]

    if tail_guard and esf_k.size >= 20:
        tail_len = max(5, esf_k.size // 10)
        head = esf_k[:-tail_len]
        tail = esf_k[-tail_len:].copy()
        tail = np.maximum.accumulate(tail)
        if head.size > 0 and tail[0] < head[-1]:
            tail += (head[-1] - tail[0])
        esf_k = np.concatenate([head, tail])

    return centers_k, esf_k, counts[keep], s

def slanted_edge_esf_lsf_mtf(
    roi_bgr: np.ndarray,
    pixel_pitch_mm=0.005,
    oversample=4,
    smooth_sigma=1.0,
    normalize_esf=True,
    inv_gamma=False,
    gamma=2.2,
    channel_mode="Y",           # "Y" | "G" | "GRAY"
    window_type="Hann",         # "Hann" | "Hamming"
    derivative_mode="Central",  # "Central" | "FirstDiff"
    pixel_box_correction=False  # sinc correction
):
    # Channel selection
    if roi_bgr.ndim == 3:
        bgr = roi_bgr
        if channel_mode == "G":
            roi_gray_u8 = bgr[:, :, 1]
        elif channel_mode == "Y":
            Y = 0.299 * bgr[:, :, 2] + 0.587 * bgr[:, :, 1] + 0.114 * bgr[:, :, 0]
            roi_gray_u8 = np.clip(Y, 0, 255).astype(np.uint8)
        else:
            roi_gray_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        roi_gray_u8 = roi_bgr.astype(np.uint8)

    gray_f01 = roi_gray_u8.astype(np.float32) / 255.0
    gray_proc = np.power(np.clip(gray_f01, 0.0, 1.0), float(gamma)) if inv_gamma else gray_f01
    intens_for_hist = gray_proc if inv_gamma else roi_gray_u8.astype(np.float32)

    # Edge orientation via PCA
    pts = extract_edge_points(gray_proc)
    if pts.shape[0] < 10:
        raise ValueError("Not enough edge points in ROI. Please select a clearer slanted edge.")
    tangent, centroid = pca_line_orientation(pts)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    normal /= (np.linalg.norm(normal) + 1e-12)

    # Align brighter → +t
    h, w = gray_proc.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    px = xs.astype(np.float64) + 0.5
    py = ys.astype(np.float64) + 0.5
    t_vals = (px - centroid[0]) * normal[0] + (py - centroid[1]) * normal[1]
    band = np.abs(t_vals) < 10.0
    if np.sum(band) > 0:
        corr = np.corrcoef(t_vals[band].ravel(), (gray_proc if inv_gamma else gray_f01)[band].ravel())[0, 1]
        if corr < 0:
            normal = -normal
            t_vals = -t_vals

    centers, esf_raw, counts_k, s = robust_esf_from_hist(
        t_vals.ravel(), intens_for_hist.ravel(),
        oversample=oversample, trim_bins=2, min_frac=0.05,
        smooth_sigma=smooth_sigma, tail_guard=True
    )

    # Contrast metrics (pre-normalization)
    Imin = float(np.percentile(esf_raw, 1))
    Imax = float(np.percentile(esf_raw, 99))
    delta_I = Imax - Imin
    contrast = (Imax - Imin) / (Imax + Imin) if (Imax + Imin) > 1e-9 else 0.0

    # ESF normalization (for display)
    if normalize_esf:
        lo = np.percentile(esf_raw, 1)
        hi = np.percentile(esf_raw, 99)
        if hi > lo:
            esf_base = (esf_raw - lo) / (hi - lo)
        else:
            rng = max(1e-9, (esf_raw.max() - esf_raw.min()))
            esf_base = (esf_raw - esf_raw.min()) / rng
    else:
        esf_base = esf_raw.copy()

    esf_s = smooth_1d(esf_base, sigma=smooth_sigma)

    # LSF
    dx_mm = pixel_pitch_mm / float(s)
    if derivative_mode == "FirstDiff":
        lsf_core = np.convolve(esf_s, np.array([1.0, -1.0], dtype=np.float64), mode='valid') / dx_mm
        lsf = lsf_core
        N = lsf.size
        x_lsf = (np.arange(N) - (N // 2)) * dx_mm
    else:
        lsf = np.gradient(esf_s, dx_mm)
        N = lsf.size
        x_lsf = (np.arange(N) - (N // 2)) * dx_mm

    if N < 16:
        raise ValueError("LSF is too short. Please select a larger ROI.")

    # Window
    win = np.hamming(N) if window_type == "Hamming" else np.hanning(N)
    lsf_w = lsf * win

    # FFT → MTF
    mtf = np.abs(np.fft.rfft(lsf_w))
    if mtf[0] != 0:
        mtf = mtf / mtf[0]
    freq_mm = np.fft.rfftfreq(N, d=dx_mm)     # cycles/mm
    freq_pix = freq_mm * pixel_pitch_mm       # cycles/pixel

    # Pixel box correction (sinc)
    if pixel_box_correction:
        f_pix = freq_pix.copy()
        eps = 1e-12
        box = np.sinc(f_pix + eps)
        box = np.clip(box, 1e-3, None)
        mtf = mtf / box
        if mtf[0] != 0:
            mtf = mtf / mtf[0]

    strip = resample_strip_along_normal(gray_f01, centroid, tangent, normal, half_len=40, half_width=12)

    return {
        "esf_x_pix": centers,
        "esf": esf_s,
        "lsf_x_mm": x_lsf,
        "lsf": lsf_w,
        "freq_cyc_per_mm": freq_mm,
        "freq_cyc_per_pix": freq_pix,
        "mtf": mtf,
        "strip": strip,
        "delta_I": delta_I,
        "contrast": contrast,
    }

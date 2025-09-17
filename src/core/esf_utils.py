import numpy as np

def gaussian_kernel1d(sigma: float, radius: int):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k

def smooth_1d(y: np.ndarray, sigma=1.0):
    radius = max(1, int(round(3 * sigma)))
    if radius < 1:
        return y.copy()
    k = gaussian_kernel1d(sigma, radius)
    ypad = np.pad(y.astype(np.float64), (radius, radius), mode='reflect')
    ys = np.convolve(ypad, k, mode='valid')
    return ys

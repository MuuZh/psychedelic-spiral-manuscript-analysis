"""Measure the rotation angle (omega) of spiral waves around each vortex centroid.

Equivalent to MATLAB measure_spiral_rotation.m.
"""

import numpy as np
from scipy.signal import convolve2d


def measure_spiral_rotation(centroids_pos, centroids_neg, phase_map):
    """Measure the rotation angle of spiral wavefronts near each vortex.

    Uses Sobel edge detection to find wavefront pixels, then computes the
    circular mean angle of nearby wavefront points in polar coordinates
    relative to each vortex centroid.

    Parameters
    ----------
    centroids_pos : ndarray, shape (N_pos, 2)
        Positive vortex centroids in (x, y) = (col, row) format.
    centroids_neg : ndarray, shape (N_neg, 2)
        Negative vortex centroids in (x, y) = (col, row) format.
    phase_map : ndarray, shape (H, W)
        Phase map at the current time slice.

    Returns
    -------
    omega_pos : ndarray, shape (N_pos,)
        Rotation angles for positive vortices.
    omega_neg : ndarray, shape (N_neg,)
        Rotation angles for negative vortices.
    """
    # Sobel kernels (same as MATLAB code)
    kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]], dtype=np.float64)

    # Edge detection on phase map
    EdgeX = convolve2d(phase_map, kx, mode='same', boundary='fill')
    EdgeY = convolve2d(phase_map, ky, mode='same', boundary='fill')
    waveFront = np.sqrt(EdgeX ** 2 + EdgeY ** 2)
    wavefront_mask = waveFront > 2 * np.pi

    # Wavefront point coordinates: (row, col) from np.where -> convert to (x, y)
    wy, wx = np.where(wavefront_mask)  # wy = row indices, wx = col indices
    wx = wx.astype(np.float64)
    wy = wy.astype(np.float64)

    omega_pos = _measure_rotation_for_centroids(centroids_pos, wx, wy)
    omega_neg = _measure_rotation_for_centroids(centroids_neg, wx, wy)

    return omega_pos, omega_neg


def _measure_rotation_for_centroids(centroids, wx, wy):
    """Compute rotation angle for each centroid from nearby wavefront points."""
    n = centroids.shape[0]
    omega = np.zeros(n)

    for idx in range(n):
        cx, cy = centroids[idx]  # (col, row)

        # First try radius 3
        theta, _ = _polar_within_radius(cx, cy, wx, wy, radius=3)

        if len(theta) <= 2:
            # Mark as invalid (MATLAB sets centroid to NaN)
            omega[idx] = 0.0
            continue

        # Expand to radius 5 (matches MATLAB logic)
        theta, _ = _polar_within_radius(cx, cy, wx, wy, radius=5)

        # Circular mean angle
        sin_sum = np.sum(np.sin(theta))
        cos_sum = np.sum(np.cos(theta))
        omega[idx] = np.arctan2(sin_sum, cos_sum)

    return omega


def _polar_within_radius(cx, cy, wx, wy, radius):
    """Convert wavefront points to polar coords relative to (cx, cy) and filter by radius."""
    dx = wx - cx
    dy = wy - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    mask = r <= radius
    theta = np.arctan2(dy[mask], dx[mask])
    return theta, r[mask]

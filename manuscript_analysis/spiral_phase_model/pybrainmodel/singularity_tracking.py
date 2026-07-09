"""Detect vortex singularities in a phase gradient field.

Equivalent to the MATLAB singularity_tracking.m (pattDetection_v5).
Only the single-frame detection used by the reconstruction pipeline is implemented;
multi-frame tracking and graph-edge construction are omitted.
"""

import numpy as np
from skimage.measure import label, regionprops


def detect_singularities(vPhaseX, vPhaseY, phase_map, v_threshold=1.0):
    """Detect vortex singularities and return their centroids and rotation labels.

    Matches the MATLAB call:
        detected_spirals = singularity_tracking(-vPhaseX, -vPhaseY, phaseSig)
    Note: this function expects the *raw* phase gradients (without sign flip).
    The double negation in the MATLAB pipeline is handled internally.

    Parameters
    ----------
    vPhaseX, vPhaseY : ndarray, shape (H, W)
        Phase gradient components (raw, before any sign flip).
    phase_map : ndarray, shape (H, W)
        Phase map (unused here but kept for API consistency).

    Returns
    -------
    centroids_pos : ndarray, shape (N_pos, 2)
        Centroids of positive (counter-clockwise) vortices in (x, y) = (col, row) format.
    centroids_neg : ndarray, shape (N_neg, 2)
        Centroids of negative (clockwise) vortices in (x, y) = (col, row) format.
    """
    H, W = vPhaseX.shape

    # --- Normalise phase gradient vectors ---
    # MATLAB: passes -vPhaseX then negates again -> net effect is vPhaseX/mag
    magnitude = np.sqrt(vPhaseX ** 2 + vPhaseY ** 2)
    magnitude[magnitude == 0] = 1.0  # avoid division by zero
    Vx_norm = vPhaseX / magnitude
    Vy_norm = vPhaseY / magnitude

    # --- Compute vorticity (curl of normalised field) ---
    # curl = dVy/dx - dVx/dy  (central differences, unit grid spacing)
    dVy_dx = np.gradient(Vy_norm, axis=1)
    dVx_dy = np.gradient(Vx_norm, axis=0)
    vorticity = dVy_dx - dVx_dy

    # --- Threshold: |vorticity| > 1 ---
    vorticity_mask = np.abs(vorticity) > v_threshold

    # --- Connected components (8-connectivity) ---
    labels = label(vorticity_mask, connectivity=2)

    centroids_pos = []
    centroids_neg = []

    for region in regionprops(labels, intensity_image=vorticity):
        region_mask = labels == region.label

        # Skip mixed-rotation (interacting) patches
        region_vort = vorticity[region_mask]
        if np.any(region_vort > 0) and np.any(region_vort < 0):
            continue

        # Peak vorticity determines rotation
        peak_idx = np.argmax(np.abs(region_vort))
        peak_val = region_vort[peak_idx]

        # Centroid: skimage returns (row, col); convert to (x, y) = (col, row)
        # MATLAB's regionprops WeightedCentroid with binary intensity == regular centroid
        cy, cx = region.centroid  # row, col
        centroid_xy = [cx, cy]

        if peak_val > 0:
            centroids_pos.append(centroid_xy)
        else:
            centroids_neg.append(centroid_xy)

    centroids_pos = np.array(
        centroids_pos) if centroids_pos else np.empty((0, 2))
    centroids_neg = np.array(
        centroids_neg) if centroids_neg else np.empty((0, 2))

    return centroids_pos, centroids_neg

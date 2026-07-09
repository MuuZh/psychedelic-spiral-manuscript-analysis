"""Utility functions for phase field analysis."""

import numpy as np


def anglesubtract(a, b):
    """Circular subtraction: wraps (a - b) to [-pi, pi].
    
    Equivalent to MATLAB's anglesubtract / angle(exp(1i*(a-b))).
    """
    return np.angle(np.exp(1j * (a - b)))


def compute_phase_gradient(phase):
    """Compute the phase gradient field using central differences with circular subtraction.
    
    Matches the MATLAB pipeline's gradient computation:
        vPhaseX(iX, 2:end-1) = anglesubtract(phase(iX, 3:end), phase(iX, 1:end-2)) / 2
        vPhaseY(2:end-1, iY) = anglesubtract(phase(3:end, iY), phase(1:end-2, iY)) / 2
    
    Parameters
    ----------
    phase : ndarray, shape (H, W)
        Phase map in radians.
    
    Returns
    -------
    vPhaseX : ndarray, shape (H, W)
        Phase gradient along the x (column) direction.
    vPhaseY : ndarray, shape (H, W)
        Phase gradient along the y (row) direction.
    """
    H, W = phase.shape
    vPhaseX = np.zeros_like(phase)
    vPhaseY = np.zeros_like(phase)

    # X gradient: central differences along columns
    vPhaseX[:, 1:-1] = anglesubtract(phase[:, 2:], phase[:, :-2]) / 2.0

    # Y gradient: central differences along rows
    vPhaseY[1:-1, :] = anglesubtract(phase[2:, :], phase[:-2, :]) / 2.0

    return vPhaseX, vPhaseY

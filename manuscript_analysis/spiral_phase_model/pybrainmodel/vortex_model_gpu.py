"""GPU-accelerated vortex-based phase vector field model and optimisation.

Equivalent to MATLAB vortex_based_phase_vector_field_model.m and
optimize_radial_influence.m, but uses PyTorch autograd instead of finite
differences for dramatically faster convergence.
"""

import numpy as np
import torch
import torch.nn as nn
import time
from tqdm import tqdm


class VortexPhaseModel(nn.Module):
    """Model the phase field on masked pixels only.

    Instead of computing on the full (H, W) grid, we extract the P masked
    pixel coordinates into 1-D vectors and work exclusively on those.
    The forward() output is still (H, W) with NaN outside the mask for
    easy plotting.

    Parameters
    ----------
    height, width : int
        Full grid dimensions (e.g. 175 x 251).
    centroids_pos, centroids_neg : ndarray, shape (N, 2)
        Vortex centroids in (x, y) = (col, row) format.
    omega_pos, omega_neg : ndarray, shape (N,)
        Rotation angles for each vortex.
    mask : ndarray, shape (H, W), bool
        Region of interest.  Only these pixels are modelled.
        If None, the full grid is used.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, height, width, centroids_pos, centroids_neg,
                 omega_pos, omega_neg, precomputed_grid=None, mask=None, device='cuda'):
        super().__init__()
        self.device = torch.device(device)
        self.H = height
        self.W = width

        if precomputed_grid is not None:
            # Reuse precomputed grid tensors without copying.
            self.register_buffer('mask', precomputed_grid['mask'])
            self.register_buffer('X_m', precomputed_grid['X_m'])
            self.register_buffer('Y_m', precomputed_grid['Y_m'])
            self.P = int(self.X_m.shape[0])
        else:
            # Full coordinate grid
            col = torch.arange(width, dtype=torch.float32, device=self.device)
            row = torch.arange(height, dtype=torch.float32, device=self.device)
            X_full, Y_full = torch.meshgrid(col, row, indexing='xy')  # (H, W)

            # Mask
            if mask is not None:
                mask_t = torch.tensor(
                    mask, dtype=torch.bool, device=self.device)
            else:
                mask_t = torch.ones(
                    height, width, dtype=torch.bool, device=self.device)
            self.register_buffer('mask', mask_t)
            # Extract masked pixel coordinates -> 1-D vectors of length P
            self.register_buffer('X_m', X_full[mask_t])  # (P,)
            self.register_buffer('Y_m', Y_full[mask_t])  # (P,)
            self.P = int(self.X_m.shape[0])
            # print(
            #     f"VortexPhaseModel: {self.P} masked pixels out of {height * width}")

        # --- Positive (counter-clockwise, m=+1) vortices ---
        self.n_pos = centroids_pos.shape[0] if centroids_pos.size > 0 else 0
        if self.n_pos > 0:
            self.register_buffer('x0_pos', torch.tensor(
                centroids_pos[:, 0], dtype=torch.float32, device=self.device))
            self.register_buffer('y0_pos', torch.tensor(
                centroids_pos[:, 1], dtype=torch.float32, device=self.device))
            self.register_buffer('omega_p', torch.tensor(
                omega_pos, dtype=torch.float32, device=self.device))
            self.sigma_pos = nn.Parameter(
                10.0 * torch.ones(self.n_pos, device=self.device))

        # --- Negative (clockwise, m=-1) vortices ---
        self.n_neg = centroids_neg.shape[0] if centroids_neg.size > 0 else 0
        if self.n_neg > 0:
            self.register_buffer('x0_neg', torch.tensor(
                centroids_neg[:, 0], dtype=torch.float32, device=self.device))
            self.register_buffer('y0_neg', torch.tensor(
                centroids_neg[:, 1], dtype=torch.float32, device=self.device))
            self.register_buffer('omega_n', torch.tensor(
                omega_neg, dtype=torch.float32, device=self.device))
            self.sigma_neg = nn.Parameter(
                10.0 * torch.ones(self.n_neg, device=self.device))

    def forward(self):
        """Compute the model phase field on masked pixels only.

        Returns
        -------
        Model_phase : Tensor, shape (H, W)
            Phase values at masked pixels; NaN elsewhere.
        """
        X_comp = torch.zeros(self.P, device=self.device)
        Y_comp = torch.zeros(self.P, device=self.device)

        if self.n_neg > 0:
            X_comp, Y_comp = self._accumulate(
                X_comp, Y_comp,
                self.x0_neg, self.y0_neg, self.omega_n, self.sigma_neg,
                charge=-1.0,
            )

        if self.n_pos > 0:
            X_comp, Y_comp = self._accumulate(
                X_comp, Y_comp,
                self.x0_pos, self.y0_pos, self.omega_p, self.sigma_pos,
                charge=1.0,
            )

        # atan2 with epsilon on X_comp to stabilise gradient near zero
        phase_m = torch.atan2(Y_comp, X_comp + 1e-8)  # (P,)

        # Scatter back into full (H, W) grid
        out = torch.full((self.H, self.W), float('nan'), device=self.device)
        out[self.mask] = phase_m
        return out

    # ------------------------------------------------------------------ #

    def _accumulate(self, X_comp, Y_comp, x0, y0, omega, sigma, charge):
        """Add contributions from a set of vortices.

        Shapes: x0 (N,), X_m (P,)
        Broadcast: (N, 1) vs (1, P) -> (N, P), then sum over N -> (P,).
        """
        X_rel = self.X_m.unsqueeze(0) - x0[:, None]  # (N, P)
        Y_rel = self.Y_m.unsqueeze(0) - y0[:, None]  # (N, P)

        r_sq = X_rel ** 2 + Y_rel ** 2
        theta = torch.atan2(Y_rel, X_rel) - omega[:, None]
        theta = torch.remainder(theta, 2 * torch.pi) - torch.pi

        gaussian = torch.exp(-r_sq / (2.0 * sigma[:, None] ** 2))
        phase_change = charge * theta

        X_comp = X_comp + (gaussian * torch.cos(phase_change)).sum(dim=0)
        Y_comp = Y_comp + (gaussian * torch.sin(phase_change)).sum(dim=0)
        return X_comp, Y_comp


# ====================================================================== #
#  Cost function
# ====================================================================== #

def phase_correlation_cost(model_phase, target_phase, mask=None):
    """1 - R2 where R is Pearson correlation between model and target phase.

    Matches MATLAB computeCost: cost = 1 - r^2 with
        r = nansum(a.*b) / sqrt(nansum(a.*a) * nansum(b.*b))
    """
    if mask is not None:
        mp = model_phase[mask]
        tp = target_phase[mask]
        n_nans_in_np = torch.isnan(mp).sum().item()
        n_nans_in_tp = torch.isnan(tp).sum().item()
        if n_nans_in_np > 0 or n_nans_in_tp > 0:
            print(
                f"Warning: Found {n_nans_in_np} NaNs in model_phase and {n_nans_in_tp} NaNs in target_phase after masking")
    else:
        mp = model_phase.reshape(-1)
        tp = target_phase.reshape(-1)

    a = tp - tp.mean()
    b = mp - mp.mean()
    r = (a * b).sum() / (torch.sqrt((a * a).sum() * (b * b).sum()) + 1e-12)
    return 1.0 - r ** 2


# ====================================================================== #
#  Optimisation
# ====================================================================== #

def optimize_radial_influence(model, target_phase_np, iterations=350,
                              lr=0.1, optimizer_cls=None, verbose=True):
    device = model.device
    target = torch.tensor(target_phase_np, dtype=torch.float32, device=device)

    if optimizer_cls is None:
        optimizer_cls = torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=lr)

    cost_history = []
    sigma_pos_history = []
    sigma_neg_history = []
    t0 = time.time()
    # pbar = tqdm(range(iterations), desc="Optimizing Sigma")

    for i in range(iterations):
        optimizer.zero_grad()
        model_phase = model()

        # ====================================================
        # Compute the smooth cosine loss used for optimization.
        # ====================================================
        # This temporary loss is only used for gradients.
        if model.mask is not None:
            diff = model_phase[model.mask] - target[model.mask]
        else:
            diff = model_phase.reshape(-1) - target.reshape(-1)

        # Remove NaN values to keep gradients finite.
        diff = diff[~torch.isnan(diff)]

        if diff.numel() > 0:
            # Cosine Loss: 1 - mean(cos(diff))
            # Range is [0, 2] with smooth gradients.
            loss_optim = 1.0 - torch.cos(diff).mean()
        else:
            loss_optim = torch.tensor(0.0, device=device, requires_grad=True)

        # Backpropagate with smooth cosine gradients.
        loss_optim.backward()
        optimizer.step()

        # ====================================================
        # Compute the reporting cost using the original Pearson metric.
        # ====================================================
        # Avoid storing gradients for the reporting metric.
        with torch.no_grad():
            # Use the original phase_correlation_cost function.
            cost_display = phase_correlation_cost(
                model_phase, target, model.mask)
            cv = cost_display.item()
            if model.n_pos > 0:
                sigma_pos_history.append(
                    model.sigma_pos.detach().cpu().numpy().copy())
            if model.n_neg > 0:
                sigma_neg_history.append(
                    model.sigma_neg.detach().cpu().numpy().copy())

        # Keep the reported history as Pearson cost for consistency.
        cost_history.append(cv)

        # ====================================================
        # Print progress.
        # ====================================================
        # if verbose and ((i + 1) % 50 == 0 or i == 0):
        #     elapsed = time.time() - t0
        # R2 here is the standard Pearson R2.
        # print(f"Iter {i + 1:4d}/{iterations}  "
        #       f"Pearson Cost={cv:.6f}  R2={1 - cv:.6f}  "
        #       f"time={elapsed:.1f}s")

        # # Extract final sigma values
        # sigma_pos = model.sigma_pos.detach().cpu(
        # ).numpy() if model.n_pos > 0 else np.array([])
        # sigma_neg = model.sigma_neg.detach().cpu(
        # ).numpy() if model.n_neg > 0 else np.array([])

        # # The returned variance_captured is still based on Pearson correlation.
        # variance_captured = 1.0 - cost_history[-1]

    return np.array(sigma_pos_history), np.array(sigma_neg_history), np.array(cost_history)

# def optimize_radial_influence(model, target_phase_np, iterations=350,
#                               lr=1.0, optimizer_cls=None, verbose=True):
#     """Optimise sigma parameters using GPU-accelerated autograd.

#     Parameters
#     ----------
#     model : VortexPhaseModel
#         Model with sigma_pos / sigma_neg as learnable parameters.
#     target_phase_np : ndarray, shape (H, W)
#         Observed phase map (numpy).
#     iterations : int
#         Number of optimisation steps.
#     lr : float
#         Learning rate (default 1.0 works well with Adam).
#     optimizer_cls : torch.optim class, optional
#         Defaults to Adam.
#     verbose : bool
#         Print progress every 50 iterations.

#     Returns
#     -------
#     sigma_pos : ndarray
#     sigma_neg : ndarray
#     cost_history : list[float]
#     variance_captured : float   (= R2)
#     """
#     device = model.device
#     target = torch.tensor(target_phase_np, dtype=torch.float32, device=device)

#     if optimizer_cls is None:
#         optimizer_cls = torch.optim.Adam
#     optimizer = optimizer_cls(model.parameters(), lr=lr)

#     cost_history = []
#     t0 = time.time()
#     pbar = tqdm(range(iterations), desc="Optimizing Sigma")

#     for i in pbar:
#         optimizer.zero_grad()
#         model_phase = model()
#         # Debug print
#         # print(f"Iteration {i + 1}/{iterations} - Computing cost...")
#         cost = phase_correlation_cost(model_phase, target, model.mask)
#         if torch.isnan(cost):
#             print(f"Warning: NaN cost at iteration {i + 1}. Skipping update.")
#         cost.backward()
#         optimizer.step()

#         cv = cost.item()
#         cost_history.append(cv)
#         # print(
#         #     f"Iteration {i + 1}/{iterations} - Cost: {cv:.6f}, sigma_pos: {model.sigma_pos.data.cpu().numpy() if model.n_pos > 0 else 'N/A'}, sigma_neg: {model.sigma_neg.data.cpu().numpy() if model.n_neg > 0 else 'N/A'}")

#         if verbose and ((i + 1) % 50 == 0 or i == 0):
#             elapsed = time.time() - t0
#             print(f"Iter {i + 1:4d}/{iterations}  "
#                   f"cost={cv:.6f}  R2={1 - cv:.6f}  r={np.sqrt(max(1 - cv, 0)):.6f}  "
#                   f"time={elapsed:.1f}s")

    # # Extract final sigma values
    # sigma_pos = model.sigma_pos.detach().cpu(
    # ).numpy() if model.n_pos > 0 else np.array([])
    # sigma_neg = model.sigma_neg.detach().cpu(
    # ).numpy() if model.n_neg > 0 else np.array([])
    # variance_captured = 1.0 - cost_history[-1]

    # return sigma_pos, sigma_neg, cost_history, variance_captured

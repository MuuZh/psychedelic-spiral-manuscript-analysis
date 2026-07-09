from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


@dataclass(frozen=True)
class SpiralFocus:
    row: float
    col: float
    determinant: float
    trace: float
    stable: bool
    winding_number: float


class SpiralWaveDetector:
    """Detect stable spiral-wave foci from phase via optical flow + Jacobian tests."""

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 10.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        winding_radius: int = 2,
        winding_min: float = 0.8,
        stable_only: bool = True,
        merge_distance: float = 2.0,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.winding_radius = int(winding_radius)
        self.winding_min = float(winding_min)
        self.stable_only = bool(stable_only)
        self.merge_distance = float(merge_distance)
        if self.winding_radius < 1:
            raise ValueError(f"winding_radius must be >= 1, got {self.winding_radius}")
        if self.merge_distance < 0:
            raise ValueError(f"merge_distance must be >= 0, got {self.merge_distance}")

    @staticmethod
    def _phase_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Circular subtraction in [-pi, pi)."""
        return (a - b + np.pi) % (2.0 * np.pi) - np.pi

    @classmethod
    def _phase_gradient_axis(cls, arr: np.ndarray, axis: int) -> np.ndarray:
        """MATLAB gradient-style angular derivative without periodic boundaries."""
        arr = np.asarray(arr, dtype=np.float64)
        deriv = np.full_like(arr, np.nan, dtype=np.float64)
        n = arr.shape[axis]
        if n < 2:
            return deriv

        left = [slice(None)] * arr.ndim
        right = [slice(None)] * arr.ndim
        left[axis] = 0
        right[axis] = 1
        deriv[tuple(left)] = cls._phase_diff(arr[tuple(right)], arr[tuple(left)])

        last = [slice(None)] * arr.ndim
        prev = [slice(None)] * arr.ndim
        last[axis] = n - 1
        prev[axis] = n - 2
        deriv[tuple(last)] = cls._phase_diff(arr[tuple(last)], arr[tuple(prev)])

        if n > 2:
            mid = [slice(None)] * arr.ndim
            plus = [slice(None)] * arr.ndim
            minus = [slice(None)] * arr.ndim
            mid[axis] = slice(1, n - 1)
            plus[axis] = slice(2, n)
            minus[axis] = slice(0, n - 2)
            deriv[tuple(mid)] = cls._phase_diff(arr[tuple(plus)], arr[tuple(minus)]) * 0.5

        finite = np.isfinite(arr)
        valid = np.isfinite(deriv) & finite
        return np.where(valid, deriv, np.nan)

    @classmethod
    def _temporal_derivative(cls, phase: np.ndarray, frame_idx: int | None = None) -> np.ndarray:
        """Angular temporal derivative with no wrap across movie/window ends."""
        if frame_idx is None:
            out = np.full_like(phase, np.nan, dtype=np.float64)
            for t in range(phase.shape[2]):
                out[:, :, t] = cls._temporal_derivative(phase, t)
            return out

        t = int(frame_idx)
        frames = phase.shape[2]
        if t < 0 or t >= frames:
            raise IndexError(f"frame_idx out of range: {t}, valid [0, {frames - 1}]")

        if t > 1 and t + 2 < frames:
            out = (
                8.0 * cls._phase_diff(phase[:, :, t + 1], phase[:, :, t - 1])
                - cls._phase_diff(phase[:, :, t + 2], phase[:, :, t - 2])
            ) / 12.0
        elif t > 0 and t + 1 < frames:
            out = cls._phase_diff(phase[:, :, t + 1], phase[:, :, t - 1]) * 0.5
        elif t + 1 < frames:
            out = cls._phase_diff(phase[:, :, t + 1], phase[:, :, t])
        elif t > 0:
            out = cls._phase_diff(phase[:, :, t], phase[:, :, t - 1])
        else:
            out = np.full(phase.shape[:2], np.nan, dtype=np.float64)

        return np.where(np.isfinite(out), out, np.nan)

    def compute_derivatives(self, phase_data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spatial and temporal angular derivatives without wrap-around."""
        if phase_data.ndim != 3:
            raise ValueError(f"phase_data must be (H, W, T), got shape={phase_data.shape}")

        phase = np.asarray(phase_data, dtype=np.float64)
        return (
            self._phase_gradient_axis(phase, axis=1),
            self._phase_gradient_axis(phase, axis=0),
            self._temporal_derivative(phase),
        )

    @staticmethod
    def _avg_neighbors(field: np.ndarray) -> np.ndarray:
        """Spatial 4-neighbor average without periodic boundaries."""
        field = np.asarray(field, dtype=np.float64)
        out = np.zeros_like(field, dtype=np.float64)
        counts = np.zeros_like(field, dtype=np.float64)

        out[1:, ...] += field[:-1, ...]
        counts[1:, ...] += 1.0
        out[:-1, ...] += field[1:, ...]
        counts[:-1, ...] += 1.0
        out[:, 1:, ...] += field[:, :-1, ...]
        counts[:, 1:, ...] += 1.0
        out[:, :-1, ...] += field[:, 1:, ...]
        counts[:, :-1, ...] += 1.0

        return np.divide(out, counts, out=np.zeros_like(out), where=counts > 0)

    @staticmethod
    def _linear_gradient_axis(arr: np.ndarray, axis: int) -> np.ndarray:
        """MATLAB gradient-style non-angular derivative."""
        arr = np.asarray(arr, dtype=np.float64)
        deriv = np.full_like(arr, np.nan, dtype=np.float64)
        n = arr.shape[axis]
        if n < 2:
            return deriv

        left = [slice(None)] * arr.ndim
        right = [slice(None)] * arr.ndim
        left[axis] = 0
        right[axis] = 1
        deriv[tuple(left)] = arr[tuple(right)] - arr[tuple(left)]

        last = [slice(None)] * arr.ndim
        prev = [slice(None)] * arr.ndim
        last[axis] = n - 1
        prev[axis] = n - 2
        deriv[tuple(last)] = arr[tuple(last)] - arr[tuple(prev)]

        if n > 2:
            mid = [slice(None)] * arr.ndim
            plus = [slice(None)] * arr.ndim
            minus = [slice(None)] * arr.ndim
            mid[axis] = slice(1, n - 1)
            plus[axis] = slice(2, n)
            minus[axis] = slice(0, n - 2)
            deriv[tuple(mid)] = (arr[tuple(plus)] - arr[tuple(minus)]) * 0.5

        return deriv

    @staticmethod
    def _sparse_spatial_operators(shape: tuple[int, int]) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
        """Build row/col derivative and 4-neighbor Laplacian operators."""
        h, w = shape
        n = h * w
        drow_rows: list[int] = []
        drow_cols: list[int] = []
        drow_data: list[float] = []
        dcol_rows: list[int] = []
        dcol_cols: list[int] = []
        dcol_data: list[float] = []
        lap_rows: list[int] = []
        lap_cols: list[int] = []
        lap_data: list[float] = []

        def idx(r: int, c: int) -> int:
            return r * w + c

        for r in range(h):
            for c in range(w):
                p = idx(r, c)

                if h > 1:
                    if r == 0:
                        row_terms = [(idx(0, c), -1.0), (idx(1, c), 1.0)]
                    elif r == h - 1:
                        row_terms = [(idx(h - 2, c), -1.0), (idx(h - 1, c), 1.0)]
                    else:
                        row_terms = [(idx(r - 1, c), -0.5), (idx(r + 1, c), 0.5)]
                    for q, val in row_terms:
                        drow_rows.append(p)
                        drow_cols.append(q)
                        drow_data.append(val)

                if w > 1:
                    if c == 0:
                        col_terms = [(idx(r, 0), -1.0), (idx(r, 1), 1.0)]
                    elif c == w - 1:
                        col_terms = [(idx(r, w - 2), -1.0), (idx(r, w - 1), 1.0)]
                    else:
                        col_terms = [(idx(r, c - 1), -0.5), (idx(r, c + 1), 0.5)]
                    for q, val in col_terms:
                        dcol_rows.append(p)
                        dcol_cols.append(q)
                        dcol_data.append(val)

                degree = 0
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < h and 0 <= cc < w:
                        lap_rows.append(p)
                        lap_cols.append(idx(rr, cc))
                        lap_data.append(1.0)
                        degree += 1
                lap_rows.append(p)
                lap_cols.append(p)
                lap_data.append(-float(degree))

        drow = sparse.csr_matrix((drow_data, (drow_rows, drow_cols)), shape=(n, n))
        dcol = sparse.csr_matrix((dcol_data, (dcol_rows, dcol_cols)), shape=(n, n))
        lap = sparse.csr_matrix((lap_data, (lap_rows, lap_cols)), shape=(n, n))
        return drow, dcol, lap

    @staticmethod
    def _weighted_surround_terms(
        smooth_p: np.ndarray,
        drow_op: sparse.csr_matrix,
        dcol_op: sparse.csr_matrix,
        lap_op: sparse.csr_matrix,
    ) -> sparse.csr_matrix:
        """Approximate MATLAB surroundTerms for nonlinear smoothness penalty."""
        smooth_vec = smooth_p.ravel()
        ps_col = SpiralWaveDetector._linear_gradient_axis(smooth_p, axis=1).ravel()
        ps_row = SpiralWaveDetector._linear_gradient_axis(smooth_p, axis=0).ravel()
        return (
            sparse.diags(ps_row, 0, format="csr") @ drow_op
            + sparse.diags(ps_col, 0, format="csr") @ dcol_op
            + sparse.diags(smooth_vec, 0, format="csr") @ lap_op
        )

    def _matlab_charbonnier_velocity(
        self,
        ex: np.ndarray,
        ey: np.ndarray,
        et: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fixed-point nonlinear Charbonnier optical-flow solve, mirroring opticalFlowStep."""
        h, w = ex.shape
        n = h * w
        drow_op, dcol_op, lap_op = self._sparse_spatial_operators((h, w))

        ex = np.where(valid, ex, 0.0)
        ey = np.where(valid, ey, 0.0)
        et = np.where(valid, et, 0.0)
        ex_v = ex.ravel()
        ey_v = ey.ravel()
        et_v = et.ravel()

        u = np.zeros((h, w), dtype=np.float64)
        v = np.zeros((h, w), dtype=np.float64)
        data_e = np.full((h, w), np.inf, dtype=np.float64)
        smooth_e = np.full((h, w), np.inf, dtype=np.float64)
        beta = max(float(self.beta), np.finfo(np.float64).eps)
        alpha = max(float(self.alpha), np.finfo(np.float64).eps)
        relax = 1.1
        relax_min = 0.2
        relax_step = 0.02
        max_change = max(float(self.tol), 0.01)

        for _ in range(self.max_iter):
            last_data_e = data_e
            last_smooth_e = smooth_e

            data_e = ex * u + ey * v + et
            up_col = self._linear_gradient_axis(u, axis=1)
            up_row = self._linear_gradient_axis(u, axis=0)
            vp_col = self._linear_gradient_axis(v, axis=1)
            vp_row = self._linear_gradient_axis(v, axis=0)
            smooth_e = up_col * up_col + up_row * up_row + vp_col * vp_col + vp_row * vp_row

            data_p = 0.5 / beta / np.sqrt(beta * beta + data_e * data_e)
            smooth_p = 0.5 / beta / np.sqrt(beta * beta + smooth_e)
            data_p = np.where(valid, data_p, 0.0)
            smooth_p = np.where(valid, smooth_p, 0.0)

            with np.errstate(divide="ignore", invalid="ignore"):
                data_change = np.abs(data_e - last_data_e) / np.maximum(np.abs(data_e), 1e-12)
                smooth_change = np.abs(smooth_e - last_smooth_e) / np.maximum(np.abs(smooth_e), 1e-12)
            if (
                np.isfinite(data_change).any()
                and np.isfinite(smooth_change).any()
                and float(np.nanmax(data_change)) < max_change
                and float(np.nanmax(smooth_change)) < max_change
            ):
                break

            gamma = (data_p / alpha).ravel()
            surround = self._weighted_surround_terms(smooth_p, drow_op, dcol_op, lap_op)

            diag_u = -(ex_v * ex_v * gamma)
            diag_v = -(ey_v * ey_v * gamma)
            uv_diag = -(ex_v * ey_v * gamma)

            a11 = surround + sparse.diags(diag_u, 0, shape=(n, n), format="csr")
            a22 = surround + sparse.diags(diag_v, 0, shape=(n, n), format="csr")
            a12 = sparse.diags(uv_diag, 0, shape=(n, n), format="csr")
            a21 = sparse.diags(uv_diag, 0, shape=(n, n), format="csr")
            a = sparse.bmat([[a11, a12], [a21, a22]], format="csr")
            a = a + sparse.eye(2 * n, format="csr") * 1e-10
            b = np.concatenate([gamma * et_v * ex_v, gamma * et_v * ey_v])

            try:
                solved = sparse_linalg.spsolve(a, b)
            except Exception:
                solved = sparse_linalg.lsqr(a, b, atol=1e-8, btol=1e-8, iter_lim=1000)[0]

            u_exact = solved[:n].reshape(h, w)
            v_exact = solved[n:].reshape(h, w)
            u_new = (1.0 - relax) * u + relax * u_exact
            v_new = (1.0 - relax) * v + relax * v_exact
            u = np.where(valid, u_new, 0.0)
            v = np.where(valid, v_new, 0.0)
            if relax > relax_min:
                relax = max(relax_min, relax - relax_step)

        return np.where(valid, u, np.nan), np.where(valid, v, np.nan)

    def compute_velocity_field(self, phase_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute MATLAB-style nonlinear Charbonnier velocity for every frame."""
        if phase_data.ndim != 3:
            raise ValueError(f"phase_data must be (H, W, T), got shape={phase_data.shape}")

        phase = np.asarray(phase_data, dtype=np.float64)
        h, w, frames = phase.shape
        u = np.full((h, w, frames), np.nan, dtype=np.float64)
        v = np.full((h, w, frames), np.nan, dtype=np.float64)
        for t in range(frames):
            u[:, :, t], v[:, :, t] = self.compute_velocity_frame(phase, frame_idx=t)
        return u, v

    def compute_velocity_frame(self, phase_cube: np.ndarray, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute optical flow for one target frame using the full provided movie."""
        if phase_cube.ndim != 3:
            raise ValueError(f"phase_cube must be (H, W, T), got shape={phase_cube.shape}")
        t = int(frame_idx)
        h, w, frames = phase_cube.shape
        if t < 0 or t >= frames:
            raise IndexError(f"frame_idx out of range: {t}, valid [0, {frames - 1}]")

        phase = np.asarray(phase_cube, dtype=np.float64)
        f0 = phase[:, :, t]
        if t + 1 < frames:
            f1 = phase[:, :, t + 1]
        elif t > 0:
            f1 = phase[:, :, t]
            f0 = phase[:, :, t - 1]
        else:
            return np.full((h, w), np.nan), np.full((h, w), np.nan)

        ex0 = self._phase_gradient_axis(f0, axis=1)
        ey0 = self._phase_gradient_axis(f0, axis=0)
        ex1 = self._phase_gradient_axis(f1, axis=1)
        ey1 = self._phase_gradient_axis(f1, axis=0)
        ex = (ex0 + ex1) * 0.5
        ey = (ey0 + ey1) * 0.5

        if 0 < t and t + 2 < frames:
            et = (
                self._phase_diff(phase[:, :, t - 1], phase[:, :, t + 2]) / 12.0
                - (2.0 / 3.0) * self._phase_diff(f0, f1)
            )
        else:
            et = self._phase_diff(f1, f0)

        valid = (
            np.isfinite(ex)
            & np.isfinite(ey)
            & np.isfinite(et)
            & np.isfinite(f0)
            & np.isfinite(f1)
        )
        return self._matlab_charbonnier_velocity(ex, ey, et, valid)

    @staticmethod
    def _matlab_winding_path(
        loc_row: float,
        loc_col: float,
        radius: int,
        shape: tuple[int, int],
        mode: str,
    ) -> list[tuple[int, int]]:
        """Replicate windingNumberAngles centre/point square paths."""
        h, w = shape
        if mode == "centre":
            base_r = int(np.floor(loc_row))
            base_c = int(np.floor(loc_col))
            row_offsets = list(range(-radius + 1, radius + 1)) + list(range(radius, -radius, -1))
            col_offsets = (
                list(range(0, -radius, -1))
                + list(range(-radius + 1, radius + 1))
                + list(range(radius, 0, -1))
            )
        elif mode == "point":
            base_r = int(np.floor(loc_row + 0.5))
            base_c = int(np.floor(loc_col + 0.5))
            row_offsets = list(range(-radius, radius)) + list(range(radius, -radius, -1))
            col_offsets = (
                list(range(0, -radius, -1))
                + list(range(-radius, radius))
                + list(range(radius, 0, -1))
            )
        else:
            raise ValueError(f"Unsupported winding path mode: {mode}")

        path = [(base_r + dr, base_c + dc) for dr, dc in zip(row_offsets, col_offsets)]
        if any(r < 0 or c < 0 or r >= h or c >= w for r, c in path):
            return []
        return path

    def _winding_number(
        self,
        u2d: np.ndarray,
        v2d: np.ndarray,
        loc_row: float,
        loc_col: float,
        mode: str,
    ) -> float:
        path = self._matlab_winding_path(
            loc_row=loc_row,
            loc_col=loc_col,
            radius=self.winding_radius,
            shape=u2d.shape,
            mode=mode,
        )
        if len(path) < 4:
            return float("nan")

        rows = np.array([p[0] for p in path], dtype=int)
        cols = np.array([p[1] for p in path], dtype=int)
        u_ring = u2d[rows, cols]
        v_ring = v2d[rows, cols]
        if not (np.all(np.isfinite(u_ring)) and np.all(np.isfinite(v_ring))):
            return float("nan")

        theta = np.arctan2(v_ring, u_ring)
        delta = np.diff(np.r_[theta, theta[0]])
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return float(np.round(np.sum(delta) / (2.0 * np.pi)))

    def _winding_both(self, u2d: np.ndarray, v2d: np.ndarray, row: float, col: float) -> tuple[float, float]:
        return (
            self._winding_number(u2d, v2d, row, col, mode="centre"),
            self._winding_number(u2d, v2d, row, col, mode="point"),
        )

    @staticmethod
    def _bilinear_coeff(corners: np.ndarray) -> tuple[float, float, float, float]:
        f00 = float(corners[0, 0])
        f10 = float(corners[1, 0])
        f01 = float(corners[0, 1])
        f11 = float(corners[1, 1])
        return f00, f01 - f00, f10 - f00, f11 - f10 - f01 + f00

    @classmethod
    def _bilinear_value(cls, coeff: tuple[float, float, float, float], s: float, t: float) -> float:
        a, b, c, d = coeff
        return a + b * t + c * s + d * s * t

    @classmethod
    def _bilinear_zero_roots(cls, uc: np.ndarray, vc: np.ndarray) -> list[tuple[float, float]]:
        """Find u=v=0 roots inside one 2x2 bilinear cell."""
        ucoef = cls._bilinear_coeff(uc)
        vcoef = cls._bilinear_coeff(vc)
        starts = [
            (0.5, 0.5),
            (0.25, 0.25),
            (0.25, 0.75),
            (0.75, 0.25),
            (0.75, 0.75),
            (0.5, 0.25),
            (0.5, 0.75),
            (0.25, 0.5),
            (0.75, 0.5),
        ]
        roots: list[tuple[float, float]] = []

        for s0, t0 in starts:
            s = float(s0)
            t = float(t0)
            for _ in range(20):
                fu = cls._bilinear_value(ucoef, s, t)
                fv = cls._bilinear_value(vcoef, s, t)
                j = np.array(
                    [
                        [ucoef[2] + ucoef[3] * t, ucoef[1] + ucoef[3] * s],
                        [vcoef[2] + vcoef[3] * t, vcoef[1] + vcoef[3] * s],
                    ],
                    dtype=np.float64,
                )
                try:
                    step = np.linalg.solve(j, np.array([fu, fv], dtype=np.float64))
                except np.linalg.LinAlgError:
                    break
                s -= float(step[0])
                t -= float(step[1])
                if abs(step[0]) + abs(step[1]) < 1e-10:
                    break

            if not (-1e-8 <= s <= 1.0 + 1e-8 and -1e-8 <= t <= 1.0 + 1e-8):
                continue
            s = float(np.clip(s, 0.0, 1.0))
            t = float(np.clip(t, 0.0, 1.0))
            if abs(cls._bilinear_value(ucoef, s, t)) > 1e-7:
                continue
            if abs(cls._bilinear_value(vcoef, s, t)) > 1e-7:
                continue
            if not any((s - rs) ** 2 + (t - rt) ** 2 < 1e-10 for rs, rt in roots):
                roots.append((s, t))

        return roots

    @staticmethod
    def _bilinear_jacobian(
        uc: np.ndarray,
        vc: np.ndarray,
        s: float,
        t: float,
    ) -> tuple[float, float, float, float]:
        ucoef = SpiralWaveDetector._bilinear_coeff(uc)
        vcoef = SpiralWaveDetector._bilinear_coeff(vc)
        du_drow = ucoef[2] + ucoef[3] * t
        du_dcol = ucoef[1] + ucoef[3] * s
        dv_drow = vcoef[2] + vcoef[3] * t
        dv_dcol = vcoef[1] + vcoef[3] * s
        return du_dcol, du_drow, dv_dcol, dv_drow

    def find_spirals(self, u2d: np.ndarray, v2d: np.ndarray) -> List[SpiralFocus]:
        """Find stable foci, require MATLAB-style centre and point winding checks."""
        if u2d.ndim != 2 or v2d.ndim != 2:
            raise ValueError("u2d and v2d must be 2D arrays")
        if u2d.shape != v2d.shape:
            raise ValueError(f"shape mismatch: {u2d.shape} vs {v2d.shape}")

        h, w = u2d.shape
        foci: List[SpiralFocus] = []

        for r in range(h - 1):
            for c in range(w - 1):
                uc = np.array(
                    [[u2d[r, c], u2d[r, c + 1]], [u2d[r + 1, c], u2d[r + 1, c + 1]]],
                    dtype=np.float64,
                )
                vc = np.array(
                    [[v2d[r, c], v2d[r, c + 1]], [v2d[r + 1, c], v2d[r + 1, c + 1]]],
                    dtype=np.float64,
                )
                if not (np.all(np.isfinite(uc)) and np.all(np.isfinite(vc))):
                    continue
                if not (uc.min() <= 0.0 <= uc.max() and vc.min() <= 0.0 <= vc.max()):
                    continue

                for s, t in self._bilinear_zero_roots(uc, vc):
                    j11, j12, j21, j22 = self._bilinear_jacobian(uc, vc, s, t)
                    if not np.all(np.isfinite([j11, j12, j21, j22])):
                        continue

                    det = float(j11 * j22 - j12 * j21)
                    tr = float(j11 + j22)
                    if not (det > 0.0 and (tr * tr) < (4.0 * det)):
                        continue

                    stable = tr < 0.0
                    if self.stable_only and not stable:
                        continue

                    row = r + float(s)
                    col = c + float(t)
                    winding_centre, winding_point = self._winding_both(u2d, v2d, row, col)
                    if not (
                        np.isfinite(winding_centre)
                        and np.isfinite(winding_point)
                        and abs(winding_centre) > self.winding_min
                        and abs(winding_point) > self.winding_min
                        and np.sign(winding_centre) == np.sign(winding_point)
                    ):
                        continue

                    foci.append(
                        SpiralFocus(
                            row=row,
                            col=col,
                            determinant=det,
                            trace=tr,
                            stable=stable,
                            winding_number=winding_centre,
                        )
                    )

        return self._merge_nearby_foci(foci)

    def _merge_nearby_foci(self, foci: List[SpiralFocus]) -> List[SpiralFocus]:
        """Greedily merge dense clusters of nearby foci into one determinant-weighted seed."""
        if self.merge_distance <= 0 or len(foci) <= 1:
            return foci

        remaining = set(range(len(foci)))
        merged: List[SpiralFocus] = []
        radius2 = self.merge_distance * self.merge_distance

        while remaining:
            seed = max(remaining, key=lambda i: abs(foci[i].determinant))
            remaining.remove(seed)
            cluster = [seed]
            frontier = [seed]

            while frontier:
                current = frontier.pop()
                cr = foci[current].row
                cc = foci[current].col
                near = [
                    i
                    for i in remaining
                    if (foci[i].row - cr) ** 2 + (foci[i].col - cc) ** 2 <= radius2
                ]
                for i in near:
                    remaining.remove(i)
                    frontier.append(i)
                    cluster.append(i)

            weights = np.array([max(abs(foci[i].determinant), 1e-12) for i in cluster])
            rows = np.array([foci[i].row for i in cluster])
            cols = np.array([foci[i].col for i in cluster])
            dets = np.array([foci[i].determinant for i in cluster])
            traces = np.array([foci[i].trace for i in cluster])
            windings = np.array([foci[i].winding_number for i in cluster])
            trace = float(np.average(traces, weights=weights))
            winding_sum = float(np.sum(windings))

            merged.append(
                SpiralFocus(
                    row=float(np.average(rows, weights=weights)),
                    col=float(np.average(cols, weights=weights)),
                    determinant=float(np.average(dets, weights=weights)),
                    trace=trace,
                    stable=trace < 0.0,
                    winding_number=float(np.sign(winding_sum) if winding_sum != 0 else windings[0]),
                )
            )

        return merged

    def detect_from_phase_cube(self, phase_cube: np.ndarray, frame_idx: int) -> List[SpiralFocus]:
        if phase_cube.ndim != 3:
            raise ValueError(f"phase_cube must be (H, W, T), got shape={phase_cube.shape}")
        t = int(frame_idx)
        if t < 0 or t >= phase_cube.shape[2]:
            raise IndexError(f"frame_idx out of range: {t}, valid [0, {phase_cube.shape[2] - 1}]")

        u2d, v2d = self.compute_velocity_frame(phase_cube, frame_idx=t)
        return self.find_spirals(u2d, v2d)

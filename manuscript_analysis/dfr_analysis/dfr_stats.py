from __future__ import annotations

import math

import numpy as np
from scipy import stats


def finite_values(values: object) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def mean_error_fields(values: object, prefix: str) -> dict[str, float | int]:
    arr = finite_values(values)
    n = int(arr.size)
    out: dict[str, float | int] = {
        f"{prefix}_n": n,
        f"{prefix}_mean": float(np.mean(arr)) if n else math.nan,
        f"{prefix}_std": float(np.std(arr, ddof=1)) if n > 1 else math.nan,
        f"{prefix}_sem": math.nan,
        f"{prefix}_ci95_low": math.nan,
        f"{prefix}_ci95_high": math.nan,
    }
    if n > 1:
        sem = float(stats.sem(arr, nan_policy="omit"))
        half_width = float(stats.t.ppf(0.975, n - 1) * sem)
        mean = float(out[f"{prefix}_mean"])
        out[f"{prefix}_sem"] = sem
        out[f"{prefix}_ci95_low"] = mean - half_width
        out[f"{prefix}_ci95_high"] = mean + half_width
    return out


def add_legacy_mean_alias(fields: dict[str, float | int], prefix: str, alias: str) -> dict[str, float | int]:
    fields[alias] = fields[f"{prefix}_mean"]
    return fields


def pearson_ci(r: float, n: int) -> tuple[float, float, float, float, float]:
    if n <= 3 or not np.isfinite(r):
        return math.nan, math.nan, math.nan, math.nan, math.nan
    clipped = float(np.clip(r, -0.999999, 0.999999))
    z = float(np.arctanh(clipped))
    z_se = 1.0 / math.sqrt(n - 3)
    low = float(np.tanh(z - 1.96 * z_se))
    high = float(np.tanh(z + 1.96 * z_se))
    r_se = float((1.0 - clipped * clipped) / math.sqrt(n - 3))
    return r_se, low, high, low * low, high * high


def slope_ci(slope: float, stderr: float, n: int) -> tuple[float, float]:
    if n <= 2 or not np.isfinite(slope) or not np.isfinite(stderr):
        return math.nan, math.nan
    half_width = float(stats.t.ppf(0.975, n - 2) * stderr)
    return float(slope - half_width), float(slope + half_width)


def proportion_error_fields(successes: int, n: int, prefix: str) -> dict[str, float | int]:
    if n <= 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_sem": math.nan,
            f"{prefix}_ci95_low": math.nan,
            f"{prefix}_ci95_high": math.nan,
        }
    p = float(successes) / float(n)
    sem = math.sqrt(p * (1.0 - p) / n) if n > 0 else math.nan
    return {
        f"{prefix}_n": int(n),
        f"{prefix}_mean": p,
        f"{prefix}_std": math.sqrt(p * (1.0 - p)) if n > 1 else math.nan,
        f"{prefix}_sem": sem,
        f"{prefix}_ci95_low": max(0.0, p - 1.96 * sem),
        f"{prefix}_ci95_high": min(1.0, p + 1.96 * sem),
    }

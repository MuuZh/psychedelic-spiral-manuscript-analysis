from __future__ import annotations

from pathlib import Path

import pandas as pd

import math

import numpy as np
from scipy import stats
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from .dfr_stats import mean_error_fields, pearson_ci, proportion_error_fields, slope_ci


def discover_spiral_delta_tables(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_file() and path.exists():
            found.append(path)
        elif path.is_dir():
            for name in ("paired_deltas_long.csv", "subject_network_metrics_wide.csv", "subject_network_metrics_long.csv"):
                found.extend(path.rglob(name))
    return sorted(set(found))


def load_spiral_deltas(paths: list[Path]) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    tables = discover_spiral_delta_tables(paths)
    if not tables:
        return pd.DataFrame(), ["No spiral delta tables were found."]
    frames = []
    for table in tables:
        try:
            df = pd.read_csv(table)
        except Exception as exc:
            warnings.append(f"Could not read {table}: {exc}")
            continue
        df["source_table"] = str(table)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), warnings
    raw = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in raw.columns}
    required = {"subid", "hemisphere"}
    if not required.issubset(cols):
        warnings.append("Spiral tables did not contain subid and hemisphere columns.")
        return pd.DataFrame(), warnings
    out = pd.DataFrame()
    out["subid"] = raw[cols["subid"]].astype(str).str.replace("^S", "", regex=True).str.zfill(2)
    out["hemisphere"] = raw[cols["hemisphere"]].astype(str).str.lower()
    if "drug" in cols:
        out["drug"] = raw[cols["drug"]].astype(str)
    elif "comparison" in cols:
        out["drug"] = raw[cols["comparison"]].astype(str).str.split("_", expand=True)[0]
    else:
        out["drug"] = ""
    if "comparison" in cols:
        out["comparison"] = raw[cols["comparison"]].astype(str)
    else:
        out["comparison"] = ""
    candidates = {
        "delta_spiral_count": ["delta_spiral_count", "spiral_count_delta", "delta_spiral_count_per_frame"],
        "delta_spiral_size": ["delta_spiral_size", "mean_spiral_size_delta", "delta_mean_spiral_size"],
        "delta_spiral_entropy": ["delta_spiral_entropy", "path_entropy_delta", "delta_path_entropy"],
    }
    for target, names in candidates.items():
        for name in names:
            if name.lower() in cols:
                out[target] = raw[cols[name.lower()]]
                break
    keep_metrics = [c for c in candidates if c in out.columns]
    if not keep_metrics:
        warnings.append("No recognizable spiral delta metric columns were found.")
        return pd.DataFrame(), warnings
    return out.drop_duplicates(subset=["drug", "comparison", "subid", "hemisphere"]), warnings


def dfr_spiral_delta_correlations(dfr_delta: pd.DataFrame, spiral_delta: pd.DataFrame, min_n: int) -> pd.DataFrame:
    if dfr_delta.empty or spiral_delta.empty:
        return pd.DataFrame()
    rows = []
    dfr_metrics = ["delta_region_count", "delta_boundary_count"]
    spiral_metrics = [c for c in spiral_delta.columns if c.startswith("delta_spiral")]
    merged = dfr_delta.merge(spiral_delta, on=["drug", "comparison", "subid", "hemisphere"], how="inner")
    for keys, group in merged.groupby(["drug", "comparison", "hemisphere"], dropna=False):
        for dfr_metric in dfr_metrics:
            for spiral_metric in spiral_metrics:
                x = group[spiral_metric].to_numpy(float)
                y = group[dfr_metric].to_numpy(float)
                mask = np.isfinite(x) & np.isfinite(y)
                x = x[mask]
                y = y[mask]
                if len(x) >= max(3, min_n) and np.std(x) > 0 and np.std(y) > 0:
                    pr, pp = stats.pearsonr(x, y)
                    sr, sp = stats.spearmanr(x, y)
                    slope, _, r_value, _, stderr = stats.linregress(x, y)
                    pearson_se, pearson_low, pearson_high, r2_low, r2_high = pearson_ci(pr, len(x))
                    spearman_se, spearman_low, spearman_high, _, _ = pearson_ci(sr, len(x))
                    slope_low, slope_high = slope_ci(slope, stderr, len(x))
                else:
                    pr = pp = sr = sp = slope = r_value = stderr = math.nan
                    pearson_se = pearson_low = pearson_high = r2_low = r2_high = math.nan
                    spearman_se = spearman_low = spearman_high = math.nan
                    slope_low = slope_high = math.nan
                rows.append(
                    {
                        "drug": keys[0],
                        "comparison": keys[1],
                        "hemisphere": keys[2],
                        "dfr_metric": dfr_metric,
                        "spiral_metric": spiral_metric,
                        "n": int(len(x)),
                        "pearson_r": float(pr) if np.isfinite(pr) else math.nan,
                        "pearson_r_se": float(pearson_se) if np.isfinite(pearson_se) else math.nan,
                        "pearson_r_ci95_low": float(pearson_low) if np.isfinite(pearson_low) else math.nan,
                        "pearson_r_ci95_high": float(pearson_high) if np.isfinite(pearson_high) else math.nan,
                        "pearson_p": float(pp) if np.isfinite(pp) else math.nan,
                        "spearman_rho": float(sr) if np.isfinite(sr) else math.nan,
                        "spearman_rho_se": float(spearman_se) if np.isfinite(spearman_se) else math.nan,
                        "spearman_rho_ci95_low": float(spearman_low) if np.isfinite(spearman_low) else math.nan,
                        "spearman_rho_ci95_high": float(spearman_high) if np.isfinite(spearman_high) else math.nan,
                        "spearman_p": float(sp) if np.isfinite(sp) else math.nan,
                        "slope": float(slope) if np.isfinite(slope) else math.nan,
                        "slope_se": float(stderr) if np.isfinite(stderr) else math.nan,
                        "slope_ci95_low": float(slope_low) if np.isfinite(slope_low) else math.nan,
                        "slope_ci95_high": float(slope_high) if np.isfinite(slope_high) else math.nan,
                        "r_squared": float(r_value * r_value) if np.isfinite(r_value) else math.nan,
                        "r_squared_ci95_low": float(r2_low) if np.isfinite(r2_low) else math.nan,
                        "r_squared_ci95_high": float(r2_high) if np.isfinite(r2_high) else math.nan,
                    }
                )
    return pd.DataFrame(rows)


def spiral_overlap_delta_table(spiral_overlap: pd.DataFrame) -> pd.DataFrame:
    if spiral_overlap.empty:
        return pd.DataFrame()
    metric_map = {
        "spiral_centers_per_frame": "delta_spiral_count",
        "mean_spiral_size": "delta_spiral_size",
        "mean_spiral_boundary_distance": "delta_spiral_boundary_distance",
        "median_spiral_boundary_distance": "delta_spiral_median_boundary_distance",
        "fraction_centers_within_0px": "delta_spiral_boundary_overlap_0px",
        "fraction_centers_within_1px": "delta_spiral_boundary_overlap_1px",
        "fraction_centers_within_2px": "delta_spiral_boundary_overlap_2px",
        "fraction_centers_within_3px": "delta_spiral_boundary_overlap_3px",
    }
    available = [metric for metric in metric_map if metric in spiral_overlap.columns]
    if not available:
        return pd.DataFrame()

    rows = []
    group_cols = ["drug", "comparison", "hemisphere"]
    for keys, group in spiral_overlap.groupby(group_cols, dropna=False):
        pivot = group.pivot_table(index="subid", columns="role", values=available, aggfunc="mean")
        needed_roles = {"Drug", "PCB"}
        if not needed_roles.issubset(set(pivot.columns.get_level_values(1))):
            continue
        for subid, vals in pivot.iterrows():
            row = {
                "drug": keys[0],
                "comparison": keys[1],
                "subid": str(subid),
                "hemisphere": keys[2],
            }
            has_metric = False
            for metric in available:
                drug_key = (metric, "Drug")
                pcb_key = (metric, "PCB")
                if drug_key not in pivot.columns or pcb_key not in pivot.columns:
                    continue
                drug_value = vals[drug_key]
                pcb_value = vals[pcb_key]
                if pd.notna(drug_value) and pd.notna(pcb_value):
                    row[metric_map[metric]] = float(drug_value - pcb_value)
                    has_metric = True
            if has_metric:
                rows.append(row)
    return pd.DataFrame(rows)


def resolve_spiral_bundle(row: pd.Series, spiral_bundle_roots: list[Path]) -> Path | None:
    phase_parent = Path(row["phase_cube"]).parent
    if (phase_parent / "frame_index.parquet").exists():
        return phase_parent
    bundle_name = Path(str(row["bundle_dir"])).name
    drug = str(row["drug"])
    for root in spiral_bundle_roots:
        candidates = [
            root / drug / bundle_name,
            root / bundle_name,
        ]
        for candidate in candidates:
            if (candidate / "frame_index.parquet").exists():
                return candidate
    hemi_suffix = "L" if str(row["hemisphere"]).lower() == "left" else "R"
    token = f"{row['condition']}_S{int(row['subid']):02d}"
    for root in spiral_bundle_roots:
        search_root = root / drug if (root / drug).exists() else root
        matches = sorted(
            p
            for p in search_root.rglob("frame_index.parquet")
            if token in p.parent.name and p.parent.name.endswith(hemi_suffix)
        )
        if matches:
            return matches[0].parent
    return None


def load_spiral_centers(bundle_dir: Path) -> pd.DataFrame:
    frame_path = bundle_dir / "frame_index.parquet"
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing frame_index.parquet: {bundle_dir}")
    frame_index = pd.read_parquet(frame_path)
    if {"weighted_centroid_x", "weighted_centroid_y"}.issubset(frame_index.columns):
        x_col = "weighted_centroid_x"
        y_col = "weighted_centroid_y"
    elif {"centroid_x", "centroid_y"}.issubset(frame_index.columns):
        x_col = "centroid_x"
        y_col = "centroid_y"
    else:
        raise ValueError(f"frame_index.parquet lacks centroid columns: {bundle_dir}")
    needed = ["abs_time", x_col, y_col]
    extra = [c for c in ["instantaneous_size", "instantaneous_power", "pattern_id"] if c in frame_index.columns]
    centers = frame_index[needed + extra].copy()
    centers = centers.rename(columns={x_col: "center_x", y_col: "center_y"})
    centers["abs_time"] = centers["abs_time"].astype(int)
    return centers


def compute_spiral_boundary_overlap(
    row: pd.Series,
    boundary_stack: np.ndarray,
    spiral_bundle_roots: list[Path],
    *,
    overlap_radii: tuple[int, ...] = (0, 1, 2, 3),
    show_progress: bool = False,
) -> tuple[dict[str, object] | None, list[dict[str, str]], dict[int, pd.DataFrame]]:
    if str(row.get("source", "")) != "empirical":
        return None, [], {}
    failures: list[dict[str, str]] = []
    bundle = resolve_spiral_bundle(row, spiral_bundle_roots)
    if bundle is None:
        failures.append({"bundle_dir": str(row.get("bundle_dir", "")), "error": "spiral_bundle_missing"})
        return None, failures, {}

    try:
        centers = load_spiral_centers(bundle)
    except Exception as exc:
        failures.append({"bundle_dir": str(bundle), "error": f"spiral_centers_failed: {exc}"})
        return None, failures, {}

    if centers.empty:
        base = {
            "spiral_bundle_dir": str(bundle),
            "n_spiral_centers": 0,
            "mean_spiral_boundary_distance": math.nan,
            "median_spiral_boundary_distance": math.nan,
        }
        for radius in overlap_radii:
            base[f"fraction_centers_within_{radius}px"] = math.nan
        return base, failures, {}

    n_frames = boundary_stack.shape[2]
    distances: list[float] = []
    sizes: list[float] = []
    powers: list[float] = []
    qc_centers: dict[int, pd.DataFrame] = {}
    grouped = list(centers.groupby("abs_time", sort=True))
    iterator = grouped
    if show_progress and len(grouped) > 20:
        iterator = tqdm(grouped, desc=f"Spiral overlap {bundle.name}", unit="frame", leave=False)

    for frame, frame_rows in iterator:
        frame = int(frame)
        if frame < 0 or frame >= n_frames:
            continue
        boundary = boundary_stack[:, :, frame].astype(bool, copy=False)
        if np.any(boundary):
            dist_map = distance_transform_edt(~boundary)
        else:
            dist_map = np.full(boundary.shape, np.nan, dtype=float)
        ys = np.rint(frame_rows["center_y"].to_numpy(float)).astype(int)
        xs = np.rint(frame_rows["center_x"].to_numpy(float)).astype(int)
        valid = (ys >= 0) & (ys < boundary.shape[0]) & (xs >= 0) & (xs < boundary.shape[1])
        vals = np.full(len(frame_rows), np.nan, dtype=float)
        vals[valid] = dist_map[ys[valid], xs[valid]]
        distances.extend(vals[np.isfinite(vals)].astype(float).tolist())
        if "instantaneous_size" in frame_rows:
            sizes.extend(frame_rows["instantaneous_size"].to_numpy(float)[np.isfinite(vals)].tolist())
        if "instantaneous_power" in frame_rows:
            powers.extend(frame_rows["instantaneous_power"].to_numpy(float)[np.isfinite(vals)].tolist())
        one = frame_rows.copy()
        one["boundary_distance"] = vals
        qc_centers[frame] = one

    dist = np.asarray(distances, dtype=float)
    size_values = np.asarray(sizes, dtype=float)
    power_values = np.asarray(powers, dtype=float)
    base = {
        "spiral_bundle_dir": str(bundle),
        "n_spiral_centers": int(len(dist)),
        **mean_error_fields(dist, "spiral_boundary_distance"),
        "mean_spiral_boundary_distance": float(np.mean(dist)) if len(dist) else math.nan,
        "median_spiral_boundary_distance": float(np.median(dist)) if len(dist) else math.nan,
        **mean_error_fields(size_values, "spiral_size"),
        "mean_spiral_size": float(np.mean(sizes)) if sizes else math.nan,
        **mean_error_fields(power_values, "spiral_power"),
        "mean_spiral_power": float(np.mean(powers)) if powers else math.nan,
    }
    for radius in overlap_radii:
        base[f"fraction_centers_within_{radius}px"] = float(np.mean(dist <= radius)) if len(dist) else math.nan
        base.update(proportion_error_fields(int(np.sum(dist <= radius)), int(len(dist)), f"fraction_centers_within_{radius}px"))
    return base, failures, qc_centers

"""Network segregation metrics for phase-FC group analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


NETWORK_ORDER = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def compute_network_segregation(
    network_values: pd.DataFrame,
    parcels: pd.DataFrame,
    *,
    network_order: list[str] | None = None,
) -> pd.DataFrame:
    """Compute raw/normalized and weighted/unweighted segregation per network."""
    order = network_order or NETWORK_ORDER
    available = set(network_values["network_a"]) | set(network_values["network_b"])
    networks = [network for network in order if network in available]
    node_counts = parcels.groupby("network")["parcel_id"].count().to_dict()

    within: dict[str, float] = {}
    between: dict[tuple[str, str], float] = {}
    for row in network_values.itertuples(index=False):
        if row.type == "within":
            within[row.network_a] = float(row.value)
        else:
            between[tuple(sorted((row.network_a, row.network_b)))] = float(row.value)

    rows = []
    for network in networks:
        w_value = within.get(network, np.nan)
        n_network = float(node_counts.get(network, np.nan))
        b_values = []
        weights = []
        for other in networks:
            if other == network:
                continue
            b_values.append(between.get(tuple(sorted((network, other))), np.nan))
            weights.append(n_network * float(node_counts.get(other, np.nan)))

        b_arr = np.asarray(b_values, dtype=float)
        w_arr = np.asarray(weights, dtype=float)
        finite = np.isfinite(b_arr)
        mean_between = float(np.nanmean(b_arr)) if np.any(finite) else np.nan
        finite_w = finite & np.isfinite(w_arr) & (w_arr > 0)
        weighted_between = (
            float(np.sum(b_arr[finite_w] * w_arr[finite_w]) / np.sum(w_arr[finite_w]))
            if np.any(finite_w)
            else np.nan
        )

        for weighted, between_mean in [(False, mean_between), (True, weighted_between)]:
            raw_value = w_value - between_mean
            norm_value = raw_value / w_value if np.isfinite(w_value) and w_value != 0 else np.nan
            rows.append(
                {
                    "network": network,
                    "variant": "raw_weighted" if weighted else "raw_unweighted",
                    "weighted": bool(weighted),
                    "normalized": False,
                    "value": raw_value,
                    "within": w_value,
                    "mean_between": between_mean,
                    "n_network": int(node_counts.get(network, 0)),
                }
            )
            rows.append(
                {
                    "network": network,
                    "variant": "normalized_weighted" if weighted else "normalized_unweighted",
                    "weighted": bool(weighted),
                    "normalized": True,
                    "value": norm_value,
                    "within": w_value,
                    "mean_between": between_mean,
                    "n_network": int(node_counts.get(network, 0)),
                }
            )

    return pd.DataFrame(rows)

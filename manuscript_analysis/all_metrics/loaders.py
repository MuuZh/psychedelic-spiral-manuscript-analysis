from __future__ import annotations

import json
import re
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from .config import Config


def load_bundles(cfg: Config) -> pd.DataFrame:
    """Scan detect_results_dir for bundle folders; infer group/subid/hemisphere from metadata or suffix."""

    def parse_suffix(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        m = re.search(r"_([A-Za-z]+)(\d+)([LR])$", name)
        if not m:
            return None, None, None
        grp_raw, subid, hemi_token = m.groups()
        hemi = "left" if hemi_token.upper() == "L" else "right"
        grp_raw_low = grp_raw.lower()
        group = None
        if grp_raw_low.endswith("dmt") or grp_raw_low == cfg.group_drug.lower() or grp_raw_low == "dmt":
            group = cfg.group_drug
        elif grp_raw_low.endswith("lsd") or grp_raw_low == cfg.group_drug.lower() or grp_raw_low == "lsd":
            group = cfg.group_drug
        elif grp_raw_low.endswith("pcb") or grp_raw_low == cfg.group_pcb.lower() or grp_raw_low == "pcb":
            group = cfg.group_pcb
        return group, subid, hemi

    def infer_from_name(name: str):
        hemi = None
        m = re.search(r"([0-9]+)([LR])?$", name)
        subid = m.group(1) if m else name
        if m and m.group(2):
            hemi = "left" if m.group(2).upper() == "L" else "right"
        else:
            low = name.lower()
            if low.endswith("l") or "left" in low:
                hemi = "left"
            elif low.endswith("r") or "right" in low:
                hemi = "right"
        return hemi, subid

    rows = []
    root = Path(cfg.detect_results_dir)
    candidates = []
    if (root / "phase_cube.npy").exists() or (root / "metadata.json").exists():
        candidates.append(root)
    for d1 in root.glob("*"):
        if d1.is_dir():
            if (d1 / "phase_cube.npy").exists() or (d1 / "metadata.json").exists():
                candidates.append(d1)
            else:
                for d2 in d1.glob("*"):
                    if d2.is_dir() and ((d2 / "phase_cube.npy").exists() or (d2 / "metadata.json").exists()):
                        candidates.append(d2)

    for bundle_dir in candidates:
        meta_path = bundle_dir / "metadata.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                meta = {}

        group = meta.get("group")
        hemi = meta.get("hemisphere")
        subid = meta.get("subject_id") or meta.get(
            "subid") or meta.get("subject")

        if group is None or hemi is None or subid is None:
            grp_suf, sub_suf, hemi_suf = parse_suffix(bundle_dir.name)
            group = group or grp_suf
            hemi = hemi or hemi_suf
            subid = subid or sub_suf
        if group is None or hemi is None or subid is None:
            hemi_infer, sub_infer = infer_from_name(bundle_dir.name)
            hemi = hemi or hemi_infer
            subid = subid or sub_infer

        if group is None or hemi is None or subid is None:
            continue

        rows.append(
            {
                "group": group,
                "subid": str(subid),
                "hemisphere": hemi,
                "bundle_dir": bundle_dir,
                "phase_cube": bundle_dir / "phase_cube.npy",
                "vortex_coords": bundle_dir / "vortex_coords.json",
                "vortex_occupancy": bundle_dir / "vortex_occupancy.npy",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        logging.warning("No bundles found under %s", cfg.detect_results_dir)
    return df

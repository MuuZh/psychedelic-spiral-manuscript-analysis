#!/usr/bin/env python3
"""Run the Workbench FC batch PowerShell script for one or more groups."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_CONFIG: Dict[str, Any] = {}


@dataclass
class GroupJob:
    name: str
    dtseries_dir: Path
    out_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fc_batch_workbench.ps1 for one or more groups."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON config file. If omitted, the script uses DEFAULT_CONFIG inside the file.",
    )
    parser.add_argument(
        "--write-example-config",
        type=Path,
        help="Write an example JSON config to this path and exit.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("run_fc_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None:
        return json.loads(json.dumps(DEFAULT_CONFIG))

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_groups(raw_groups: Iterable[Dict[str, Any]]) -> List[GroupJob]:
    jobs: List[GroupJob] = []
    for item in raw_groups:
        jobs.append(
            GroupJob(
                name=str(item["name"]),
                dtseries_dir=Path(item["dtseries_dir"]),
                out_dir=Path(item["out_dir"]),
            )
        )
    if not jobs:
        raise ValueError("Config must contain at least one group job.")
    return jobs


def stream_subprocess(command: List[str], logger: logging.Logger) -> int:
    logger.info("Running command: %s", shlex.join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("[PS] %s", line.rstrip())
    process.wait()
    return process.returncode


def build_command(config: Dict[str, Any], group: GroupJob) -> List[str]:
    powershell_executable = str(config.get(
        "powershell_executable", "powershell"))
    ps_script = str(config["powershell_script"])

    command = [
        powershell_executable,
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        ps_script,
        "-WbDir",
        str(config["wb_dir"]),
        "-DtseriesDir",
        str(group.dtseries_dir),
        "-AtlasDlabel",
        str(config["atlas_dlabel"]),
        "-OutDir",
        str(group.out_dir),
        "-Pattern",
        str(config.get("pattern", "*.dtseries.nii")),
        "-ParcelMethod",
        str(config.get("parcel_method", "MEAN")),
        "-MemLimitGb",
        str(config.get("mem_limit_gb", 8)),
    ]

    if config.get("skip_if_exists", False):
        command.append("-SkipIfExists")
    if config.get("overwrite", False):
        command.append("-Overwrite")

    return command


def main() -> int:
    args = parse_args()

    if args.write_example_config:
        args.write_example_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_example_config.write_text(
            json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8"
        )
        print(f"Example config written to {args.write_example_config}")
        return 0

    config = load_config(args.config)
    groups = validate_groups(config["groups"])

    log_root = Path(config.get("log_dir", Path.cwd() / "fc_pipeline_logs"))
    log_path = log_root / "run_fc_pipeline.log"
    logger = setup_logger(log_path)

    logger.info("Loaded %d group job(s).", len(groups))
    logger.info("Workbench dir: %s", config["wb_dir"])
    logger.info("Atlas dlabel: %s", config["atlas_dlabel"])
    logger.info("PowerShell script: %s", config["powershell_script"])

    failures = 0
    for idx, group in enumerate(groups, start=1):
        logger.info("[%d/%d] Starting group '%s'",
                    idx, len(groups), group.name)
        logger.info("    dtseries_dir=%s", group.dtseries_dir)
        logger.info("    out_dir=%s", group.out_dir)
        command = build_command(config, group)
        return_code = stream_subprocess(command, logger)
        if return_code != 0:
            failures += 1
            logger.error("Group '%s' failed with exit code %d",
                         group.name, return_code)
        else:
            logger.info("Group '%s' finished successfully.", group.name)

    if failures:
        logger.error(
            "Pipeline finished with %d failed group job(s).", failures)
        return 1

    logger.info("Pipeline finished successfully for all groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

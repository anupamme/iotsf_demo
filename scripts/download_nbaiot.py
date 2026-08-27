#!/usr/bin/env python3
"""
Download N-BaIoT dataset into data/nbaiot/.

Credential resolution order
---------------------------
1. KAGGLE_USERNAME + KAGGLE_KEY environment variables (set directly)
2. ~/.kaggle/kaggle.json (standard Kaggle config file)
3. Prints manual download instructions and exits with code 1.

Usage
-----
    # With env vars:
    KAGGLE_USERNAME=myuser KAGGLE_KEY=myapikey python scripts/download_nbaiot.py

    # With existing kaggle.json:
    python scripts/download_nbaiot.py

Manual download (if Kaggle API unavailable)
-------------------------------------------
1. Go to https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset
2. Click Download (requires free Kaggle account)
3. Unzip into data/nbaiot/ so that:
       data/nbaiot/Danmini_Doorbell/benign_traffic.csv
       data/nbaiot/Danmini_Doorbell/mirai_attacks/scan.csv
       ...
4. Then run: python scripts/evaluate_nbaiot.py --data-dir data/nbaiot/
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "nbaiot"
KAGGLE_DATASET = "mkashifn/nbaiot-dataset"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"


def _resolve_credentials() -> bool:
    """
    Ensure Kaggle credentials are available for the CLI.

    If KAGGLE_USERNAME + KAGGLE_KEY are set in the environment, write them
    to ~/.kaggle/kaggle.json so the kaggle CLI can find them, then return True.
    If kaggle.json already exists, return True immediately.
    Otherwise return False.
    """
    username = os.environ.get("KAGGLE_USERNAME", "")
    key = os.environ.get("KAGGLE_KEY", "")

    if KAGGLE_JSON.exists():
        print(f"Using credentials from {KAGGLE_JSON}")
        return True

    if username and key:
        print(f"Using KAGGLE_USERNAME / KAGGLE_KEY environment variables.")
        KAGGLE_JSON.parent.mkdir(parents=True, exist_ok=True)
        creds = json.dumps({"username": username, "key": key})
        KAGGLE_JSON.write_text(creds)
        KAGGLE_JSON.chmod(0o600)
        print(f"Wrote credentials to {KAGGLE_JSON}")
        return True

    # Neither source available
    print(
        "No Kaggle credentials found.\n"
        "Set environment variables before running:\n"
        "    export KAGGLE_USERNAME=your_username\n"
        "    export KAGGLE_KEY=your_api_key\n"
        "Or place your kaggle.json at ~/.kaggle/kaggle.json\n"
        "(Download it from https://www.kaggle.com/settings → API → Create New Token)"
    )
    return False


def _find_kaggle_bin() -> str:
    """Return path to kaggle CLI, preferring the venv installation."""
    venv_kaggle = ROOT_DIR / ".venv" / "bin" / "kaggle"
    if venv_kaggle.exists():
        return str(venv_kaggle)
    found = shutil.which("kaggle")
    if found:
        return found
    return ""


def try_kaggle_download(output_dir: Path) -> bool:
    """Attempt download via kaggle CLI. Returns True on success."""
    kaggle_bin = _find_kaggle_bin()
    if not kaggle_bin:
        print("kaggle CLI not found. Install with:  pip install kaggle")
        return False

    if not _resolve_credentials():
        return False

    print(f"Downloading N-BaIoT via Kaggle API into {output_dir} ...")
    cmd = [
        kaggle_bin, "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(output_dir),
        "--unzip",
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Download complete → {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Kaggle download failed (exit code {e.returncode}).")
        return False


def verify_structure(data_dir: Path) -> bool:
    """Check that at least the Danmini Doorbell benign file exists (device ID 1)."""
    benign = data_dir / "1.benign.csv"
    if benign.exists():
        print(f"Verified: {benign}")
        return True
    # Also check old subdirectory layout just in case
    alt = data_dir / "Danmini_Doorbell" / "benign_traffic.csv"
    if alt.exists():
        print(f"Verified (subdirectory layout): {alt}")
        return True
    print(
        f"Expected file not found: {benign}\n"
        f"  Check that the dataset was extracted correctly into {data_dir}"
    )
    return False


def print_manual_instructions():
    sep = "=" * 60
    lines = [
        "",
        sep,
        "Manual Download Instructions",
        sep,
        "The N-BaIoT dataset (~300 MB per device, ~10 GB full) is",
        "available from:",
        "",
        "  Kaggle (recommended, free account required):",
        "    https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset",
        "",
        "  UCI ML Repository:",
        "    https://archive.ics.uci.edu/dataset/442",
        "",
        "After downloading, extract so that:",
        f"  {DATA_DIR}/Danmini_Doorbell/benign_traffic.csv",
        f"  {DATA_DIR}/Danmini_Doorbell/mirai_attacks/scan.csv",
        "  ... etc.",
        "",
        "Then run:",
        "  python scripts/evaluate_nbaiot.py --data-dir data/nbaiot/",
        sep,
    ]
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Download N-BaIoT dataset")
    parser.add_argument(
        "--output-dir", default=str(DATA_DIR),
        help=f"Destination directory (default: {DATA_DIR})"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target directory: {output_dir.resolve()}")

    success = try_kaggle_download(output_dir)

    if success:
        verify_structure(Path(args.output_dir))
    else:
        print_manual_instructions()
        sys.exit(1)


if __name__ == "__main__":
    main()

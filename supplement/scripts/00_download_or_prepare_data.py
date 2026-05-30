#!/usr/bin/env python3
"""Download or prepare datasets for reproducibility experiments.

Downloads:
  - ETTh2.csv (Electricity Transformer Temperature, hourly)
  - national_illness.csv (ILI weekly data)
  - M4-Monthly train/test (for Chronos cross-backbone experiment)

CICIoT2023 must be obtained separately (see note below).
"""

import os
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

DATASETS = {
    "ETTh2.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv",
    "national_illness.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/illness/national_illness.csv",
    "m4/Monthly-train.csv": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Monthly-train.csv",
    "m4/Monthly-test.csv": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/Monthly-test.csv",
}


def download_file(url: str, dest: Path):
    if dest.exists():
        print(f"  Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved to {dest}")


def main():
    print("Preparing datasets...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DATASETS.items():
        dest = DATA_DIR / filename
        download_file(url, dest)

    print("\nAll forecasting datasets ready.")
    print(
        "\nNOTE: CICIoT2023 (IoT negative control) must be obtained separately from:"
    )
    print("  https://www.unb.ca/cic/datasets/iotdataset-2023.html")
    print("  Place processed features in data/ciciot2023/")


if __name__ == "__main__":
    main()

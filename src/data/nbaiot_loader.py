"""
N-BaIoT Dataset Loader

Maps N-BaIoT's 115 statistical features to the 12-feature proxy space
used by HNIDS (matching the CICIoT2023 feature semantics), then creates
128-timestep windows suitable for Moirai inference.

Feature mapping rationale
--------------------------
N-BaIoT uses temporal-decay statistics over 5 windows (L0.01=100ms,
L0.1=500ms, L1=1.5s, L3=10s, L5=60s) for multiple aggregation groups
(MI_dir, HH, HH_jit, HpHp, H).  We extract 12 proxies from the 60s
window (L5) to match CICIoT2023 flow-level aggregates:

  CICIoT2023 feature   N-BaIoT proxy               Source column / derivation
  flow_duration        60.0 (constant)              Longest temporal window span
  fwd_pkts_tot         MI_dir_L5_weight             Src→dst IP pkt count, 60s
  bwd_pkts_tot         H_L5_weight                  Src host total pkt count, 60s
  fwd_data_pkts_tot    HpHp_L5_weight               Port-pair pkt count ≈ data pkts
  bwd_data_pkts_tot    HH_L5_weight                 Host-pair pkt count ≈ bidir data
  fwd_pkts_per_sec     MI_dir_L5_weight / 60        Derived rate
  bwd_pkts_per_sec     H_L5_weight / 60             Derived rate
  flow_pkts_per_sec    (MI_dir+H)_L5_weight / 60   Total rate
  fwd_byts_b_avg       MI_dir_L5_mean               Mean fwd pkt size
  bwd_byts_b_avg       H_L5_mean                    Mean src-host pkt size
  fwd_iat_mean         HH_jit_L5_mean               Inter-arrival jitter, 60s window
  bwd_iat_mean         HH_jit_L0.01_mean            Short-window jitter as bwd IAT proxy

The mapping is imperfect (N-BaIoT is packet-triggered, CICIoT2023 is
flow-record based) but preserves the conceptual ordering of the features
that the trained HNIDS detector relies on.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.preprocessor import create_sequences


# ---------------------------------------------------------------------------
# Device → numeric ID map
# (Kaggle variant uses flat files: {N}.benign.csv, {N}.mirai.scan.csv, etc.)
# Source: device_info.csv included in the Kaggle download.
# ---------------------------------------------------------------------------

DEVICE_IDS = {
    "danmini_doorbell":            1,
    "ecobee_thermostat":           2,
    "ennio_doorbell":              3,
    "philips_b120n10":             4,
    "provision_pt737e":            5,
    "provision_pt838":             6,
    "samsung_snhv6410pn":          7,
    "simple_home_xcs7_1002":       8,
    "simple_home_xcs7_1003":       9,
}

# Attack-type key → filename suffix (without leading "{N}.")
# Presence of each file varies by device; missing files are skipped silently.
MIRAI_SUFFIXES = {
    "mirai_scan":      "mirai.scan.csv",
    "mirai_ack":       "mirai.ack.csv",
    "mirai_syn":       "mirai.syn.csv",
    "mirai_udp":       "mirai.udp.csv",
    "mirai_udpplain":  "mirai.udpplain.csv",
}

GAFGYT_SUFFIXES = {
    "gafgyt_scan":  "gafgyt.scan.csv",
    "gafgyt_junk":  "gafgyt.junk.csv",
    "gafgyt_tcp":   "gafgyt.tcp.csv",
    "gafgyt_udp":   "gafgyt.udp.csv",
    "gafgyt_combo": "gafgyt.combo.csv",
}

ALL_ATTACK_SUFFIXES = {**MIRAI_SUFFIXES, **GAFGYT_SUFFIXES}

BENIGN_SUFFIX = "benign.csv"

# ---------------------------------------------------------------------------
# Proxy column specification
# ---------------------------------------------------------------------------

# Each entry: (type, value_or_colname)
#   'constant' → fill column with float constant
#   'col'      → direct column rename
#   'derived'  → callable(df) → pd.Series
_PROXY_SPEC: List[Tuple[str, str, object]] = [
    ("flow_duration",     "constant", 60.0),
    ("fwd_pkts_tot",      "col",      "MI_dir_L5_weight"),
    ("bwd_pkts_tot",      "col",      "H_L5_weight"),
    ("fwd_data_pkts_tot", "col",      "HpHp_L5_weight"),
    ("bwd_data_pkts_tot", "col",      "HH_L5_weight"),
    ("fwd_pkts_per_sec",  "derived",  lambda df: df["MI_dir_L5_weight"] / 60.0),
    ("bwd_pkts_per_sec",  "derived",  lambda df: df["H_L5_weight"] / 60.0),
    ("flow_pkts_per_sec", "derived",  lambda df: (df["MI_dir_L5_weight"] + df["H_L5_weight"]) / 60.0),
    ("fwd_byts_b_avg",    "col",      "MI_dir_L5_mean"),
    ("bwd_byts_b_avg",    "col",      "H_L5_mean"),
    ("fwd_iat_mean",      "col",      "HH_jit_L5_mean"),
    ("bwd_iat_mean",      "col",      "HH_jit_L0.01_mean"),
]

PROXY_FEATURE_NAMES = [name for name, *_ in _PROXY_SPEC]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_proxy_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the 12 proxy features from a raw N-BaIoT DataFrame.

    Raises KeyError if a required source column is missing.
    """
    proxy = pd.DataFrame(index=df.index)
    for feat_name, kind, spec in _PROXY_SPEC:
        if kind == "constant":
            proxy[feat_name] = float(spec)
        elif kind == "col":
            if spec not in df.columns:
                raise KeyError(
                    f"N-BaIoT column '{spec}' not found. "
                    f"Available columns (first 20): {list(df.columns[:20])}"
                )
            proxy[feat_name] = df[spec].values
        elif kind == "derived":
            proxy[feat_name] = spec(df).values
        else:
            raise ValueError(f"Unknown proxy spec kind: {kind}")
    return proxy


def _read_csv(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read a N-BaIoT CSV, optionally limiting rows."""
    if not path.exists():
        raise FileNotFoundError(f"N-BaIoT file not found: {path}")
    df = pd.read_csv(path, nrows=max_rows)
    logger.debug(f"Read {len(df)} rows from {path.name}")
    return df


def _to_windows(
    raw_2d: np.ndarray,
    seq_length: int = 128,
    stride: int = 128,
) -> np.ndarray:
    """
    Create non-overlapping windows from a (n_rows, 12) array.

    Drops any trailing rows that don't fill a complete window.
    Returns shape (n_windows, seq_length, 12).
    """
    if len(raw_2d) < seq_length:
        return np.empty((0, seq_length, raw_2d.shape[1]), dtype=raw_2d.dtype)
    return create_sequences(raw_2d, seq_length=seq_length, stride=stride)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nbaiot(
    data_dir: str,
    device: str = "danmini_doorbell",
    max_samples_per_class: int = 5000,
    seq_length: int = 128,
    seed: int = 42,
    val_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess N-BaIoT data for a single IoT device.

    The StandardScaler is fit on benign samples only (simulating deployment
    where the scaler is trained on known-good traffic).  Both benign and
    attack sequences are transformed with this scaler and clipped to [-5, 5].

    Parameters
    ----------
    data_dir : str
        Root directory containing device subdirectories from N-BaIoT.
        Expected layout::

            data_dir/
            └── Danmini_Doorbell/
                ├── benign_traffic.csv
                ├── mirai_attacks/
                │   ├── scan.csv
                │   └── ...
                └── gafgyt_attacks/
                    ├── scan.csv
                    └── ...

    device : str
        One of the keys in DEVICE_DIRS (default: 'danmini_doorbell').
    max_samples_per_class : int
        Maximum raw rows to read per class before windowing (prevents OOM).
    seq_length : int
        Timesteps per window (default: 128, matching HNIDS).
    seed : int
        Random seed for class-level subsampling.
    val_fraction : float
        Fraction of benign sequences held out for threshold calibration.

    Returns
    -------
    X_benign_train : np.ndarray  shape (n_b_train, 128, 12)
    X_benign_val   : np.ndarray  shape (n_b_val,   128, 12)
    X_attack       : np.ndarray  shape (n_a,        128, 12)
    y_attack_names : np.ndarray  shape (n_a,)  dtype str — attack-type label per window
    scaler         : StandardScaler fitted on benign rows
    attack_types   : List[str]  — ordered list of attack-type keys present in data_dir
    """
    rng = np.random.default_rng(seed)
    data_path = Path(data_dir)

    if device not in DEVICE_IDS:
        raise ValueError(
            f"Unknown device '{device}'. Choose from: {list(DEVICE_IDS.keys())}"
        )
    dev_id = DEVICE_IDS[device]

    # ------------------------------------------------------------------
    # 1. Load benign traffic
    # ------------------------------------------------------------------
    benign_path = data_path / f"{dev_id}.{BENIGN_SUFFIX}"
    benign_raw = _read_csv(benign_path, max_rows=max_samples_per_class)
    benign_proxy = _build_proxy_df(benign_raw).dropna()
    benign_arr = benign_proxy.values.astype(np.float64)
    logger.info(f"Benign rows after NaN drop: {len(benign_arr)}")

    # ------------------------------------------------------------------
    # 2. Fit scaler on benign rows only
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(benign_arr)
    benign_norm = np.clip(scaler.transform(benign_arr), -5.0, 5.0)

    # ------------------------------------------------------------------
    # 3. Window benign sequences and split into train/val
    # ------------------------------------------------------------------
    benign_seq = _to_windows(benign_norm, seq_length=seq_length, stride=seq_length)
    if len(benign_seq) == 0:
        raise RuntimeError(
            f"No complete {seq_length}-timestep windows from benign data "
            f"({len(benign_norm)} rows). "
            "Increase max_samples_per_class or reduce seq_length."
        )

    n_val = max(1, int(len(benign_seq) * val_fraction))
    idx = rng.permutation(len(benign_seq))
    X_benign_val   = benign_seq[idx[:n_val]]
    X_benign_train = benign_seq[idx[n_val:]]
    logger.info(
        f"Benign sequences: {len(benign_seq)} total "
        f"({len(X_benign_train)} train, {len(X_benign_val)} val)"
    )

    # ------------------------------------------------------------------
    # 4. Load attack traffic
    # ------------------------------------------------------------------
    attack_windows_list: List[np.ndarray] = []
    attack_label_list:   List[np.ndarray] = []
    attack_types_found:  List[str]        = []

    for atk_key, suffix in ALL_ATTACK_SUFFIXES.items():
        atk_path = data_path / f"{dev_id}.{suffix}"
        if not atk_path.exists():
            logger.warning(f"Attack file not found (skipping): {atk_path}")
            continue

        try:
            atk_raw  = _read_csv(atk_path, max_rows=max_samples_per_class)
            atk_prx  = _build_proxy_df(atk_raw).dropna()
            atk_arr  = atk_prx.values.astype(np.float64)
            atk_norm = np.clip(scaler.transform(atk_arr), -5.0, 5.0)
            atk_seq  = _to_windows(atk_norm, seq_length=seq_length, stride=seq_length)

            if len(atk_seq) == 0:
                logger.warning(f"No complete windows from {atk_key} ({len(atk_arr)} rows); skipping")
                continue

            attack_windows_list.append(atk_seq)
            attack_label_list.append(np.full(len(atk_seq), atk_key, dtype=object))
            attack_types_found.append(atk_key)
            logger.info(f"  {atk_key}: {len(atk_seq)} windows")

        except Exception as exc:
            logger.warning(f"Error loading {atk_key}: {exc}")

    if not attack_windows_list:
        raise RuntimeError(
            f"No attack data found under {device_dir}. "
            "Check that N-BaIoT was downloaded correctly."
        )

    X_attack      = np.concatenate(attack_windows_list, axis=0)
    y_attack_names = np.concatenate(attack_label_list, axis=0)

    logger.success(
        f"N-BaIoT loaded: {len(X_benign_train)} benign-train, "
        f"{len(X_benign_val)} benign-val, "
        f"{len(X_attack)} attack windows "
        f"({len(attack_types_found)} attack types)"
    )

    return (
        X_benign_train,
        X_benign_val,
        X_attack,
        y_attack_names,
        scaler,
        attack_types_found,
    )


def get_device_benign_path(data_dir: str, device: str) -> Path:
    """Return the path to the benign CSV for a given device."""
    if device not in DEVICE_IDS:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(DEVICE_IDS.keys())}")
    dev_id = DEVICE_IDS[device]
    return Path(data_dir) / f"{dev_id}.{BENIGN_SUFFIX}"


def verify_nbaiot_columns(csv_path: str) -> Dict[str, bool]:
    """
    Check which proxy source columns are present in a N-BaIoT CSV.

    Useful for debugging missing-column errors before a full load.

    Returns dict mapping required_column → present_in_file.
    """
    df = pd.read_csv(csv_path, nrows=1)
    required = {
        spec for _, kind, spec in _PROXY_SPEC
        if kind == "col"
    }
    return {col: col in df.columns for col in sorted(required)}

#!/usr/bin/env python3
"""
Check which mode (mock vs real Diffusion-TS) was used to generate demo data files.
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def check_demo_data_npz():
    """Check demo_data.npz file."""
    data_path = ROOT_DIR / "data" / "synthetic" / "demo_data.npz"

    if not data_path.exists():
        print(f"❌ File not found: {data_path}")
        return None

    data = np.load(data_path, allow_pickle=True)
    metadata = data['metadata'].item()

    print("=" * 70)
    print("demo_data.npz (used by some pages)")
    print("=" * 70)
    print(f"Generated: {metadata.get('timestamp', 'Unknown')}")
    print(f"Samples: {metadata.get('n_benign', '?')} benign + {metadata.get('n_attacks', '?')} attacks")

    # Check if diffusion_mode is in metadata
    if 'diffusion_mode' in metadata:
        mode = metadata['diffusion_mode']
        if mode == 'real':
            print(f"✅ Generator Mode: REAL Diffusion-TS")
        else:
            print(f"📊 Generator Mode: MOCK (statistical)")
    else:
        print(f"⚠️  Generator Mode: NOT RECORDED (old format)")
        print(f"   → This file was generated before mode tracking was added")

    return metadata.get('diffusion_mode', 'unknown')


def check_npy_files():
    """Check individual .npy attack files."""
    synthetic_dir = ROOT_DIR / "data" / "synthetic"

    attack_files = [
        'slow_exfiltration_stealth_95.npy',
        'lotl_mimicry_stealth_90.npy',
        'beacon_stealth_85.npy',
        'benign_samples.npy'
    ]

    print("\n" + "=" * 70)
    print("Individual .npy files (used by Streamlit demo)")
    print("=" * 70)

    for filename in attack_files:
        filepath = synthetic_dir / filename
        if filepath.exists():
            # Get file modification time
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            print(f"✓ {filename}")
            print(f"  Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"✗ {filename} - NOT FOUND")

    print("\n⚠️  Note: .npy files don't store metadata about generation mode.")
    print("   To know which mode was used, check when they were last modified")
    print("   and whether Diffusion-TS was installed at that time.")


def check_current_mode():
    """Check which mode would be used if we generate now."""
    print("\n" + "=" * 70)
    print("Current Generator Configuration")
    print("=" * 70)

    try:
        from src.models import IoTDiffusionGenerator

        generator = IoTDiffusionGenerator(seq_length=128, feature_dim=12)
        generator.initialize()

        if not generator._mock_mode:
            print("✅ Current Mode: REAL Diffusion-TS")
            print(f"   Model: {type(generator.model).__name__}")
            print(f"   Location: lib/Diffusion-TS/")
            print("\n   If you regenerate data now, it will use REAL Diffusion-TS")
        else:
            print("📊 Current Mode: MOCK (statistical)")
            print("   Using built-in statistical generator")
            print("\n   If you regenerate data now, it will use MOCK mode")
            print("\n   To use real Diffusion-TS:")
            print("   → Run: bash scripts/install_diffusion_ts_compatible.sh")
    except Exception as e:
        print(f"❌ Error checking generator: {e}")


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("Data Generation Mode Checker")
    print("=" * 70)
    print()

    # Check demo_data.npz
    check_demo_data_npz()

    # Check .npy files
    check_npy_files()

    # Check current configuration
    check_current_mode()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("To regenerate data with current mode:")
    print("  → python scripts/precompute_demo_data.py --seed 42")
    print()
    print("To regenerate attack files:")
    print("  → python scripts/precompute_attacks.py --n-samples 5 --seed 42")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Adapter module for Diffusion-TS integration.

This module provides a compatibility layer between the iotsf_demo project
and the Diffusion-TS library, handling imports and path management.
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger

# Diffusion-TS import availability
DIFFUSION_TS_AVAILABLE = False
DIFFUSION_TS_PATH = None

def setup_diffusion_ts_path():
    """
    Setup Python path to include Diffusion-TS library.

    Returns:
        bool: True if Diffusion-TS is available, False otherwise
    """
    global DIFFUSION_TS_AVAILABLE, DIFFUSION_TS_PATH

    # Try to find Diffusion-TS in lib directory
    root_dir = Path(__file__).parent.parent.parent
    diffusion_ts_path = root_dir / "lib" / "Diffusion-TS"

    if diffusion_ts_path.exists():
        DIFFUSION_TS_PATH = str(diffusion_ts_path)
        if DIFFUSION_TS_PATH not in sys.path:
            sys.path.insert(0, DIFFUSION_TS_PATH)

        # Try importing (actual class name is Diffusion_TS, not GaussianDiffusion)
        try:
            from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
            from Models.interpretable_diffusion.transformer import Transformer
            DIFFUSION_TS_AVAILABLE = True
            logger.info(f"✅ Diffusion-TS loaded from: {diffusion_ts_path}")
            return True
        except ImportError as e:
            logger.warning(f"Diffusion-TS found but import failed: {e}")
            return False
    else:
        logger.debug(f"Diffusion-TS not found at: {diffusion_ts_path}")
        return False


def get_diffusion_model():
    """
    Import and return Diffusion-TS model classes.

    Returns:
        tuple: (Diffusion_TS, Transformer, model_utils) or (None, None, None)
    """
    if not DIFFUSION_TS_AVAILABLE:
        if not setup_diffusion_ts_path():
            return None, None, None

    try:
        from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
        from Models.interpretable_diffusion.transformer import Transformer
        from Models.interpretable_diffusion import model_utils
        return Diffusion_TS, Transformer, model_utils
    except ImportError as e:
        logger.error(f"Failed to import Diffusion-TS: {e}")
        return None, None, None


# Initialize on module import
setup_diffusion_ts_path()

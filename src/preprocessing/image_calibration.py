# 카메라 왜곡 보정
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

def load_instrinsic_txt(path: str | Path) -> np.ndarray:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"[load_intrinsic_txt] File not found: {path}")

    K = np.loadtxt(str(path), dtype=float)
    if K.shape != (3, 3):
        raise ValueError(f"[load_intrinsic_txt] Invalid intrinsic matrix shape: {K.shape}, expected (3, 3)")

    return K

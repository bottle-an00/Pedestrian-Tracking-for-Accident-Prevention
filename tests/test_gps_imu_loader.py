from pathlib import Path

from src.core.config import load_yaml
from src.io.gps_loader import *

def test_gps_imu_loader():
    loader = GpsImuLoader()

    cfg = load_yaml("configs/system.yaml")

    root = Path(cfg["test_data_dir"]["gps"])

    if not root.exists():
        return

    paths = loader.list_gps_imu_paths(root)
    assert isinstance(paths, list)
    assert len(paths) == 5

    if paths:
        for path, sample in loader.iter_data(root):
            assert path.exists()

            assert sample is not None
            assert isinstance(sample, GpsImuSample)
            assert hasattr(sample, "time")
            assert hasattr(sample, "x")
            assert hasattr(sample, "y")
            assert hasattr(sample, "z")
            assert hasattr(sample, "roll")
            assert hasattr(sample, "pitch")
            assert hasattr(sample, "heading")

from pathlib import Path
from src.core.config import load_yaml
from src.io.pcd_loader import PcdLoader

import numpy as np
import open3d as o3d

Pcd = o3d.geometry.PointCloud

def test_pcd_loader():
    loader = PcdLoader()

    cfg = load_yaml("configs/system.yaml")

    test_lidar_dir = Path(cfg["test_data_dir"]["lidar"])

    paths = loader.list_pcd_paths(test_lidar_dir)

    assert isinstance(paths, list)
    assert len(paths) == 5

    if paths:
        for path, pcds in loader.iter_pcds(test_lidar_dir, as_numpy=True):
            assert path.exists()
            for pcd in pcds:
                assert isinstance(pcd, np.ndarray)
                assert np.issubdtype(pcd.dtype, np.floating)

        for path, pcd in loader.iter_pcds(test_lidar_dir, as_numpy=False):
            assert path.exists()
            assert isinstance(pcd, Pcd)
            assert pcd.points is not None

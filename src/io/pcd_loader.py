# LiDAR data 로더
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import numpy as np
import open3d as o3d

PCD_EXTS = {".pcd"}

class PcdLoader:

    def __init__(self):
        pass
    def iter_pcd_paths(self, root_dir: str | Path) -> Iterable[Path]:
        root = Path(root_dir)

        if not root.exists():
            raise FileNotFoundError(f"[iter_pcd_paths] Directory not found: {root}")

        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in PCD_EXTS:
                yield p

    def list_pcd_paths(self, root_dir: str | Path) -> List[Path]:
        return list(self.iter_pcd_paths(root_dir))

    def load_pcd(self,
                 path: str | Path,
                 as_numpy: bool = False) -> o3d.geometry.PointCloud:

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[PcdLoader] File not found: {path}")

        pcd = o3d.io.read_point_cloud(str(path))

        if as_numpy:
            return np.asarray(pcd.points, dtype=np.float32)

        return pcd

    def iter_pcds(
        self,
        root_dir: str | Path,
        as_numpy: bool = False
    ) -> Iterator[Tuple[Path, o3d.geometry.PointCloud | np.ndarray]]:

        for pcd_path in self.iter_pcd_paths(root_dir):
            try:
                data = self.load_pcd(pcd_path, as_numpy=as_numpy)
            except Exception as e:
                print(f"[iter_pcds] Failed to read {pcd_path}: {e}")
                continue

            yield pcd_path, data
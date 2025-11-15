# 데이터 입출력, 동기화
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple

import cv2
import numpy as np

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

class ImageLoader:

    def __init__(self):
        pass

    def iter_img_paths(self, root_dir: str | Path) -> Iterable[Path]:
        root = Path(root_dir)

        if not root.exists():
            raise FileNotFoundError(f"[iter_img_paths] Directory not fount: {root}")

        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p

    def list_img_paths(self, root_dir:str | Path) -> List[Path]:
        return list(self.iter_img_paths(root_dir))

    def iter_imgs_cv2(self, root_dir: str | Path,
                      color: bool = True
    ) -> Iterator[Tuple[Path, np.ndarray]]:

        flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE

        for img_path in self.iter_img_paths(root_dir):
            img = cv2.imread(str(img_path), flag)
            if img is not None:
                yield img_path, img

from pathlib import Path

from src.core.config import load_yaml
from src.io.image_loader import ImageLoader


def test_image_reader_list_and_iter():
    loader = ImageLoader()

    cfg = load_yaml("configs/system.yaml")

    root = Path(cfg["test_data_dir"]["images"])

    if not root.exists():
        return

    paths = loader.list_img_paths(root)
    assert isinstance(paths, list)

    if paths:
        for path, img in loader.iter_imgs_cv2(root):
            assert path.exists()
            # img는 numpy ndarray여야 함
            assert hasattr(img, "shape")
            # 최소 2D (H, W) 혹은 3D (H, W, C)
            assert len(img.shape) in (2, 3)
            break

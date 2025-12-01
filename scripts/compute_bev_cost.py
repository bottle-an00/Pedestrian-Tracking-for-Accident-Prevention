"""Measure BEV warp performance across sequences/resolutions.

Usage:
    python scripts/compute_bev_cost.py --images <image_dir> [--calib_w CALIB_W --calib_h CALIB_H]

Outputs a small CSV summary with average time per frame and std.
"""
import time
import argparse
from pathlib import Path
import numpy as np
import cv2

from src.bev.bev_transformer import BevTransformer
from src.io.image_loader import ImageLoader


def benchmark(image_dir: Path, repeats: int = 3, warmup: int = 5, calib_image_size=None):
    image_loader = ImageLoader()
    transformer = BevTransformer(calib_image_size=calib_image_size)

    times = []

    imgs = [img for _, img in image_loader.iter_imgs_cv2(image_dir)]
    if len(imgs) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # warmup
    for i in range(min(warmup, len(imgs))):
        transformer.warp_image(imgs[i])

    for r in range(repeats):
        t0 = time.time()
        for img in imgs:
            transformer.warp_image(img)
        t1 = time.time()
        times.append((t1 - t0) / len(imgs))

    times = np.array(times)
    return times.mean(), times.std(), len(imgs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Path to image directory')
    parser.add_argument('--calib_w', type=int, default=None)
    parser.add_argument('--calib_h', type=int, default=None)
    args = parser.parse_args()

    img_dir = Path(args.images)
    if not img_dir.exists():
        raise ValueError('Image dir not found: ' + str(img_dir))

    calib_size = (args.calib_w, args.calib_h) if args.calib_w and args.calib_h else None

    mean, std, n = benchmark(img_dir, calib_image_size=calib_size)

    print(f"Images: {n}, mean_time_per_frame: {mean:.6f}s, std: {std:.6f}s")

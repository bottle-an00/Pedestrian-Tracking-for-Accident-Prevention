"""Example script: load an image, compute BEV, overlay risk zone, and save result."""
from pathlib import Path
import argparse
import cv2
from src.bev.bev_transformer import BevTransformer
from src.visualization.overlay_2d import Visualizer
from src.core.config import load_yaml


def main(image_path, out_path, front=10.0, width=2.0, calib_w=None, calib_h=None):
    cfg = load_yaml('configs/system.yaml')
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError('Cannot read image: ' + str(image_path))

    calib_size = (calib_w, calib_h) if calib_w and calib_h else None
    transformer = BevTransformer(calib_image_size=calib_size)
    bev = transformer.warp_image(img)

    vis = Visualizer()
    out = vis.draw_risk_zone_bev(bev, front_m=front, width_m=width, bev_resolution=float(cfg['bev']['resolution']))

    cv2.imwrite(str(out_path), out)
    print('Saved:', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--front', type=float, default=10.0)
    parser.add_argument('--width', type=float, default=2.0)
    parser.add_argument('--calib_w', type=int, default=None)
    parser.add_argument('--calib_h', type=int, default=None)
    args = parser.parse_args()
    main(args.image, args.out, front=args.front, width=args.width, calib_w=args.calib_w, calib_h=args.calib_h)

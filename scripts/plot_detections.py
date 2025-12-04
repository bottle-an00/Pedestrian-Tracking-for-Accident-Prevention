"""Plot detections_by_id.json and save PNGs.

Produces:
 - plots/trajectories.png (all trajectories in BEV x/y)
 - plots/{id}_trajectory.png, {id}_x_vs_frame.png, {id}_y_vs_frame.png

Usage: python scripts/plot_detections.py <json_path> [out_dir]
"""
import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(p):
    with open(p, 'r') as f:
        return json.load(f)


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def plot_all_trajectories(data, out_dir):
    plt.figure(figsize=(8, 8))
    for id_str, recs in data.items():
        xs = [r['x_m'] for r in recs]
        ys = [r['y_m'] for r in recs]
        plt.plot(xs, ys, marker='o', label=f'id={id_str}')
        plt.scatter(xs[0], ys[0], marker='s', s=30)
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('All trajectories (BEV meters)')
    plt.legend(loc='best')
    out = Path(out_dir) / 'trajectories.png'
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print('Saved', out)


def plot_per_id(id_str, recs, out_dir):
    id_dir = Path(out_dir) / f'id_{id_str}'
    ensure_dir(id_dir)
    frames = [r['frame'] for r in recs]
    times = [r['time'] for r in recs]
    xs = [r['x_m'] for r in recs]
    ys = [r['y_m'] for r in recs]

    # trajectory
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys, marker='o')
    plt.gca().invert_yaxis()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Trajectory id={id_str}')
    out1 = id_dir / f'{id_str}_trajectory.png'
    plt.savefig(out1, bbox_inches='tight')
    plt.close()

    # x vs frame
    plt.figure()
    plt.plot(frames, xs, marker='o')
    plt.xlabel('frame')
    plt.ylabel('x (m)')
    plt.title(f'id={id_str} x vs frame')
    out2 = id_dir / f'{id_str}_x_vs_frame.png'
    plt.savefig(out2, bbox_inches='tight')
    plt.close()

    # y vs frame
    plt.figure()
    plt.plot(frames, ys, marker='o')
    plt.xlabel('frame')
    plt.ylabel('y (m)')
    plt.title(f'id={id_str} y vs frame')
    out3 = id_dir / f'{id_str}_y_vs_frame.png'
    plt.savefig(out3, bbox_inches='tight')
    plt.close()

    print('Saved', out1, out2, out3)


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/plot_detections.py <json_path> [out_dir]')
        sys.exit(1)
    p = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else p.parent / 'plots'
    ensure_dir(out_dir)
    data = load_json(p)
    plot_all_trajectories(data, out_dir)
    # per id
    for id_str, recs in data.items():
        plot_per_id(id_str, recs, out_dir)

if __name__ == '__main__':
    main()

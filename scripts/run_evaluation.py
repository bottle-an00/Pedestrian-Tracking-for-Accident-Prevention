#!/usr/bin/env python3
"""평가 실행 CLI"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import MultiSequenceEvaluator
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/detector/yolo_detector.yaml")
    parser.add_argument("--data-root", type=str, help="Multi-sequence data root")
    parser.add_argument("--dataset-name", type=str, default=None,
                       help="Dataset subfolder name (e.g., '13_전방 보행자_val')")
    parser.add_argument("--image-dir", type=str, help="Single sequence image dir")
    parser.add_argument("--image-label-dir", type=str, help="Single sequence label dir")
    parser.add_argument("--lidar-label-dir", type=str, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--use-tracking", action="store_true")
    parser.add_argument("--output", type=str, default="results/eval_result.pkl")
    parser.add_argument("--pred-json-dir", type=str, default=None,
                        help="Directory containing prediction JSON files for trajectory evaluation")
    parser.add_argument("--pred-frame-start", type=int, default=None,
                        help="Optional: start frame index (inclusive) for ADE/FDE evaluation")
    parser.add_argument("--pred-frame-end", type=int, default=None,
                        help="Optional: end frame index (inclusive) for ADE/FDE evaluation")

    args = parser.parse_args()

    evaluator = MultiSequenceEvaluator(
        model_path=args.model,
        config_path=args.config,
        iou_threshold=args.iou_threshold,
        use_tracking=args.use_tracking,
    )

    if args.data_root:
        result = evaluator.evaluate_dataset(args.data_root, dataset_name=args.dataset_name)
    elif args.image_dir and args.image_label_dir:
        result = evaluator.evaluate_single_sequence(
            image_dir=args.image_dir,
            image_label_dir=args.image_label_dir,
        )
        # If pred JSONs provided, run trajectory ADE/FDE evaluation for this single sequence
        traj_results = None
        if args.pred_json_dir:
                seq_path = Path(args.image_dir).parent
                traj_results = evaluator.evaluate_prediction_sequence(
                    image_label_dir=args.image_label_dir,
                    lidar_label_dir=args.lidar_label_dir,
                    pred_json_dir=args.pred_json_dir,
                    T=20,
                    distance_threshold=2.0,
                    use_hungarian=False,
                    calibration_dir=seq_path,
                    result=result,
                    pred_frame_start=args.pred_frame_start,
                    pred_frame_end=args.pred_frame_end,
                )
    else:
        parser.error("Either --data-root or (--image-dir and --image-label-dir) required")

    # If trajectory results were computed, attach them to overall_metrics so the
    # saved text log includes ADE/FDE summary.
    try:
        if 'traj_results' in locals() and traj_results:
            result.overall_metrics['trajectory'] = {
                'mean_ADE': float(traj_results.get('mean_ADE', float('nan'))),
                'mean_FDE': float(traj_results.get('mean_FDE', float('nan'))),
                'min_ADE': float(traj_results.get('min_ADE', float('nan'))),
                'max_ADE': float(traj_results.get('max_ADE', float('nan'))),
                'min_FDE': float(traj_results.get('min_FDE', float('nan'))),
                'max_FDE': float(traj_results.get('max_FDE', float('nan'))),
            }
    except Exception:
        pass

    result.save(args.output)

    # 요약 출력
    det = result.overall_metrics.get("detection", {})
    trk = result.overall_metrics.get("tracking", {})
    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Sequences: {len(result.sequences)}, Frames: {sum(s.n_frames for s in result.sequences.values())}")
    print(f"{'='*60}")
    print(f"Detection:  AP@50={det.get('AP@50', 0):.4f}  Recall={det.get('Recall@50', 0):.4f}  Precision={det.get('Precision@50', 0):.4f}")
    print(f"Tracking:   HOTA={trk.get('HOTA', 0):.4f}   MOTA={trk.get('MOTA', 0):.4f}    IDS={trk.get('ID_Switch_total', 0)}")
    print(f"{'='*60}")
    # Trajectory summary (if computed)
    try:
        if 'traj_results' in locals() and traj_results:
            mean_ADE = traj_results.get('mean_ADE')
            mean_FDE = traj_results.get('mean_FDE')
            if mean_ADE is not None and mean_FDE is not None:
                print(f"Trajectory: mean_ADE={mean_ADE:.4f}  mean_FDE={mean_FDE:.4f}")
            else:
                print("Trajectory: No valid ADE/FDE samples collected.")
    except Exception:
        pass

    print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""GT + Prediction 비교 시각화 CLI"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization import visualize_gt_pred


def main():
    parser = argparse.ArgumentParser(description="Visualize GT vs Prediction")
    parser.add_argument("--result", type=str, default="results/eval_result.pkl",
                       help="Evaluation result pkl file")
    parser.add_argument("--data-root", type=str, required=True,
                       help="Data root directory")
    parser.add_argument("--dataset-name", type=str, default="13_전방 보행자_val",
                       help="Dataset name")
    parser.add_argument("--sequence", type=str, required=True,
                       help="Sequence number (e.g., 053)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--show", action="store_true",
                       help="Show images while processing")

    args = parser.parse_args()

    visualize_gt_pred(
        result_pkl=args.result,
        data_root=args.data_root,
        sequence=args.sequence,
        dataset_name=args.dataset_name,
        output_dir=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()

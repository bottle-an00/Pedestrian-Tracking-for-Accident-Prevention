# YOLO 단독 실행 스크립트
#
# 사용법:
#     PYTHONPATH=. python scripts/run_detector.py --base_dir "/path/to/dataset" --sequence 053

import argparse
from src.app.main import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detector Pipeline")
    parser.add_argument("--base_dir", type=str, required=True, help="데이터셋 시퀀스들이 있는 상위 폴더")
    parser.add_argument("--sequence", type=str, default=None, help="특정 시퀀스만 실행 (예: 001)")
    parser.add_argument("--config", type=str, default="configs/detector/yolo_detector.yaml", help="설정 파일 경로")

    args = parser.parse_args()

    pipeline = Pipeline(config_path=args.config)
    pipeline.run(base_dir=args.base_dir, target_sequence=args.sequence)

"""평가 실행 모듈

- SequenceResult: 단일 시퀀스 평가 결과
- EvaluationResult: 전체 평가 결과 (다중 시퀀스)
- SequenceEvaluator: 단일 시퀀스 평가기
- MultiSequenceEvaluator: 다중 시퀀스 평가기
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import cv2

from ..detection.yolo_detector import YoloDetector
from ..tracking.tracker import ByteTracker
from ..core.config import load_yaml
from ..io.image_loader import ImageLoader

from .gt_loader import GTLoader
from .evaluators import DetectionEvaluator, TrackingEvaluator
from .matching import compute_iou_matrix


@dataclass
class SequenceResult:
    """단일 시퀀스 평가 결과"""
    sequence_id: str
    n_frames: int = 0
    detection_metrics: Dict[str, float] = field(default_factory=dict)
    tracking_metrics: Dict[str, float] = field(default_factory=dict)
    localization_metrics: Dict[str, float] = field(default_factory=dict)
    matched_pairs: List[Dict] = field(default_factory=list)
    unmatched_gt: List[Dict] = field(default_factory=list)
    unmatched_pred: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "sequence_id": self.sequence_id,
            "n_frames": self.n_frames,
            "detection": self.detection_metrics,
            "tracking": self.tracking_metrics,
            "localization": self.localization_metrics,
        }


@dataclass
class EvaluationResult:
    """전체 평가 결과 (다중 시퀀스)"""
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequences: Dict[str, SequenceResult] = field(default_factory=dict)
    overall_metrics: Dict[str, Dict] = field(default_factory=dict)

    def save(self, path: Union[str, Path]):
        """pkl로 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": self.metadata,
            "sequences": {k: {
                "sequence_id": v.sequence_id,
                "n_frames": v.n_frames,
                "detection_metrics": v.detection_metrics,
                "tracking_metrics": v.tracking_metrics,
                "localization_metrics": v.localization_metrics,
                "matched_pairs": v.matched_pairs,
                "unmatched_gt": v.unmatched_gt,
                "unmatched_pred": v.unmatched_pred,
            } for k, v in self.sequences.items()},
            "overall_metrics": self.overall_metrics,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Results saved: {path}")

        txt_path = path.with_suffix(".txt")
        self.save_txt(txt_path)

    def save_txt(self, path: Union[str, Path]):
        """텍스트 로그 파일 저장"""
        path = Path(path)
        lines = []

        lines.append("=" * 80)
        lines.append("EVALUATION RESULTS")
        lines.append("=" * 80)
        lines.append("")

        lines.append("[Metadata]")
        lines.append(f"  Timestamp: {self.metadata.get('timestamp', 'N/A')}")
        lines.append(f"  Model: {self.metadata.get('model_path', 'N/A')}")
        lines.append(f"  Data Root: {self.metadata.get('data_root', 'N/A')}")
        lines.append(f"  Dataset: {self.metadata.get('dataset_name', 'N/A')}")
        lines.append(f"  IoU Threshold: {self.metadata.get('iou_threshold', 'N/A')}")
        lines.append(f"  Use Tracking: {self.metadata.get('use_tracking', 'N/A')}")
        lines.append(f"  Sequences: {len(self.sequences)}")
        lines.append(f"  Total Frames: {sum(s.n_frames for s in self.sequences.values())}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("[Overall Metrics]")
        lines.append("=" * 80)
        for name, metrics in self.overall_metrics.items():
            lines.append(f"\n  [{name.upper()}]")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k:<20}: {v:.4f}")
                else:
                    lines.append(f"    {k:<20}: {v}")
        lines.append("")

        lines.append("=" * 80)
        lines.append("[Per-Sequence Summary]")
        lines.append("=" * 80)
        lines.append(f"{'Seq':<8} {'Frames':>6} {'TP':>6} {'FP':>6} {'FN':>6} {'AP@50':>8} {'Prec':>7} {'Recall':>7} {'MOTA':>8} {'HOTA':>8} {'IDS':>5} {'FootUV':>10}")
        lines.append("-" * 100)

        for seq_id, seq in self.sequences.items():
            det = seq.detection_metrics
            trk = seq.tracking_metrics
            loc = seq.localization_metrics
            lines.append(
                f"{seq_id:<8} "
                f"{seq.n_frames:>6} "
                f"{det.get('TP', 0):>6} "
                f"{det.get('FP', 0):>6} "
                f"{det.get('FN', 0):>6} "
                f"{det.get('AP@50', 0):>8.4f} "
                f"{det.get('Precision@50', 0):>7.3f} "
                f"{det.get('Recall@50', 0):>7.3f} "
                f"{trk.get('MOTA', 0):>8.4f} "
                f"{trk.get('HOTA', 0):>8.4f} "
                f"{trk.get('ID_Switch', 0):>5} "
                f"{loc.get('FootUV_MAE', 0):>8.2f}px"
            )
        lines.append("")

        lines.append("=" * 80)
        lines.append("[Per-Sequence Detail]")
        lines.append("=" * 80)

        for seq_id, seq in self.sequences.items():
            lines.append(f"\n--- {seq_id} ({seq.n_frames} frames) ---")

            lines.append("  Detection:")
            for k, v in seq.detection_metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k:<20}: {v:.4f}")
                else:
                    lines.append(f"    {k:<20}: {v}")

            lines.append("  Tracking:")
            for k, v in seq.tracking_metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k:<20}: {v:.4f}")
                else:
                    lines.append(f"    {k:<20}: {v}")

            lines.append("  Localization:")
            for k, v in seq.localization_metrics.items():
                if isinstance(v, float):
                    lines.append(f"    {k:<20}: {v:.4f}")
                else:
                    lines.append(f"    {k:<20}: {v}")

        lines.append("")
        lines.append("=" * 80)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Log saved: {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationResult":
        """pkl에서 로드"""
        with open(path, "rb") as f:
            data = pickle.load(f)

        result = cls()
        result.metadata = data.get("metadata", {})
        result.overall_metrics = data.get("overall_metrics", {})

        for k, v in data.get("sequences", {}).items():
            result.sequences[k] = SequenceResult(
                sequence_id=v["sequence_id"],
                n_frames=v["n_frames"],
                detection_metrics=v["detection_metrics"],
                tracking_metrics=v["tracking_metrics"],
                localization_metrics=v["localization_metrics"],
                matched_pairs=v.get("matched_pairs", []),
                unmatched_gt=v.get("unmatched_gt", []),
                unmatched_pred=v.get("unmatched_pred", []),
            )
        return result

    def to_dataframes(self) -> Dict[str, Any]:
        """DataFrame으로 변환"""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pip install pandas")

        dfs = {}

        rows = []
        for seq_id, seq in self.sequences.items():
            row = {"sequence_id": seq_id, "n_frames": seq.n_frames}
            row.update({f"det_{k}": v for k, v in seq.detection_metrics.items()})
            row.update({f"trk_{k}": v for k, v in seq.tracking_metrics.items()})
            row.update({f"loc_{k}": v for k, v in seq.localization_metrics.items()})
            rows.append(row)
        if rows:
            dfs["sequences"] = pd.DataFrame(rows)

        overall_rows = []
        for name, metrics in self.overall_metrics.items():
            row = {"metric_type": name}
            row.update(metrics)
            overall_rows.append(row)
        if overall_rows:
            dfs["overall"] = pd.DataFrame(overall_rows)

        all_pairs = []
        for seq_id, seq in self.sequences.items():
            for pair in seq.matched_pairs:
                pair_with_seq = {"sequence_id": seq_id}
                pair_with_seq.update(pair)
                all_pairs.append(pair_with_seq)
        if all_pairs:
            dfs["matched_pairs"] = pd.DataFrame(all_pairs)

        all_unmatched_gt = []
        for seq_id, seq in self.sequences.items():
            for item in seq.unmatched_gt:
                item_with_seq = {"sequence_id": seq_id}
                item_with_seq.update(item)
                all_unmatched_gt.append(item_with_seq)
        if all_unmatched_gt:
            dfs["unmatched_gt"] = pd.DataFrame(all_unmatched_gt)

        all_unmatched_pred = []
        for seq_id, seq in self.sequences.items():
            for item in seq.unmatched_pred:
                item_with_seq = {"sequence_id": seq_id}
                item_with_seq.update(item)
                all_unmatched_pred.append(item_with_seq)
        if all_unmatched_pred:
            dfs["unmatched_pred"] = pd.DataFrame(all_unmatched_pred)

        return dfs

    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS")
        print("=" * 70)
        print(f"Model: {self.metadata.get('model_path', 'N/A')}")
        print(f"Sequences: {len(self.sequences)}")
        print(f"Total Frames: {sum(s.n_frames for s in self.sequences.values())}")
        print("=" * 70)

        print("\n[Overall Metrics]")
        for name, metrics in self.overall_metrics.items():
            print(f"\n  {name}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

        if len(self.sequences) > 1:
            print("\n[Per-Sequence Summary]")
            print(f"{'Sequence':<15} {'Frames':>7} {'AP@50':>8} {'MOTA':>8} {'HOTA':>8} {'FootUV':>10}")
            print("-" * 60)
            for seq_id, seq in self.sequences.items():
                ap50 = seq.detection_metrics.get("AP@50", 0)
                mota = seq.tracking_metrics.get("MOTA", 0)
                hota = seq.tracking_metrics.get("HOTA", 0)
                foot = seq.localization_metrics.get("FootUV_MAE", 0)
                print(f"{seq_id:<15} {seq.n_frames:>7} {ap50:>8.4f} {mota:>8.4f} {hota:>8.4f} {foot:>10.2f}px")


class SequenceEvaluator:
    """단일 시퀀스 평가기 (Detection + Tracking)"""

    def __init__(
        self,
        detector: Union[YoloDetector, ByteTracker],
        iou_threshold: float = 0.3,
    ):
        self.detector = detector
        self.iou_threshold = iou_threshold

        self.gt_loader = GTLoader(target_classes=["pedestrian"])
        self.img_loader = ImageLoader()

        self.det_eval = DetectionEvaluator(iou_threshold)
        self.track_eval = TrackingEvaluator(iou_threshold)

    def reset(self):
        self.det_eval.reset()
        self.track_eval.reset()
        self.matched_pairs = []
        self.unmatched_gt = []
        self.unmatched_pred = []

    def _collect_matches(self, gt_objects, predictions, frame_id):
        """시각화용 매칭 정보 수집"""
        gt_boxes = [obj.bbox for obj in gt_objects if obj.bbox is not None]
        pred_boxes = [self._convert_bbox(p.get("bbox")) for p in predictions if p.get("bbox")]

        if not gt_boxes or not pred_boxes:
            for obj in gt_objects:
                if obj.bbox:
                    self.unmatched_gt.append({
                        "frame_id": frame_id,
                        "gt_bbox": obj.bbox,
                        "gt_track_id": obj.track_id,
                    })
            for p in predictions:
                if p.get("bbox"):
                    self.unmatched_pred.append({
                        "frame_id": frame_id,
                        "pred_bbox": p.get("bbox"),
                        "pred_track_id": p.get("track_id"),
                    })
            return

        iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)
        matches, unmatched_gt_idx, unmatched_pred_idx = self._greedy_match(iou_matrix, self.iou_threshold)

        for gt_idx, pred_idx, iou in matches:
            gt_obj = gt_objects[gt_idx]
            pred = predictions[pred_idx]
            self.matched_pairs.append({
                "frame_id": frame_id,
                "gt_bbox": gt_obj.bbox,
                "gt_track_id": gt_obj.track_id,
                "pred_bbox": pred.get("bbox"),
                "pred_track_id": pred.get("track_id"),
                "iou": float(iou),
            })

        for gt_idx in unmatched_gt_idx:
            gt_obj = gt_objects[gt_idx]
            self.unmatched_gt.append({
                "frame_id": frame_id,
                "gt_bbox": gt_obj.bbox,
                "gt_track_id": gt_obj.track_id,
            })

        for pred_idx in unmatched_pred_idx:
            pred = predictions[pred_idx]
            self.unmatched_pred.append({
                "frame_id": frame_id,
                "pred_bbox": pred.get("bbox"),
                "pred_track_id": pred.get("track_id"),
            })

    def _convert_bbox(self, bbox):
        """[x,y,w,h] -> (x1,y1,x2,y2)"""
        if bbox is None:
            return None
        x, y, w, h = bbox
        return (x, y, x + w, y + h)

    def _greedy_match(self, iou_matrix, threshold):
        """Greedy IoU 매칭"""
        n_gt, n_pred = iou_matrix.shape
        matrix = iou_matrix.copy()
        matches, matched_gt, matched_pred = [], set(), set()

        while True:
            max_iou = matrix.max()
            if max_iou < threshold:
                break
            gt_idx, pred_idx = np.unravel_index(matrix.argmax(), matrix.shape)
            matches.append((gt_idx, pred_idx, max_iou))
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            matrix[gt_idx, :] = -1
            matrix[:, pred_idx] = -1

        return matches, [i for i in range(n_gt) if i not in matched_gt], \
               [i for i in range(n_pred) if i not in matched_pred]

    def evaluate(
        self,
        image_dir: Path,
        image_label_dir: Path,
        sequence_id: str = None,
    ) -> SequenceResult:
        """시퀀스 평가 실행"""
        self.reset()
        sequence_id = sequence_id or image_dir.name

        img_paths = self.img_loader.list_img_paths(image_dir)
        n_frames = 0

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            frame_id = img_path.stem
            img_label = image_label_dir / f"{frame_id}.json"

            if not img_label.exists():
                continue

            gt_frame = self.gt_loader.load_frame(img_label)

            if hasattr(self.detector, 'process'):
                predictions = self.detector.process(img)
            else:
                predictions = self.detector.detect(img)
                for i, pred in enumerate(predictions):
                    pred["track_id"] = pred.get("id", i)

            for pred in predictions:
                if "id" in pred and "track_id" not in pred:
                    pred["track_id"] = pred["id"]

            gt_objects = gt_frame.get_pedestrians()

            self.det_eval.update(gt_objects, predictions, frame_id)
            self.track_eval.update(gt_objects, predictions, frame_id)
            self._collect_matches(gt_objects, predictions, frame_id)

            n_frames += 1

        return SequenceResult(
            sequence_id=sequence_id,
            n_frames=n_frames,
            detection_metrics=self.det_eval.compute(),
            tracking_metrics=self.track_eval.compute(),
            localization_metrics={},
            matched_pairs=self.matched_pairs,
            unmatched_gt=self.unmatched_gt,
            unmatched_pred=self.unmatched_pred,
        )


class MultiSequenceEvaluator:
    """다중 시퀀스 평가기"""

    def __init__(
        self,
        model_path: str,
        config_path: str = "configs/detector/yolo_detector.yaml",
        iou_threshold: float = 0.3,
        use_tracking: bool = True,
        half: bool = True,
    ):
        self.model_path = model_path
        self.iou_threshold = iou_threshold
        self.use_tracking = use_tracking

        cfg = load_yaml(config_path)
        conf_thres = cfg.get("conf_threshold", cfg.get("conf_thres", 0.3))
        if not isinstance(conf_thres, (dict, float, int)):
            conf_thres = 0.3
        target_classes = cfg.get("target_classes", ["person"])
        imgsz = cfg.get("imgsz", 1280)

        if use_tracking:
            self.detector = ByteTracker(model_path, conf_thres, target_classes, imgsz, half=half)
        else:
            self.detector = YoloDetector(model_path, conf_thres, target_classes, imgsz, half=half)

        self.seq_evaluator = SequenceEvaluator(self.detector, iou_threshold)

    def evaluate_dataset(
        self,
        data_root: Union[str, Path],
        dataset_name: str = None,
        image_subdir: str = "image0",
        image_label_subdir: str = "image0",
        lidar_label_subdir: str = "lidar",
    ) -> EvaluationResult:
        """데이터셋 전체 평가 (다중 시퀀스)"""
        data_root = Path(data_root)

        result = EvaluationResult()
        result.metadata = {
            "model_path": self.model_path,
            "data_root": str(data_root),
            "dataset_name": dataset_name,
            "iou_threshold": self.iou_threshold,
            "use_tracking": self.use_tracking,
            "timestamp": datetime.now().isoformat(),
        }

        raw_data_root = data_root / "원천데이터"
        label_data_root = data_root / "라벨링데이터"

        if not raw_data_root.exists():
            raw_data_root = data_root
        if not label_data_root.exists():
            label_data_root = data_root

        if dataset_name:
            raw_data_root = raw_data_root / dataset_name
            label_data_root = label_data_root / dataset_name

        if not dataset_name:
            subdirs = [d for d in raw_data_root.iterdir() if d.is_dir()]
            if subdirs and not (subdirs[0] / image_subdir).exists():
                for subdir in subdirs:
                    if (subdir / next(subdir.iterdir()).name / image_subdir).exists():
                        raw_data_root = subdir
                        label_data_root = label_data_root / subdir.name
                        result.metadata["dataset_name"] = subdir.name
                        break

        sequence_dirs = sorted([d for d in raw_data_root.iterdir() if d.is_dir()])

        for seq_dir in sequence_dirs:
            seq_id = seq_dir.name

            image_dir = seq_dir / image_subdir
            image_label_dir = label_data_root / seq_id / image_label_subdir

            if not image_dir.exists() or not image_label_dir.exists():
                continue

            seq_result = self.seq_evaluator.evaluate(
                image_dir=image_dir,
                image_label_dir=image_label_dir,
                sequence_id=seq_id,
            )

            result.sequences[seq_id] = seq_result

        result.overall_metrics = self._compute_overall_metrics(result.sequences)

        return result

    def evaluate_single_sequence(
        self,
        image_dir: Union[str, Path],
        image_label_dir: Union[str, Path],
        sequence_id: str = None,
    ) -> EvaluationResult:
        """단일 시퀀스 평가"""
        image_dir = Path(image_dir)
        image_label_dir = Path(image_label_dir)

        result = EvaluationResult()
        result.metadata = {
            "model_path": self.model_path,
            "image_dir": str(image_dir),
            "iou_threshold": self.iou_threshold,
            "use_tracking": self.use_tracking,
            "timestamp": datetime.now().isoformat(),
        }

        seq_id = sequence_id or image_dir.name
        seq_result = self.seq_evaluator.evaluate(
            image_dir=image_dir,
            image_label_dir=image_label_dir,
            sequence_id=seq_id,
        )

        result.sequences[seq_id] = seq_result
        result.overall_metrics = self._compute_overall_metrics(result.sequences)

        return result

    def _compute_overall_metrics(self, sequences: Dict[str, SequenceResult]) -> Dict[str, Dict]:
        """시퀀스별 메트릭 평균 계산"""
        if not sequences:
            return {}

        det_metrics = {}
        for key in ["AP@50", "AP@75", "mAP@50:95", "Precision@50", "Recall@50", "F1@50"]:
            values = [s.detection_metrics.get(key, 0) for s in sequences.values()]
            det_metrics[key] = np.mean(values) if values else 0.0

        trk_metrics = {}
        for key in ["HOTA", "MOTA", "MOTP", "IDF1"]:
            values = [s.tracking_metrics.get(key, 0) for s in sequences.values()]
            trk_metrics[key] = np.mean(values) if values else 0.0

        trk_metrics["ID_Switch_total"] = sum(s.tracking_metrics.get("ID_Switch", 0) for s in sequences.values())

        loc_metrics = {}
        for key in ["FootUV_MAE", "FootUV_STD", "BEV_MAE", "BEV_STD"]:
            values = [s.localization_metrics.get(key, 0) for s in sequences.values() if s.localization_metrics.get(key, 0) > 0]
            loc_metrics[key] = np.mean(values) if values else 0.0

        return {
            "detection": det_metrics,
            "tracking": trk_metrics,
            "localization": loc_metrics,
        }

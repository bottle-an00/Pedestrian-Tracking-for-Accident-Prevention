"""GT + Prediction 비교 시각화 모듈"""

from pathlib import Path
from typing import Dict, List, Optional
import cv2
import pickle
from tqdm import tqdm


class GTPredVisualizer:
    """GT와 Prediction 결과를 비교 시각화하는 클래스"""

    # 색상 정의 (BGR)
    COLOR_TP_GT = (0, 255, 0)      # Green - GT (TP)
    COLOR_TP_PRED = (255, 0, 0)   # Blue - Pred (TP)
    COLOR_FN = (0, 0, 255)         # Red - False Negative
    COLOR_FP = (255, 0, 255)       # Magenta - False Positive
    COLOR_TEXT = (255, 255, 255)   # White
    COLOR_IOU = (255, 255, 0)      # Cyan

    def __init__(self, result_pkl: str):
        """
        Args:
            result_pkl: 평가 결과 pkl 파일 경로
        """
        with open(result_pkl, "rb") as f:
            self.results = pickle.load(f)

    def get_sequences(self) -> List[str]:
        """사용 가능한 시퀀스 목록 반환"""
        return list(self.results.get("sequences", {}).keys())

    def visualize_sequence(
        self,
        sequence: str,
        data_root: str,
        dataset_name: str = "13_전방 보행자_val",
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> Dict:
        """
        시퀀스의 GT-Pred 비교 시각화 생성

        Args:
            sequence: 시퀀스 ID (e.g., "053", "073")
            data_root: 데이터 루트 경로
            dataset_name: 데이터셋 이름
            output_dir: 출력 디렉토리 (None이면 자동 생성)
            show: 시각화 중 화면 표시 여부

        Returns:
            시각화 통계 딕셔너리
        """
        seq_data = self.results["sequences"].get(sequence)
        if seq_data is None:
            print(f"Sequence {sequence} not found in results")
            return {}

        # 프레임별 데이터 매핑
        frame_matches = self._build_frame_mapping(seq_data.get("matched_pairs", []))
        frame_unmatched_gt = self._build_frame_mapping(seq_data.get("unmatched_gt", []))
        frame_unmatched_pred = self._build_frame_mapping(seq_data.get("unmatched_pred", []))

        # 경로 설정
        data_root = Path(data_root)
        image_dir = data_root / "원천데이터" / dataset_name / sequence / "image0"

        if output_dir:
            out_path = Path(output_dir)
        else:
            out_path = Path("results") / "gt_pred_vis" / sequence
        out_path.mkdir(parents=True, exist_ok=True)

        # 이미지 파일 찾기
        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(image_dir.glob("*.jpg"))

        stats = {
            "sequence": sequence,
            "n_images": len(image_files),
            "n_matched": len(seq_data.get("matched_pairs", [])),
            "n_fn": len(seq_data.get("unmatched_gt", [])),
            "n_fp": len(seq_data.get("unmatched_pred", [])),
            "output_dir": str(out_path),
        }

        print(f"Sequence: {sequence}")
        print(f"Images: {stats['n_images']}")
        print(f"Matched pairs (TP): {stats['n_matched']}")
        print(f"Unmatched GT (FN): {stats['n_fn']}")
        print(f"Unmatched Pred (FP): {stats['n_fp']}")

        for img_path in tqdm(image_files, desc=f"[{sequence}]"):
            frame_id = img_path.stem
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # 시각화 그리기
            matches = frame_matches.get(frame_id, [])
            unmatched_gts = frame_unmatched_gt.get(frame_id, [])
            unmatched_preds = frame_unmatched_pred.get(frame_id, [])

            self._draw_matches(img, matches)
            self._draw_false_negatives(img, unmatched_gts)
            self._draw_false_positives(img, unmatched_preds)
            self._draw_stats_overlay(img, frame_id, len(matches), len(unmatched_gts), len(unmatched_preds))

            cv2.imwrite(str(out_path / f"{frame_id}.jpg"), img)

            if show:
                cv2.imshow("GT vs Pred", img)
                if cv2.waitKey(1) == 27:
                    break

        if show:
            cv2.destroyAllWindows()

        print(f"Done! Saved to {out_path}")
        return stats

    def _build_frame_mapping(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """프레임 ID별로 아이템 그룹화"""
        mapping = {}
        for item in items:
            fid = item.get("frame_id")
            if fid not in mapping:
                mapping[fid] = []
            mapping[fid].append(item)
        return mapping

    def _draw_matches(self, img, matches: List[Dict]):
        """TP 매칭 그리기 (GT: green, Pred: blue)"""
        for m in matches:
            # GT bbox (green)
            gt_bbox = m.get("gt_bbox")
            if gt_bbox:
                x1, y1, x2, y2 = map(int, gt_bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_TP_GT, 2)
                label = f"GT:{m.get('gt_track_id', '?')}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TP_GT, 2)

            # Pred bbox (blue) - format: [x, y, w, h]
            pred_bbox = m.get("pred_bbox")
            if pred_bbox:
                px, py, pw, ph = pred_bbox
                px1, py1, px2, py2 = int(px), int(py), int(px + pw), int(py + ph)
                cv2.rectangle(img, (px1, py1), (px2, py2), self.COLOR_TP_PRED, 2)
                label = f"P:{m.get('pred_track_id', '?')}"
                cv2.putText(img, label, (px1, py2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TP_PRED, 2)

            # IoU 표시
            iou = m.get("iou", 0)
            if gt_bbox:
                cv2.putText(img, f"IoU:{iou:.2f}", (int(gt_bbox[0]), int(gt_bbox[1]) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_IOU, 1)

    def _draw_false_negatives(self, img, items: List[Dict]):
        """FN 그리기 (red)"""
        for item in items:
            gt_bbox = item.get("gt_bbox")
            if gt_bbox:
                x1, y1, x2, y2 = map(int, gt_bbox)
                cv2.rectangle(img, (x1, y1), (x2, y2), self.COLOR_FN, 2)
                label = f"FN:{item.get('gt_track_id', '?')}"
                cv2.putText(img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_FN, 2)

    def _draw_false_positives(self, img, items: List[Dict]):
        """FP 그리기 (magenta)"""
        for item in items:
            pred_bbox = item.get("pred_bbox")
            if pred_bbox:
                px, py, pw, ph = pred_bbox
                px1, py1, px2, py2 = int(px), int(py), int(px + pw), int(py + ph)
                cv2.rectangle(img, (px1, py1), (px2, py2), self.COLOR_FP, 2)
                cv2.putText(img, "FP", (px1, py1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_FP, 2)

    def _draw_stats_overlay(self, img, frame_id: str, n_tp: int, n_fn: int, n_fp: int):
        """프레임 통계 오버레이"""
        cv2.putText(img, f"Frame: {frame_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
        cv2.putText(img, f"TP:{n_tp} FN:{n_fn} FP:{n_fp}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.COLOR_TEXT, 2)
        # Legend
        cv2.putText(img, "Green=GT(TP) Blue=Pred Red=FN Magenta=FP",
                   (10, img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)


def visualize_gt_pred(
    result_pkl: str,
    data_root: str,
    sequence: str,
    dataset_name: str = "13_전방 보행자_val",
    output_dir: Optional[str] = None,
    show: bool = False,
) -> Dict:
    """GT-Pred 시각화 함수 (하위 호환성)"""
    visualizer = GTPredVisualizer(result_pkl)
    return visualizer.visualize_sequence(
        sequence=sequence,
        data_root=data_root,
        dataset_name=dataset_name,
        output_dir=output_dir,
        show=show,
    )

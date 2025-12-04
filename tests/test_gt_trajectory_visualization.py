"""
GT Trajectory Visualization Test

이 테스트는 GT 라벨에서 BEV 범위 내에 있는 객체들의 궤적을 시각화하고 JSON으로 저장합니다.
- BEV 이미지 상에 GT 객체의 과거/미래 궤적 시각화
- 각 프레임마다 JSON 파일로 궤적 데이터 저장
"""

from pathlib import Path
import numpy as np
import cv2
import json

from src.core.config import load_yaml
from src.io.image_loader import ImageLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.visualization.overlay_2d import Visualizer
from src.evaluation.gt_loader import GTLoader


def test_gt_trajectory_visualization():
    """GT 궤적 시각화 및 JSON 저장 테스트"""
    
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    image_dir = Path(cfg[root_dir]["images"])
    out_root = Path(cfg[root_dir]["outputs"]) / "gt_trajectory_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    # 출력 디렉토리 설정
    out_gt_bev_traj = out_root / "gt_bev_trajectory"
    out_gt_traj_json = out_root / "gt_trajectory_json"
    out_bev_base = out_root / "bev_base"
    
    for p in [out_gt_bev_traj, out_gt_traj_json, out_bev_base]:
        p.mkdir(parents=True, exist_ok=True)

    # Calibration 로드
    calib_loader = CalibrationInfoLoader()
    intrinsics = calib_loader.load_camera_calibration(
        Path(cfg[root_dir]["calibration"]["calib_Camera"])
    )
    extrinsics = calib_loader.load_camera_extrinsics(
        Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"])
    )
    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    # 이미지 로더 및 시각화 도구
    image_loader = ImageLoader()
    vis = Visualizer()

    # GT 데이터 로드
    seq_id = Path(image_dir).parent.name
    gt_dir = Path('data') / 'gt' / seq_id / Path(image_dir).name
    gt_loader = GTLoader(target_classes=['pedestrian'])
    
    # GT 프레임 맵 생성
    gt_frame_map = {}
    if gt_dir.exists():
        print(f"[GT Loader] Loading GT data from: {gt_dir}")
        for p in sorted(gt_dir.glob('*.json')):
            stem = p.stem
            fidx = None
            parts = stem.split('_')
            if parts and parts[-1].lstrip('0').isdigit():
                try:
                    fidx = int(parts[-1])
                except Exception:
                    fidx = None
            
            if fidx is None:
                try:
                    fidx = int(stem)
                except Exception:
                    continue
            
            try:
                data = gt_loader.load_image_label(p)
                objs, _ = gt_loader.parse_image_label(data)
                gt_frame_map[fidx] = objs
                print(f"  Frame {fidx}: {len(objs)} GT objects")
            except Exception as e:
                print(f"  Frame {fidx}: Error loading - {e}")
                gt_frame_map[fidx] = []
    else:
        print(f"[GT Loader] GT directory not found: {gt_dir}")
        print("[GT Loader] Skipping GT trajectory visualization")
        return

    print(f"[GT Loader] Loaded {len(gt_frame_map)} frames")

    # 이미지 처리
    img_iter = image_loader.iter_imgs_cv2(image_dir)
    frame_idx = 0

    for img_path, image in img_iter:
        print(f"\n[Frame {frame_idx}] Processing {img_path.name}")

        # BEV 이미지 생성
        bev_img = H.warp(image)
        bev_h, bev_w = bev_img.shape[:2]
        
        # 기본 BEV 이미지 저장 (참고용)
        cv2.imwrite(str(out_bev_base / f"bev_{img_path.name}"), bev_img)

        # GT Trajectory 시각화 및 JSON 데이터 생성
        gt_bev_traj_img = bev_img.copy()
        gt_trajectory_data = {
            "frame_id": frame_idx,
            "image_name": img_path.name,
            "bev_dimensions": {"height": bev_h, "width": bev_w},
            "trajectories": []
        }
        
        # 현재 프레임의 GT 객체 가져오기
        current_gt_objects = gt_frame_map.get(frame_idx, [])
        print(f"  Current GT objects: {len(current_gt_objects)}")
        
        objects_in_bev = 0
        
        for gt_obj in current_gt_objects:
            if gt_obj.foot_uv is None:
                continue
            
            # GT foot_uv를 BEV 좌표로 변환
            u_gt, v_gt = gt_obj.foot_uv
            try:
                bx_gt, by_gt = H.pixel_to_bev_warp(u_gt, v_gt)
            except Exception as e:
                print(f"    Object conversion error: {e}")
                continue
            
            # BEV 범위 내에 있는지 확인 (필터링)
            if bx_gt < 0 or by_gt < 0 or bx_gt >= bev_h or by_gt >= bev_w:
                continue
            
            objects_in_bev += 1
            
            # track_id 또는 instance_id 가져오기
            track_id = gt_obj.track_id if gt_obj.track_id is not None else gt_obj.instance_id
            if track_id is None:
                continue
            
            # 궤적 포인트 수집 (모든 과거와 미래 프레임)
            trajectory_points_bev = []
            trajectory_points_world = []
            
            # 모든 가용 프레임에서 궤적 수집 (제한 없음)
            # gt_frame_map에 있는 모든 프레임 탐색
            all_frame_indices = sorted(gt_frame_map.keys())
            
            for target_frame_idx in all_frame_indices:
                # 현재 프레임으로부터의 offset 계산
                offset = target_frame_idx - frame_idx
                
                # 대상 프레임의 GT 객체 가져오기
                target_gt_objects = gt_frame_map.get(target_frame_idx, [])
                
                # track_id로 매칭되는 GT 객체 찾기
                matched_gt = None
                for tgt_obj in target_gt_objects:
                    tgt_id = tgt_obj.track_id if tgt_obj.track_id is not None else tgt_obj.instance_id
                    if tgt_id == track_id:
                        matched_gt = tgt_obj
                        break
                
                if matched_gt is None or matched_gt.foot_uv is None:
                    continue
                
                # BEV 좌표로 변환
                u_tgt, v_tgt = matched_gt.foot_uv
                try:
                    bx_tgt, by_tgt = H.pixel_to_bev_warp(u_tgt, v_tgt)
                except Exception:
                    continue
                
                if bx_tgt < 0 or by_tgt < 0:
                    continue
                
                # BEV 픽셀을 world 미터 좌표로 변환
                try:
                    x_m, y_m = H.pixel_to_world(float(by_tgt), float(bx_tgt), flipped=True)
                except Exception:
                    x_m, y_m = None, None
                
                # BEV 궤적 포인트 추가
                trajectory_points_bev.append({
                    "frame_offset": offset,
                    "frame_id": target_frame_idx,
                    "bev_x": float(bx_tgt),
                    "bev_y": float(by_tgt)
                })
                
                # World 궤적 포인트 추가
                if x_m is not None and y_m is not None:
                    trajectory_points_world.append({
                        "frame_offset": offset,
                        "frame_id": target_frame_idx,
                        "world_x": float(x_m),
                        "world_y": float(y_m)
                    })
            
            # BEV 이미지에 궤적 그리기
            if len(trajectory_points_bev) > 1:
                bev_pts = [(pt["bev_x"], pt["bev_y"]) for pt in trajectory_points_bev]
                gt_bev_traj_img = vis.draw_polyline(
                    gt_bev_traj_img, 
                    int(track_id), 
                    bev_pts,
                    color=(0, 255, 0),  # Green for GT
                    thickness=2
                )
                
                # 현재 위치 강조 표시
                cv2.circle(gt_bev_traj_img, (int(by_gt), int(bx_gt)), 8, (0, 255, 255), -1)
                cv2.circle(gt_bev_traj_img, (int(by_gt), int(bx_gt)), 10, (0, 128, 128), 2)
                
            elif len(trajectory_points_bev) == 1:
                pt = trajectory_points_bev[0]
                gt_bev_traj_img = vis.draw_points(
                    gt_bev_traj_img, 
                    int(track_id), 
                    [(pt["bev_x"], pt["bev_y"])], 
                    radius=5
                )
            
            # JSON 데이터에 추가
            gt_trajectory_data["trajectories"].append({
                "track_id": int(track_id),
                "class": gt_obj.class_name,
                "current_foot_uv": [float(u_gt), float(v_gt)],
                "current_bev": [float(bx_gt), float(by_gt)],
                "trajectory_bev": trajectory_points_bev,
                "trajectory_world": trajectory_points_world,
                "num_points": len(trajectory_points_bev)
            })
            
            print(f"    Track {track_id}: {len(trajectory_points_bev)} trajectory points")
        
        # GT 궤적 시각화 이미지 저장
        cv2.imwrite(str(out_gt_bev_traj / f"gt_traj_{img_path.name}"), gt_bev_traj_img)
        
        # GT 궤적 JSON 저장
        json_path = out_gt_traj_json / f"gt_traj_frame_{frame_idx:06d}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(gt_trajectory_data, jf, indent=2, ensure_ascii=False)
        
        print(f"  Objects in BEV: {objects_in_bev}/{len(current_gt_objects)}")
        print(f"  Trajectories saved: {len(gt_trajectory_data['trajectories'])}")
        
        frame_idx += 1

    print(f"\n[Complete] Processed {frame_idx} frames")
    print(f"  Output directory: {out_root}")
    print(f"  - BEV trajectory images: {out_gt_bev_traj}")
    print(f"  - Trajectory JSON files: {out_gt_traj_json}")
    
    assert True


if __name__ == "__main__":
    test_gt_trajectory_visualization()

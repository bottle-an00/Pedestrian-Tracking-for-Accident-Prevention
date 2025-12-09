from pathlib import Path
import numpy as np
import cv2
import os
import json

from src.core.config import load_yaml
from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader
from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography
from src.tracking.tracker import ByteTracker
from src.bev.bev_transformer import BevTransformer, Detections_bev
from src.trajectory.trajectory_manager import TrajectoryBuffer
from src.visualization.overlay_2d import Visualizer
from src.trajectory.pedestrian_state_manager import PedestrianStateManager
from src.trajectory.bev_zone_risk_manager import PedestrianStateManager as BevZoneManager
from src.trajectory.ekf_manager import EKFManager
from src.trajectory.realtime_fixed_lag_smoothing import RealTimeEKFWithLagSmoothing
from src.trajectory.ekf_tracker_cv import EKFTrackerCV


def test_full_pipeline():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    # Allow overriding the dataset at runtime via environment variable (useful for batch runs)
    import os
    env_seq = os.environ.get('DATASET_ID')
    if env_seq:
        image_dir = Path('data') / 'raw' / env_seq / 'image0'
        gps_dir = Path('data') / 'raw' / env_seq / 'gps'
        out_root = Path('data') / 'raw' / env_seq / 'outputs' / 'full_pipeline_test'
    else:
        image_dir = Path(cfg[root_dir]["images"])
        gps_dir = Path(cfg[root_dir]["gps"])

        out_root = Path(cfg[root_dir]["outputs"]) / "full_pipeline_test"
    out_root.mkdir(parents=True, exist_ok=True)

    out_ekf_bev = out_root / "ekf_bev"
    out_ekf_img = out_root / "ekf_img"
    pred_out_root = out_root / "pred_json"

    for p in [out_ekf_bev, out_ekf_img]:
        p.mkdir(parents=True, exist_ok=True)

    frame_idx = 0

    calib_loader = CalibrationInfoLoader()
    intrinsics = calib_loader.load_camera_calibration(Path(cfg[root_dir]["calibration"]["calib_Camera"]))
    extrinsics = calib_loader.load_camera_extrinsics(Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"]))

    H = Homography(intrinsics=intrinsics, extrinsics=extrinsics)

    image_loader = ImageLoader()
    gps_loader = GpsImuLoader()
    vis = Visualizer()

    yolo_cfg = load_yaml("configs/detector/yolo_detector.yaml")

    tracker = ByteTracker(
        model_path=yolo_cfg["yolo_model_path"],
        conf_thres_config=yolo_cfg["conf_threshold"],
        target_class_names=yolo_cfg["target_classes"],
        imgsz=yolo_cfg.get("imgsz", 1280)
    )

    bev_conv = BevTransformer(homography=H)

    # Mode selection:
    # If True, use a standalone CV EKFManager to provide current+future predictions (no lag smoothing).
    # If False, use PedestrianStateManager with ekf_constructor (e.g. lag-wrapped EKF) as the single-truth filter.
    USE_EKF_MANAGER = True

    # If True, only visualize objects that are currently detected by the detector
    VIS_ONLY_DETECTIONS = True

    # Prepare PedestrianStateManager (kept available if we choose non-manager mode later)
    state_manager = PedestrianStateManager(
        obs_len=10,
        max_missing=5,
        ekf_constructor=lambda: RealTimeEKFWithLagSmoothing(
            EKFTrackerCV(initial_pos=(0.0, 0.0), initial_vel=(0.0, 0.0), dt=0.1, sigma_meas=0.4, debug=False),
            lag_size=5,
            future_horizon=20,
            default_dt=0.1,
        ),
        dt=0.1
    )

    # bookkeeping for logging
    LAG_SWITCH_STEPS = 20
    switched_tracks = set()

    # If requested, instantiate a standalone CV EKFManager for initial usage
    ekf_manager = None
    if USE_EKF_MANAGER:
        ekf_manager = EKFManager()
        # use EKFManager default dt for timestamping/prediction if available
        dt_for_timestamps = getattr(ekf_manager, 'default_dt', 0.1)
    else:
        dt_for_timestamps = getattr(state_manager, 'dt', 0.1)

    #load DataSet
    img_iter = image_loader.iter_imgs_cv2(image_dir)
    gps_iter = gps_loader.iter_data(gps_dir)

    bev_risk_manager = BevZoneManager()

    # prepare GT frame map for BEV visualization (data/gt/<seq>/image0)
    try:
        from src.evaluation.gt_loader import GTLoader
        seq_id = Path(image_dir).parent.name
        gt_dir = Path('data') / 'gt' / seq_id / Path(image_dir).name
        gt_loader = GTLoader(target_classes=['pedestrian'])
        gt_frame_map = {}
        if gt_dir.exists():
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
                except Exception:
                    objs = []
                gt_frame_map[fidx] = objs
    except Exception:
        gt_frame_map = {}

    # Main loop
    for (img_path, image), (gps_path, gps_data) in zip(img_iter, gps_iter):
        matched_count = 0
        print(f"[Full Pipeline] Processing {img_path.name}")

        detections = tracker.process(image)

        bev_img = H.warp(image)
        foot_bevs = bev_conv.foot_uv_to_foot_bev(detections)

        foot_bevs_m = []
        for fb in foot_bevs:
            bx, by = fb.foot_bev
            x_m, y_m = H.pixel_to_world(float(by), float(bx), flipped=True)
            foot_bevs_m.append(Detections_bev(id=fb.id, foot_bev=(x_m, y_m)))

        # vehicale_bev setting
        vehicle_bev = bev_risk_manager.compute_vehicle_bev(image,H)
        bev_risk_manager.set_zones_from_gps(gps_data, vehicle_bev=vehicle_bev)

        # We'll update BOTH EKF providers each frame when available:
        #  - ekf_manager (CV-only)
        #  - state_manager (may wrap a lag smoother)
        # Collect per-source outputs and also compute selected outputs (prefer LAG when available).

        ekf_now_cv = {}
        ekf_future_cv = {}
        ekf_now_lag = {}
        ekf_future_lag = {}

        # Timestamp for ekf_manager updates
        timestamp = float(frame_idx) * float(dt_for_timestamps)

        # Update EKFManager (CV) if present
        if ekf_manager is not None:
            for fbm in foot_bevs_m:
                try:
                    ekf_manager.update(int(fbm.id), np.asarray(fbm.foot_bev, dtype=float), timestamp)
                except Exception:
                    continue

            for tid in ekf_manager.get_all_track_ids():
                state = ekf_manager.get_state(tid)
                ekf_now_cv[tid] = state[:2].tolist() if state is not None else None
                fut = ekf_manager.predict_future(tid, steps=20, dt=dt_for_timestamps)
                ekf_future_cv[tid] = [p.tolist() for p in fut] if fut is not None else None

        # Update PedestrianStateManager (lag-wrapped EKF) always — it will create trackers as needed
        detections_for_mgr = [(fbm.id, fbm.foot_bev) for fbm in foot_bevs_m]
        state_manager.update(detections_for_mgr, frame_idx)

        for tid, ts in state_manager.active_tracks.items():
            # current filtered position from lag-wrapped ekf
            cur_pos = None
            try:
                if hasattr(ts.ekf, 'get_current_filtered_position'):
                    cur_pos = ts.ekf.get_current_filtered_position()
                elif hasattr(ts.ekf, 'get_current_filtered_state'):
                    s = ts.ekf.get_current_filtered_state()
                    cur_pos = (float(s[0]), float(s[1]))
                elif hasattr(ts.ekf, 'get_position'):
                    cur_pos = ts.ekf.get_position()
            except Exception:
                cur_pos = None

            ekf_now_lag[tid] = cur_pos

            # future prediction: prefer smoothed predictor when available
            fut = None
            try:
                if hasattr(ts.ekf, 'predict_future_with_smoothing'):
                    fut = ts.ekf.predict_future_with_smoothing(horizon_steps=20, dt=getattr(ts.ekf, 'default_dt', state_manager.dt))
            except Exception:
                fut = None

            if fut is None:
                try:
                    if hasattr(ts.ekf, 'predict_future'):
                        try:
                            fut = ts.ekf.predict_future(steps=20, dt=getattr(ts.ekf, 'default_dt', state_manager.dt))
                        except TypeError:
                            fut = ts.ekf.predict_future(20)
                except Exception:
                    fut = None

            ekf_future_lag[tid] = [p.tolist() if hasattr(p, 'tolist') else p for p in fut] if fut is not None else None

        # Choose selected outputs for downstream processing: prefer lag when available and observed enough
        ekf_now = {}
        ekf_future = {}
        ekf_source_map = {}

        # Union of track ids seen by either source
        all_ids = set(list(ekf_now_cv.keys()) + list(ekf_now_lag.keys()))
        for tid in all_ids:
            # Prefer lag if it exists and obs_count >= threshold
            use_lag = False
            ts = state_manager.active_tracks.get(tid, None)
            if ts is not None and getattr(ts, 'obs_count', 0) >= LAG_SWITCH_STEPS and ekf_now_lag.get(tid) is not None:
                use_lag = True

            if use_lag:
                ekf_now[tid] = ekf_now_lag.get(tid)
                ekf_future[tid] = ekf_future_lag.get(tid)
                ekf_source_map[tid] = 'LAG'
            else:
                ekf_now[tid] = ekf_now_cv.get(tid, ekf_now_lag.get(tid))
                ekf_future[tid] = ekf_future_cv.get(tid, ekf_future_lag.get(tid))
                # mark source
                if tid in ekf_now_cv and ekf_now_cv.get(tid) is not None:
                    ekf_source_map[tid] = 'CV'
                elif tid in ekf_now_lag and ekf_now_lag.get(tid) is not None:
                    ekf_source_map[tid] = 'LAG'
                else:
                    ekf_source_map[tid] = 'NONE'

        ekf_bev_img = bev_img.copy()
        ekf_img = image.copy()

        # Per-frame EKF source log: which EKF provided the current estimate per track
        ekf_source_map = {}

        for tid, ts in state_manager.active_tracks.items():

            if getattr(ts, 'obs_count', 0) >= LAG_SWITCH_STEPS:
                ekf_source_map[tid] = 'LAG'
            else:
                ekf_source_map[tid] = 'CV'

        print(f"[EKFSource][frame={frame_idx}] {ekf_source_map}")

        # which track ids are present in current detections
        detected_ids = set(int(fbm.id) for fbm in foot_bevs_m)

        for tid, xy in ekf_now.items():
            if xy is None:
                continue
            # optionally skip visualization for tracks that are not currently detected
            if VIS_ONLY_DETECTIONS and int(tid) not in detected_ids:
                continue
            lx, ly = float(xy[0]), float(xy[1])
            u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
            ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [(v_bev, u_bev)], radius=6)
            u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
            if u_img != -1 and v_img != -1:
                ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf", "score": 1.0, "foot_uv": (u_img, v_img)}])

        for tid, future in ekf_future.items():
            if future is None:
                continue
            # optionally skip visualization for tracks that are not currently detected
            if VIS_ONLY_DETECTIONS and int(tid) not in detected_ids:
                continue
            bev_future_pts = []
            img_uvs = []
            for p in future:
                lx, ly = float(p[0]), float(p[1])
                u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                bev_future_pts.append((v_bev, u_bev))
                u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                if u_img != -1 and v_img != -1:
                    img_uvs.append((u_img, v_img))

            if len(bev_future_pts) == 1:
                ekf_bev_img = vis.draw_points(ekf_bev_img, tid, bev_future_pts, radius=4)
            elif bev_future_pts:
                ekf_bev_img = vis.draw_polyline(ekf_bev_img, tid, bev_future_pts, color=(255, 0, 0), thickness=2)

            if img_uvs:
                ekf_img = vis.draw_on_img(ekf_img, [{"bbox": (0,0,0,0), "id": tid, "class": "ekf_future", "score": 1.0, "foot_uv": uv} for uv in img_uvs])

        # Update risk states
        obj_states = {}
        for tid, future in ekf_future.items():
            state = "SAFE"
            if future and len(future) > 0:
                updated_state = bev_risk_manager.update_state(tid, future)
                state = updated_state
            obj_states[tid] = state

        frame_idx += 1

        # Visualizations
        gt_vis = ekf_bev_img.copy()
        gt_vis = bev_risk_manager.draw_zones_on_bev(gt_vis, alpha=0.25)
        ekf_img = bev_risk_manager.draw_zones_on_image(ekf_img, H, alpha=0.25)
        ekf_img = bev_risk_manager.draw_detections_on_image(ekf_img, detections)

        # Additionally: match current detections to GT and draw matched GT future positions
        # onto a copy of the EKF BEV image. Also label which detections matched GT.
        gt_steps = 20  # number of future frames to visualize
        objs_now = gt_frame_map.get(frame_idx - 1, [])

        # build detection BEV points for matching (use the same foot_bevs produced earlier)
        det_bev_pts = []
        for fb in foot_bevs:
            try:
                bx, by = fb.foot_bev
            except Exception:
                continue
            det_bev_pts.append({"id": int(fb.id), "bx": float(bx), "by": float(by)})

        MATCH_PX = 40.0

        for det in det_bev_pts:
            best = None
            bd = float("inf")
            for gobj in objs_now:
                if gobj.foot_uv is None:
                    continue
                u_gt, v_gt = gobj.foot_uv
                try:
                    bx_gt, by_gt = H.pixel_to_bev_warp(u_gt, v_gt)
                except Exception:
                    continue
                if bx_gt < 0 or by_gt < 0:
                    continue
                dx = det["bx"] - float(bx_gt)
                dy = det["by"] - float(by_gt)
                d = (dx*dx + dy*dy) ** 0.5
                if d < bd:
                    bd = d
                    best = (gobj, bx_gt, by_gt)

            if best is not None and bd <= MATCH_PX:
                gobj, bx_gt, by_gt = best
                gid = gobj.track_id if gobj.track_id is not None else gobj.instance_id
                if gid is None:
                    continue

                future_bev = []
                for t in range(1, gt_steps+1):
                    fidx = (frame_idx - 1) + t
                    future_objs = gt_frame_map.get(fidx, [])
                    for fo in future_objs:
                        if fo.track_id == gid:
                            if fo.foot_uv is None:
                                continue
                            try:
                                bx_f, by_f = H.pixel_to_bev_warp(fo.foot_uv[0], fo.foot_uv[1])
                                future_bev.append((bx_f, by_f))
                            except Exception:
                                continue
                            break

                if future_bev:
                    gt_vis = vis.draw_polyline(gt_vis, gid, future_bev, color=(0,255,0), thickness=2)

        cv2.imwrite(str(out_ekf_bev / f"ekf_bev_gt_{img_path.name}"), gt_vis)
        cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

        # --- Minimal prediction JSON export for ADE/FDE evaluator ---
        # Create per-sequence predictions directory
        pred_out_root.mkdir(parents=True, exist_ok=True)

        # Build prediction JSON structure expected by evaluator
        # Each frame file: frame_<frame_idx>.json with {'objects':[{'id':id,'pos':[x,y],'future':[[x1,y1],...]}, ...]}
        pred_objs = []
        T_required = 20
        for tid, fut in ekf_future.items():
            cur = ekf_now.get(tid)
            if cur is None:
                continue
            # ensure future is a list of length T_required; if shorter, pad with last available
            future_list = []
            if fut is None:
                future_list = []
            else:
                for p in fut:
                    try:
                        future_list.append([float(p[0]), float(p[1])])
                    except Exception:
                        pass
            if len(future_list) < T_required:
                if future_list:
                    last = future_list[-1]
                    while len(future_list) < T_required:
                        future_list.append(last)
                else:
                    # replicate current position
                    last = [float(cur[0]), float(cur[1])]
                    future_list = [last for _ in range(T_required)]
            # trim to T_required
            future_list = future_list[:T_required]

            pred_objs.append({'id': int(tid), 'pos': [float(cur[0]), float(cur[1])], 'future': future_list})

        pred_frame_path = pred_out_root / f"frame_{frame_idx}.json"
        try:
            with open(pred_frame_path, 'w') as pf:
                json.dump({'objects': pred_objs}, pf, indent=2)
        except Exception:
            pass

    # End of sequence: print summary of switched tracks
    print(f"[SwitchSummary] total_switched={len(switched_tracks)}, ids={sorted(list(switched_tracks))}")


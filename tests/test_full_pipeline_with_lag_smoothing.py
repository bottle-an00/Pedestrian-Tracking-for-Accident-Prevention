from pathlib import Path
import numpy as np
import cv2
import os

from src.core.config import load_yaml

from src.io.image_loader import ImageLoader
from src.io.gps_loader import GpsImuLoader

from src.calibration.load_calibration_info import CalibrationInfoLoader
from src.calibration.homography import Homography

from src.tracking.tracker import ByteTracker

from src.bev.bev_transformer import BevTransformer, Detections_bev
import json
# ego-motion compensation intentionally removed for this test
from src.trajectory.trajectory_manager import TrajectoryBuffer

from src.trajectory.ekf_manager import EKFManager
from src.visualization.overlay_2d import Visualizer
from src.trajectory.pedestrian_state_manager import PedestrianStateManager
from src.trajectory.ekf_tracker import EKFTracker

# Import the new smoother wrapper
from src.trajectory.realtime_fixed_lag_smoothing import RealTimeEKFWithLagSmoothing

from src.trajectory.ekf_tracker_cv import EKFTrackerCV

def test_full_pipeline_with_lag_smoothing():
    """
    Full pipeline test similar to `test_full_pipeline.py` but uses
    RealTimeEKFWithLagSmoothing to produce smoothed-origin future predictions.

    Notes:
    - The smoother wraps an EKFTrackerCV instance; real-time calls use
      smoother.update(z, dt) which performs forward-only filtering.
    - Future predictions use smoother.predict_future_with_smoothing(...),
      which runs fixed-lag RTS smoothing over recent history and then
      rollouts a CV trajectory from the smoothed current state.
    - All other pipeline logic remains unchanged.
    """

    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    image_dir = Path(cfg[root_dir]["images"])
    gps_dir = Path(cfg[root_dir]["gps"])
    out_root = Path(cfg[root_dir]["outputs"]) / "full_pipeline_with_lag_smoothing"
    out_root.mkdir(parents=True, exist_ok=True)

    out_bev_traj = out_root / "bev_trajectory"
    out_bev_result = out_root / "bev_result"
    out_img_traj = out_root / "img_trajectory"
    out_tracking = out_root / "tracking"
    out_detection = out_root / "detection"
    out_ekf_bev = out_root / "ekf_bev"
    out_ekf_img = out_root / "ekf_img"
    out_bev_risk = out_root / "bev_risk"
    out_bev_debug = out_root / "bev_to_world_debug"

    for p in [out_bev_traj, out_bev_result, out_img_traj, out_tracking, out_ekf_bev, out_ekf_img, out_bev_risk]:
        p.mkdir(parents=True, exist_ok=True)

    # detection outputs should mirror `src/app/main.py` behavior
    out_detection.mkdir(parents=True, exist_ok=True)

    out_bev_debug.mkdir(parents=True, exist_ok=True)

    frame_idx = 0

    # recorder for per-id detections: { id: [ {frame, time, x_m, y_m}, ... ] }
    detections_by_id = {}

    calib_loader = CalibrationInfoLoader()

    intrinsics = calib_loader.load_camera_calibration(
        Path(cfg[root_dir]["calibration"]["calib_Camera"])
    )
    extrinsics = calib_loader.load_camera_extrinsics(
        Path(cfg[root_dir]["calibration"]["calib_LiDAR_Camera"])
    )

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
    trajectory_buffer = TrajectoryBuffer(max_length=100)

    eval_mode = True

    # Use the improved Constant-Velocity EKF tracker (EKFTrackerCV) via EKFManager-compatible constructor

    # Wrap each EKFTrackerCV instance with RealTimeEKFWithLagSmoothing via the ekf_constructor
    state_manager = PedestrianStateManager(
        obs_len=5,
        max_missing=5,
        ekf_constructor=lambda: RealTimeEKFWithLagSmoothing(
            EKFTrackerCV(initial_pos=(0.0, 0.0), initial_vel=(0.0, 0.0), dt=0.1, sigma_meas=0.4, debug=False),
            lag_size=5,
            future_horizon=20,
            default_dt=0.1,
        )
    )

    # prepare GT frame map for BEV visualization (data/gt/<seq>/image0)
    from src.evaluation.gt_loader import GTLoader
    seq_id = Path(image_dir).parent.name
    gt_dir = Path('data') / 'gt' / seq_id / Path(image_dir).name
    gt_loader = GTLoader(target_classes=['pedestrian'])
    gt_frame_map = {}
    if gt_dir.exists():
        for p in sorted(gt_dir.glob('*.json')):
            stem = p.stem
            # Prefer the filename suffix as the frame index. Example: '13_053_000' -> '000' -> 0
            fidx = None
            parts = stem.split('_')
            if parts and parts[-1].lstrip('0').isdigit():
                try:
                    fidx = int(parts[-1])
                except Exception:
                    fidx = None
            # Fallback: if suffix parsing failed, try parsing the whole stem (numeric-only stems)
            if fidx is None:
                try:
                    fidx = int(stem)
                except Exception:
                    # unable to determine frame index for this file; skip
                    continue
            try:
                data = gt_loader.load_image_label(p)
                objs, _ = gt_loader.parse_image_label(data)
            except Exception:
                objs = []
            gt_frame_map[fidx] = objs

    img_iter = image_loader.iter_imgs_cv2(image_dir)
    gps_iter = gps_loader.iter_data(gps_dir)

    for (img_path, image), (gps_path, gps_data) in zip(img_iter, gps_iter):
        # per-frame match counter
        matched_count = 0
        print(f"[Full Pipeline] Processing {img_path.name}")

        # YOLO tracking
        detections = tracker.process(image)
        img_tracking = vis.draw_on_img_with_keypoints(
            image.copy(), detections, seq_name=Path(image_dir).parent.name
        )

        # Save to outputs/detection/ with same filename pattern used in main
        cv2.imwrite(str(out_detection / f"det_{img_path.name}"), img_tracking)

        # Image to BEV
        bev_img = H.warp(image)
        foot_bevs = bev_conv.foot_uv_to_foot_bev(detections)

        # Convert BEV pixel indices (bx,by) -> world meters (x_m, y_m) BEFORE ego compensation / EKF
        foot_bevs_m = []
        for fb in foot_bevs:
            bx, by = fb.foot_bev
            # note: pixel_to_world expects (u, v) -> (x_m, y_m) where u=col, v=row
            x_m, y_m = H.pixel_to_world(float(by), float(bx), flipped=True)
            foot_bevs_m.append(Detections_bev(id=fb.id, foot_bev=(x_m, y_m)))

        bev_result = vis.draw_on_BEV(
            bev_img.copy(), 0, [p.foot_bev for p in foot_bevs]
        )
        cv2.imwrite(str(out_bev_result / f"bev_{img_path.name}"), bev_result)


        # Trajectory update WITHOUT ego-motion compensation: keep local vehicle-frame coords
        compensated = foot_bevs_m
        trajectory_buffer.add(compensated, gps_data.time)

        # Record each detection's local position (vehicle-frame meters) by id
        for fbm in foot_bevs_m:
            try:
                tid = int(fbm.id)
                x_m, y_m = float(fbm.foot_bev[0]), float(fbm.foot_bev[1])
            except Exception:
                continue
            detections_by_id.setdefault(tid, []).append({
                "frame": int(frame_idx),
                "time": float(gps_data.time),
                "x_m": x_m,
                "y_m": y_m
            })

        # Update PedestrianStateManager with local (vehicle-frame) detections
        # Note: use the local world coords (foot_bevs_m) before ego compensation
        detections_for_mgr = [(fbm.id, fbm.foot_bev) for fbm in foot_bevs_m]
        state_manager.update(detections_for_mgr, frame_idx)

        # If evaluation mode, generate and save future predictions (T=20)
        if eval_mode:
            # Only pre-fill futures when the smoother's lag window is filled.
            # This avoids filling ts.future_traj early using obs_len-based helpers.
            for track_id, ts in state_manager.active_tracks.items():
                try:
                    hist_len = 0
                    try:
                        hist_len = len(getattr(ts.ekf, '_x_hist', []))
                    except Exception:
                        hist_len = 0
                    lag_size = getattr(ts.ekf, 'lag_size', 0)
                    use_smoothing = hist_len >= (lag_size + 1)

                    if use_smoothing and hasattr(ts.ekf, 'predict_future_with_smoothing'):
                        try:
                            ekf_under = getattr(ts.ekf, 'ekf', ts.ekf)
                            dt_local = getattr(ts.ekf, 'default_dt', state_manager.dt)
                            # fut = ts.ekf.predict_future_with_smoothing(horizon_steps=20, dt=dt_local)
                            fut = ekf_under.predict_future(steps=20, dt=getattr(ekf_under, 'default_dt', state_manager.dt))
                            if fut is not None:
                                ts.future_traj = [(float(p[0]), float(p[1])) for p in fut]
                            else:
                                ts.future_traj = None
                        except Exception:
                            ts.future_traj = None
                    else:
                        # fallback: use underlying EKF's forward-only predict if available
                        try:
                            ekf_under = getattr(ts.ekf, 'ekf', ts.ekf)
                            if hasattr(ekf_under, 'predict_future'):
                                try:
                                    fut = ekf_under.predict_future(steps=20, dt=getattr(ekf_under, 'default_dt', state_manager.dt))
                                except TypeError:
                                    fut = ekf_under.predict_future(20)
                                ts.future_traj = [(float(p[0]), float(p[1])) for p in fut] if fut is not None else None
                            else:
                                ts.future_traj = None
                        except Exception:
                            ts.future_traj = None
                except Exception:
                    ts.future_traj = None

            # saved positions are expected to already be local BEV meters (m)
            state_manager.save_predictions_json(frame_idx, str(out_root), eval_mode=True)


    # EKFManager disabled: use per-track smoothing EKF via state_manager instead
    # ekf_now = {}
    # (we intentionally skip ekf_manager.update calls)

        ekf_bev_img = bev_img.copy()
        ekf_img = image.copy()

        # # Visualize EKF current predictions: convert world->BEV pixel for drawing
        # for tid, xy in ekf_now.items():
        #     if xy is None:
        #         continue

        #     # xy is in local vehicle-frame meters (no ego compensation)
        #     lx, ly = float(xy[0]), float(xy[1])

        #     # local -> BEV image pixel (u: col, v: row)
        #     u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)

        #     # draw on BEV image (vis.draw_points expects (row, col) pairs)
        #     ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [(v_bev, u_bev)], radius=6)

        #     # convert BEV pixel -> original image pixel for overlay (from local BEV)
        #     u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
        #     if u_img != -1 and v_img != -1:
        #         ekf_img = vis.draw_on_img(
        #             ekf_img,
        #             [{"bbox": (0,0,0,0), "id": tid,
        #               "class": "ekf", "score": 1.0, "foot_uv": (u_img, v_img)}]
        #         )

        # --- Future trajectory visualization: use smoothed futures from state_manager.active_tracks
        horizon_steps = 20
        for track_id, ts in state_manager.active_tracks.items():
            # Determine or compute future_traj for this track
            try:
                # history length (wrapper stores history in _x_hist)
                hist_len = 0
                try:
                    hist_len = len(getattr(ts.ekf, '_x_hist', []))
                except Exception:
                    hist_len = 0
                lag_size = getattr(ts.ekf, 'lag_size', 0)
                use_smoothing = hist_len >= (lag_size + 1)
                print(f"[Smoothing] Track {track_id}: hist={hist_len}, using_smoothing={use_smoothing}")

                # only attempt to compute a new future if smoothing is available (lag filled)
                if ts.future_traj is None and use_smoothing:
                    # compute now
                    # use the track ekf's default dt where possible (keeps consistency with prior updates)
                    dt_local = getattr(ts.ekf, 'default_dt', state_manager.dt)
                    if use_smoothing and hasattr(ts.ekf, 'predict_future_with_smoothing'):
                        fut = ts.ekf.predict_future_with_smoothing(horizon_steps, dt=dt_local)
                    else:
                        # fallback to EKF's forward-only predictor if available
                        ekf_under = getattr(ts.ekf, 'ekf', None)
                        if ekf_under is not None and hasattr(ekf_under, 'predict_future'):
                            try:
                                fut = ekf_under.predict_future(steps=horizon_steps, dt=getattr(ekf_under, 'default_dt', state_manager.dt))
                            except TypeError:
                                # some implementations use (horizon_steps) positional
                                fut = ekf_under.predict_future(horizon_steps)
                        else:
                            # if smoothing isn't available and no underlying EKF exists, set None
                            fut = None

                    if fut is not None:
                        ts.future_traj = [(float(p[0]), float(p[1])) for p in fut]
                    else:
                        ts.future_traj = None
            except Exception:
                ts.future_traj = None

            # draw the smoothed/fallback future if present
            fut = ts.future_traj
            print(f"[Future Visualization] Track {track_id}: fut={fut}")
            if not fut:
                continue

            bev_future_pts = []
            img_uvs = []
            for p in fut:
                lx, ly = float(p[0]), float(p[1])
                u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                bev_future_pts.append((v_bev, u_bev))
                u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                if u_img != -1 and v_img != -1:
                    img_uvs.append((u_img, v_img))

            if len(bev_future_pts) == 1:
                ekf_bev_img = vis.draw_points(ekf_bev_img, track_id, bev_future_pts, radius=4)
            elif bev_future_pts:
                ekf_bev_img = vis.draw_polyline(
                    ekf_bev_img, track_id, bev_future_pts,
                    color=(255, 0, 0), thickness=2
                )

            if img_uvs:
                ekf_img = vis.draw_on_img(
                    ekf_img,
                    [{"bbox": (0,0,0,0), "id": track_id, "class": "ekf_future",
                      "score": 1.0, "foot_uv": uv}
                     for uv in img_uvs]
                )

        # --- Debug output: save BEV pixel -> world -> pixel mappings per frame
        debug_entry = {
            "frame": int(frame_idx),
            "detections_pixel": [],
            "detections_world": [],
            "ekf_pred_world": [],
            "ekf_pred_pixel": []
        }

        for fb in foot_bevs:
            bx, by = fb.foot_bev
            debug_entry["detections_pixel"].append({"id": int(fb.id), "u": int(by), "v": int(bx)})

        for fbm in foot_bevs_m:
            x_m, y_m = fbm.foot_bev
            debug_entry["detections_world"].append({"id": int(fbm.id), "x_m": float(x_m), "y_m": float(y_m)})

        # Note: ekf_now loop removed because EKFManager is disabled in this test.

        dbg_path = out_bev_debug / f"frame_{frame_idx:06d}.json"
        with open(dbg_path, "w") as jf:
            json.dump(debug_entry, jf, indent=2)

        frame_idx += 1

        # Additionally: match current detections to GT and draw matched GT future positions
        # onto a copy of the EKF BEV image. Also label which detections matched GT.
        gt_vis = ekf_bev_img.copy()
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

        # matching threshold in BEV pixels (tune if needed)
        MATCH_PX = 40.0

        # For each detection, try to find a GT in the same frame within threshold
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

            # if a nearby GT was found, draw its future trajectory on gt_vis and label detection
            if best is not None and bd <= MATCH_PX:
                gobj, bx_gt, by_gt = best
                gid = gobj.track_id if gobj.track_id is not None else gobj.instance_id
                if gid is None:
                    continue

                # collect future BEV points for this GT by looking up future frames
                future_bev = []
                for t in range(1, gt_steps+1):
                    fidx = (frame_idx - 1) + t
                    future_objs = gt_frame_map.get(fidx, [])
                    # find same track in future frame
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

                # draw GT future
                if future_bev:
                    gt_vis = vis.draw_polyline(gt_vis, gid, future_bev, color=(0,255,0), thickness=2)

            cv2.imwrite(str(out_ekf_bev / f"ekf_bev_gt_{img_path.name}"), gt_vis)
            cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

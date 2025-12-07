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
from src.trajectory.bev_zone_risk_manager import PedestrianStateManager as BevZoneManager
from src.trajectory.ekf_tracker import EKFTracker


def test_full_pipeline():
    cfg = load_yaml("configs/system.yaml")
    root_dir = "dataset_dir"

    image_dir = Path(cfg[root_dir]["images"])
    gps_dir = Path(cfg[root_dir]["gps"])
    from src.trajectory.ekf_tracker_cv import EKFTrackerCV
    out_root = Path(cfg[root_dir]["outputs"]) / "full_pipeline"
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
    ekf_manager = EKFManager()
    # PedestrianStateManager holds per-track EKF state and history for forecasting.
    # Set eval_mode=True when you want per-frame prediction JSONs to be saved.
    eval_mode = True

    # Use the Constant-Velocity EKF tracker via EKFManager-compatible constructor
    from src.trajectory.ekf_tracker_cv import EKFTrackerCV

    state_manager = PedestrianStateManager(
        obs_len=10,
        max_missing=5,
        ekf_constructor=lambda: EKFTrackerCV(initial_pos=(0.0, 0.0), initial_vel=(0.0, 0.0), dt=0.1, sigma_meas=0.4, debug=False)
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

    # BEV-based risk zone manager (separate from EKF state_manager)
    bev_risk_manager = BevZoneManager()

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
        # keep a copy of the detection overlay to blend into ekf_img later
        detection_overlay = img_tracking.copy()

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

        # Risk zone overlay
        speed = np.sqrt(gps_data.vel_x**2 + gps_data.vel_y**2)
        front_m = float(np.clip(speed, 2.0, 10.0))

        img_h, img_w = image.shape[:2]
        u = float(img_w / 2.0)
        v = float(img_h - 1)
        vehicle_bev = H.pixel_to_bev_warp(u, v)

        # Use bev_zone_risk_manager to set zones and draw on BEV
        bev_risk_manager.set_zones_from_gps(gps_data, vehicle_bev=vehicle_bev)
        bev_risk_img = bev_risk_manager.draw_zones_on_bev(bev_img.copy(), alpha=0.25)
        cv2.imwrite(str(out_bev_risk / f"bev_risk_{img_path.name}"), bev_risk_img)

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
            state_manager.generate_future_predictions(T=20)
            # saved positions are expected to already be local BEV meters (m)
            state_manager.save_predictions_json(frame_idx, str(out_root), eval_mode=True)

        # Trajectory to BEV image
        bev_traj_img = bev_img.copy()
        for tid, traj in trajectory_buffer.get_all().items():
            # traj already in local vehicle-frame meters (no ego compensation)
            bev_traj_img = vis.draw_on_BEV(
                bev_traj_img,
                tid,
                [p.foot_bev for p in traj]
            )

        cv2.imwrite(str(out_bev_traj / f"bev_traj_{img_path.name}"), bev_traj_img)

        # EKF update & prediction
        ekf_now = {}
        ekf_future = {}

        for d in compensated:
            tid = d.id
            # d.foot_bev is expected to be world-meter coords (after compensate)
            xy = ekf_manager.update(tid, d.foot_bev, gps_data.time)
            ekf_now[tid] = xy
            ekf_future[tid] = ekf_manager.predict_future(tid, steps=20)

        ekf_bev_img = bev_img.copy()
        ekf_img = image.copy()
        # if detection overlay exists, blend it into ekf_img for visualization
        # blend with alpha to preserve EKF text and other overlays

        # Visualize EKF current predictions: convert world->BEV pixel for drawing
        for tid, xy in ekf_now.items():
            if xy is None:
                continue

            # xy is in local vehicle-frame meters (no ego compensation)
            lx, ly = float(xy[0]), float(xy[1])

            # local -> BEV image pixel (u: col, v: row)
            u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)

            # draw on BEV image (vis.draw_points expects (row, col) pairs)
            ekf_bev_img = vis.draw_points(ekf_bev_img, tid, [(v_bev, u_bev)], radius=6)

            # convert BEV pixel -> original image pixel for overlay (from local BEV)
            u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
            if u_img != -1 and v_img != -1:
                ekf_img = vis.draw_on_img(
                    ekf_img,
                    [{"bbox": (0,0,0,0), "id": tid,
                      "class": "ekf", "score": 1.0, "foot_uv": (u_img, v_img)}]
                )

        for tid, future in ekf_future.items():
            if future is None:
                continue
            # future: list of world coords (global frame) -> convert each step to vehicle-local
            # and draw as BEV / image points
            bev_future_pts = []
            img_uvs = []
            for p in future:
                # p is in local vehicle-frame meters (no ego compensation)
                lx, ly = float(p[0]), float(p[1])

                # local -> BEV image pixel (u: col, v: row)
                u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
                bev_future_pts.append((v_bev, u_bev))

                # BEV pixel -> original image pixel for overlay
                u_img, v_img = H.bev_to_pixel(v_bev, u_bev)
                if u_img != -1 and v_img != -1:
                    img_uvs.append((u_img, v_img))

            if len(bev_future_pts) == 1:
                ekf_bev_img = vis.draw_points(ekf_bev_img, tid, bev_future_pts, radius=4)
            elif bev_future_pts:
                ekf_bev_img = vis.draw_polyline(
                    ekf_bev_img, tid, bev_future_pts,
                    color=(255, 0, 0), thickness=2
                )

            if img_uvs:
                ekf_img = vis.draw_on_img(
                    ekf_img,
                    [{"bbox": (0,0,0,0), "id": tid, "class": "ekf_future",
                      "score": 1.0, "foot_uv": uv}
                     for uv in img_uvs]
                )
        # Determine per-object risk state using bev_risk_manager and EKF future positions
        # For each track, sample 4 points across the EKF future prediction and pick the worst
        # severity among the samples (RISK > CAUTION > SAFE). Use the sample that produced
        # the worst severity when calling bev_risk_manager.update_state(...) so hysteresis
        # is applied based on the most concerning future point.
        obj_states = {}
        for tid, future in ekf_future.items():
            state = "SAFE"
                # Delegate sampling + worst-case selection to bev_risk_manager.update_state
            if future and len(future) > 0:
                updated_state = bev_risk_manager.update_state(tid, future)
                state = updated_state

            obj_states[tid] = state

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

        for tid, xy in ekf_now.items():
            if xy is None:
                continue

            # global EKF prediction
            x_m, y_m = float(xy[0]), float(xy[1])
            debug_entry["ekf_pred_world"].append({"id": int(tid), "x_m": x_m, "y_m": y_m})

            # xy is already local (vehicle-frame) meters
            lx, ly = x_m, y_m
            debug_entry["ekf_pred_local"] = debug_entry.get("ekf_pred_local", [])
            debug_entry["ekf_pred_local"].append({"id": int(tid), "x_m": lx, "y_m": ly})

            u_bev, v_bev = H.world_to_bev_img_pixel(lx, ly, flipped=True)
            debug_entry["ekf_pred_pixel"].append({"id": int(tid), "u": int(u_bev), "v": int(v_bev)})

        dbg_path = out_bev_debug / f"frame_{frame_idx:06d}.json"
        with open(dbg_path, "w") as jf:
            json.dump(debug_entry, jf, indent=2)

        frame_idx += 1

        # Additionally: match current detections to GT and draw matched GT future positions
        # onto a copy of the EKF BEV image. Also label which detections matched GT.
        gt_vis = ekf_bev_img.copy()
        # Draw BEV risk overlay on the GT/BEV visualization
        try:
            gt_vis = bev_risk_manager.draw_zones_on_bev(gt_vis, alpha=0.25)
        except Exception:
            # fallback: leave gt_vis unchanged on failure
            pass
        # Also draw BEV risk overlay projected onto the original EKF image
        try:
            ekf_img = bev_risk_manager.draw_zones_on_image(ekf_img, H, alpha=0.25)
        except Exception:
            pass
        # Draw detection bboxes on ekf_img colored by state (SAFE/CAUTION/RISK)
        try:
            ekf_img = bev_risk_manager.draw_detections_on_image(ekf_img, detections)
        except Exception:
            pass
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
                future_pts = []
                for s in range(gt_steps):
                    fidx = frame_idx - 1 + s
                    objs_f = gt_frame_map.get(fidx, [])
                    match = None
                    for of in objs_f:
                        otid = of.track_id if of.track_id is not None else of.instance_id
                        if otid == gid:
                            match = of
                            break
                    if match is None or match.foot_uv is None:
                        continue
                    u, v = match.foot_uv
                    try:
                        bx_f, by_f = H.pixel_to_bev_warp(u, v)
                    except Exception:
                        bx_f, by_f = -1, -1
                    if bx_f < 0 or by_f < 0:
                        continue
                    future_pts.append((bx_f, by_f))

                # draw GT future on gt_vis
                if future_pts:
                    if len(future_pts) == 1:
                        gt_vis = vis.draw_points(gt_vis, int(gid), [(future_pts[0][0], future_pts[0][1])], radius=4)
                    else:
                        gt_vis = vis.draw_polyline(gt_vis, int(gid), future_pts, color=(0, 255, 0), thickness=2)

                # print matched GT future coordinates (BEV pixels)
                try:
                    print(f"[GT_VIS] frame={frame_idx-1} det_id={det['id']} gid={gid} future_bev={future_pts}")
                    # also print pixel_to_world for first few future points for debugging
                    world_pts = []
                    for (r_bev, c_bev) in future_pts[:5]:
                        # convert BEV pixel (row, col) -> image pixel (u,v) then to world meters
                        try:
                            u_img, v_img = H.bev_to_pixel(c_bev, r_bev)
                            x_m, y_m = H.pixel_to_world(float(u_img), float(v_img), flipped=True)
                            world_pts.append((x_m, y_m))
                        except Exception:
                            world_pts.append((None, None))
                    print(f"[GT_VIS]  sample_world_first5={world_pts}")
                except Exception:
                    pass

                # label the matched detection point on GT image (circle + text)
                try:
                    # det bx,by are (row, col). cv2 uses (x=col, y=row).
                    col = int(round(det["by"]))
                    row = int(round(det["bx"]))
                    # draw a filled circle and a contrasting border for visibility
                    cv2.circle(gt_vis, (col, row), 8, (0, 255, 255), -1)
                    cv2.circle(gt_vis, (col, row), 10, (0, 128, 128), 2)
                    # put label slightly offset to the top-right
                    cv2.putText(gt_vis, f"GT:{int(gid)}", (col+12, row-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                    cv2.putText(gt_vis, f"GT:{int(gid)}", (col+12, row-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
                except Exception:
                    pass

        # save original EKF BEV image and GT-overlaid version
        cv2.imwrite(str(out_ekf_bev / f"ekf_bev_gt_{img_path.name}"), gt_vis)
        cv2.imwrite(str(out_ekf_img / f"ekf_img_{img_path.name}"), ekf_img)

    assert True

    # After processing all frames, save the per-id detection traces to JSON
    detections_out_path = out_root / "detections_by_id.json"
    try:
        with open(detections_out_path, "w") as jf:
            json.dump(detections_by_id, jf, indent=2)
        print(f"Saved detections_by_id JSON to: {detections_out_path}")
    except Exception as e:
        print(f"Failed to save detections_by_id JSON: {e}")

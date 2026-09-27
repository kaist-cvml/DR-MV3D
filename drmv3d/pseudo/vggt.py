"""
VGGT inference-time augmentation (fail-fast by design).

This module intentionally does NOT swallow exceptions: if VGGT/model/rendering fails,
the error should surface as a normal Python exception so you can debug the real cause.
"""

from __future__ import annotations

import ctypes.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VGGTTopdownConfig:
    ckpt_path: str
    output_dir: str
    max_input_views: int = 4
    front_keyword: str = "front"
    render_size: int = 512
    # Default: do NOT filter by confidence percentile (user wants all points; camera pose should not depend on this).
    conf_percentile: float = 0.0
    pyopengl_platform: str = "egl"
    preprocess_mode: str = "crop"  # "crop" (default, VGGT util) or "pad" (preserve all pixels)
    # Point cloud selection / filtering (visual_util.predictions_to_glb options)
    use_point_map: bool = False  # default to depth+camera unprojection (often more accurate per VGGT docs)
    mask_black_bg: bool = False
    mask_white_bg: bool = False
    conf_min: float = 0.0        # absolute confidence floor (0 disables). Applied before percentile filtering.
    # Optional: pose refinement via COLMAP-style BA (pycolmap)
    ba_enabled: bool = False
    ba_max_query_pts: int = 4096
    ba_query_frame_num: int = 8
    ba_vis_thresh: float = 0.2
    ba_max_reproj_error: float = 8.0
    ba_fine_tracking: bool = True
    ba_keypoint_extractor: str = "aliked+sp"
    # Rendering / camera controls
    view: str = "topdown"            # "topdown" or "front_oblique"
    # NOTE: We intentionally keep rendering controls minimal. VGGT "accuracy" should be governed by
    # input views / preprocessing / model branch selection, not by arbitrary render camera styles.
    # We therefore render with a fixed perspective FOV and do not expose orthographic/FOV tuning.
    cam_distance_factor: float = 0.8  # fallback if cam_distance_abs==0 (kept for compatibility)
    cam_distance_abs: float = 0.0     # absolute camera distance in scene units (0 disables; overrides factor if >0)
    cam_height_abs: float = 0.0       # absolute camera height offset in scene units (0 disables; see front_oblique)
    oblique_deg: float = 25.0         # (topdown only) 0 = pure top-down; >0 tilts toward the "front"
    pitch_down_deg: float = 35.0      # (front_oblique only) tilt-down degrees (see implementation note)
    azimuth_deg: float = 0.0          # (front_oblique only) rotate around Y axis (degrees), keeping target fixed
    front_down: bool = True           # whether "front" should appear toward the bottom of the image
    # Match VGGT visual_util's notion of scene scale: it uses the 5th/95th percentile bbox.
    # This is not intended as a user-facing knob.
    _framing_lo_percentile: float = 5.0
    _framing_hi_percentile: float = 95.0


_VGGT_MODEL_CACHE: dict[str, Any] = {}


def _ensure_vggt_on_syspath() -> None:
    """
    Make the VGGT package importable.

    ``scripts/setup_third_party.sh`` clones it to ``modules/vggt``, which is not
    on ``sys.path``. An installed copy is used if there is one; otherwise the
    search walks up from this file looking for ``modules/vggt`` and puts
    ``modules/`` on the path, so no fixed directory depth is assumed.
    """
    try:
        import vggt  # noqa: F401
        return
    except Exception:
        pass

    here = Path(__file__).resolve()
    for p in here.parents:
        modules_dir = p / "modules"
        vggt_dir = modules_dir / "vggt"
        if vggt_dir.exists() and vggt_dir.is_dir():
            if str(modules_dir) not in sys.path:
                sys.path.insert(0, str(modules_dir))
            break


def vggt_run_predictions(image_paths: list[str], cfg: VGGTTopdownConfig) -> dict[str, Any]:
    """
    Run VGGT on up to cfg.max_input_views images and return *raw* predictions in VGGT's world frame.

    This intentionally does NOT call visual_util.predictions_to_glb() and does NOT apply
    any scene alignment transforms. Downstream consumers can build their own abstractions
    directly in the model's world frame.

    Returns a dict with numpy arrays (batch dimension removed), including:
      - images: (S, 3, H, W) float in [0,1] (preprocessed)
      - depth: (S, H, W, 1)
      - depth_conf: (S, H, W)
      - extrinsic: (S, 3, 4) world->camera (OpenCV: x right, y down, z forward)
      - intrinsic: (S, 3, 3)
      - world_points_from_depth: (S, H, W, 3)
    Also includes:
      - _selected_image_paths: list[str] (the actual paths fed to VGGT)
    """
    _ensure_vggt_on_syspath()

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    selected, _front0 = _select_views(image_paths, cfg.max_input_views, cfg.front_keyword)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT augmentation requires CUDA in this setup (demo_gradio style).")

    ckpt = Path(cfg.ckpt_path)
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt}")

    cache_key = str(ckpt.resolve())
    if cache_key not in _VGGT_MODEL_CACHE:
        model = VGGT()
        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _VGGT_MODEL_CACHE[cache_key] = model
    model = _VGGT_MODEL_CACHE[cache_key]

    if cfg.preprocess_mode not in ("crop", "pad"):
        raise ValueError(f"preprocess_mode must be 'crop' or 'pad', got {cfg.preprocess_mode!r}")
    images_tensor = load_and_preprocess_images(selected, mode=cfg.preprocess_mode).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor)

    # Ensure the preprocessed images are present for downstream modules (detection/seg uses these).
    if "images" not in predictions:
        predictions["images"] = images_tensor

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy (remove batch dim)
    out: dict[str, Any] = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy().squeeze(0)
        else:
            out[k] = v

    depth_map = out["depth"]
    out["world_points_from_depth"] = unproject_depth_map_to_point_map(depth_map, out["extrinsic"], out["intrinsic"])
    out["_selected_image_paths"] = list(selected)

    # Optional: refine camera poses via BA (pycolmap) and recompute world_points_from_depth
    if bool(getattr(cfg, "ba_enabled", False)):
        out = _maybe_refine_with_ba(out, cfg=cfg)
    return out


def _maybe_refine_with_ba(preds: dict[str, Any], *, cfg: VGGTTopdownConfig) -> dict[str, Any]:
    """
    Refine (extrinsic,intrinsic) using a COLMAP-style BA step on VGGT-predicted tracks, then
    recompute world_points_from_depth from the (unchanged) depth map.
    """
    import numpy as np
    # Dependency guard (only required when BA enabled)
    try:
        import pycolmap  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "VGGT BA requested (--vggt-ba) but pycolmap is not available. "
            "Install VGGT demo deps (see vggt/requirements_demo.txt)."
        ) from e

    images = preds.get("images", None)
    depth = preds.get("depth", None)
    depth_conf = preds.get("depth_conf", None)
    extrinsic = preds.get("extrinsic", None)
    intrinsic = preds.get("intrinsic", None)
    world_points_from_depth = preds.get("world_points_from_depth", None)
    if images is None or depth is None or depth_conf is None or extrinsic is None or intrinsic is None or world_points_from_depth is None:
        raise ValueError("VGGT BA requires predictions to include: images, depth, depth_conf, extrinsic, intrinsic, world_points_from_depth")

    S = int(np.asarray(images).shape[0])

    # Run BA
    _ensure_vggt_on_syspath()
    import torch
    import torch.nn.functional as F
    from vggt.dependency.np_to_pycolmap import (
        batch_np_matrix_to_pycolmap,
        pycolmap_to_batch_np_matrix,
    )
    from vggt.dependency.track_predict import predict_tracks

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT BA requires CUDA (tracker runs on GPU).")

    # NOTE: Track prediction uses LightGlue/ALIKED under the hood, whose weights are float32.
    # Using float16 inputs can trigger dtype mismatch errors (conv2d expects input/weight dtypes to match).
    # We therefore run the BA tracking step in float32 for robustness.
    img_t = torch.from_numpy(np.asarray(images)).to(device=device, dtype=torch.float32)  # (S,3,H,W) in [0,1]
    # NOTE: VGGT's track code may index NumPy arrays with a torch boolean mask. If that mask lives on CUDA,
    # NumPy will attempt to convert it (and fail). To keep this robust, keep conf/points_3d on CPU so that
    # any derived boolean masks are CPU tensors (convertible to NumPy).
    conf_t = torch.from_numpy(np.asarray(depth_conf)).to(device="cpu", dtype=torch.float32)  # (S,H,W)
    wp_t = torch.from_numpy(np.asarray(world_points_from_depth)).to(device="cpu", dtype=torch.float32)  # (S,H,W,3)

    # VGGT's tracker currently assumes square images (assert height == width in track_predict.py).
    # Our pipeline may produce non-square tensors depending on preprocessing; to keep BA robust, pad
    # images/conf/points_3d to a square canvas with zeros (which VGGT also recommends for masking unwanted pixels).
    def _pad_to_square(img_schw: torch.Tensor, conf_shw: torch.Tensor, wp_shwc: torch.Tensor):
        # img: (S,3,H,W), conf: (S,H,W), wp: (S,H,W,3)
        H = int(img_schw.shape[-2])
        W = int(img_schw.shape[-1])
        if H == W:
            return img_schw, conf_shw, wp_shwc
        side = max(H, W)
        pad_h = side - H
        pad_w = side - W
        # pad format: (left, right, top, bottom)
        img2 = F.pad(img_schw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        conf2 = F.pad(conf_shw.unsqueeze(1), (0, pad_w, 0, pad_h), mode="constant", value=0.0).squeeze(1)
        wp2 = F.pad(wp_shwc.permute(0, 3, 1, 2), (0, pad_w, 0, pad_h), mode="constant", value=0.0).permute(0, 2, 3, 1)
        return img2, conf2, wp2

    img_t, conf_t, wp_t = _pad_to_square(img_t, conf_t, wp_t)

    # Predict tracks
    # Track prediction is inference-only; ensure no gradients are tracked.
    with torch.no_grad():
        pred_tracks, pred_vis_scores, _pred_confs, pred_points_3d, points_rgb = predict_tracks(
            img_t,
            conf=conf_t,
            points_3d=wp_t,
            masks=None,
            max_query_pts=int(cfg.ba_max_query_pts),
            query_frame_num=int(cfg.ba_query_frame_num),
            keypoint_extractor=str(cfg.ba_keypoint_extractor),
            fine_tracking=bool(cfg.ba_fine_tracking),
        )

    # Build track mask for BA
    track_mask = np.asarray(pred_vis_scores) > float(cfg.ba_vis_thresh)
    image_size = np.asarray([img_t.shape[-1], img_t.shape[-2]], dtype=np.int32)  # (W,H) but np_to_pycolmap expects [W,H] shape

    # pred_points_3d: (P,3); pred_tracks: (S,P,2); track_mask: (S,P)
    reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
        np.asarray(pred_points_3d),
        np.asarray(extrinsic),
        np.asarray(intrinsic),
        np.asarray(pred_tracks),
        image_size,
        masks=track_mask,
        max_reproj_error=float(cfg.ba_max_reproj_error),
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
        points_rgb=np.asarray(points_rgb) if points_rgb is not None else None,
    )
    if reconstruction is None:
        preds = dict(preds)
        preds["_ba"] = {"status": "skipped", "reason": "no_reconstruction_built"}
        return preds

    import pycolmap
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    # Some runs may end up with no valid 3D points (e.g., too few inlier tracks).
    # In that case, we skip BA and fall back to feed-forward poses.
    try:
        pids = list(reconstruction.point3D_ids())
    except Exception:
        pids = []
    if len(pids) == 0:
        preds = dict(preds)
        preds["_ba"] = {
            "status": "skipped",
            "reason": "empty_points3D_after_ba",
            "vis_thresh": float(cfg.ba_vis_thresh),
            "max_reproj_error": float(cfg.ba_max_reproj_error),
            "max_query_pts": int(cfg.ba_max_query_pts),
            "query_frame_num": int(cfg.ba_query_frame_num),
            "fine_tracking": bool(cfg.ba_fine_tracking),
            "keypoint_extractor": str(cfg.ba_keypoint_extractor),
        }
        return preds

    # Extract refined extrinsic/intrinsic in batch order
    _pts3d, extr_ref_4x4, intr_ref_3x3, _extra = pycolmap_to_batch_np_matrix(reconstruction, camera_type="SIMPLE_PINHOLE")
    extr_ref = np.asarray(extr_ref_4x4, dtype=np.float32)[:, :3, :4]
    intr_ref = np.asarray(intr_ref_3x3, dtype=np.float32)

    # Update preds + recompute world points using refined poses
    preds2 = dict(preds)
    preds2["extrinsic"] = extr_ref
    preds2["intrinsic"] = intr_ref
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    preds2["world_points_from_depth"] = unproject_depth_map_to_point_map(preds2["depth"], preds2["extrinsic"], preds2["intrinsic"])
    preds2["_ba"] = {
        "status": "ok",
        "n_views": int(S),
        "n_tracks_valid": int(np.count_nonzero(valid_track_mask)) if valid_track_mask is not None else None,
        "vis_thresh": float(cfg.ba_vis_thresh),
        "max_reproj_error": float(cfg.ba_max_reproj_error),
        "max_query_pts": int(cfg.ba_max_query_pts),
        "query_frame_num": int(cfg.ba_query_frame_num),
        "fine_tracking": bool(cfg.ba_fine_tracking),
        "keypoint_extractor": str(cfg.ba_keypoint_extractor),
    }
    return preds2


def _select_views(image_paths: list[str], max_views: int, front_keyword: str) -> tuple[list[str], int | None]:
    """
    Pick up to max_views images; prefer a 'front' view first (by filename keyword).
    Returns (selected_paths, front_index_in_selected or None)
    """
    if max_views <= 0:
        raise ValueError("max_views must be > 0")

    if len(image_paths) == 0:
        raise ValueError("No input images provided for VGGT")

    paths = list(image_paths)
    front_idx = None
    for i, p in enumerate(paths):
        name = os.path.basename(p).lower()
        if front_keyword.lower() in name:
            front_idx = i
            break

    selected: list[str] = []
    if front_idx is not None:
        selected.append(paths[front_idx])
        # keep relative ordering for the rest
        for i, p in enumerate(paths):
            if i == front_idx:
                continue
            selected.append(p)
    else:
        selected = paths

    selected = selected[:max_views]
    # if we moved front to 0, it is 0; else unknown
    selected_front = 0 if front_idx is not None else None
    return selected, selected_front


def _look_at_pose(eye, target, up) -> object:
    import numpy as np

    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)

    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)

    true_up = np.cross(right, forward)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-12)

    # OpenGL camera pose: columns are basis vectors of camera frame in world.
    # Camera looks along -Z in its local frame, so z_cam points backward.
    z_cam = -forward
    x_cam = right
    y_cam = true_up

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = x_cam
    pose[:3, 1] = y_cam
    pose[:3, 2] = z_cam
    pose[:3, 3] = eye
    return pose


def _rotate_vec_around_axis(v, axis, angle_rad):
    """
    Rodrigues rotation formula: rotate vector v around 'axis' by angle_rad.
    """
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1.0 - c)


def _render_pointcloud_topdown(
    points_xyz,
    colors_rgb,
    *,
    render_size: int,
    cfg: VGGTTopdownConfig,
    target_center=None,
) -> object:
    """
    Offscreen render with pyrender. Raises on any failure (missing EGL, deps, etc.).
    """
    import numpy as np
    import pyrender

    pts = np.asarray(points_xyz, dtype=np.float32)
    cols = np.asarray(colors_rgb, dtype=np.uint8)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {pts.shape}")
    if cols.shape[0] != pts.shape[0]:
        raise ValueError("colors must have same length as points")
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError(f"colors must be (N,3), got {cols.shape}")

    # Normalize colors to 0..1 float for pyrender
    cols_f = (cols.astype(np.float32) / 255.0)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.8, 0.8, 0.8])
    cloud = pyrender.Mesh.from_points(pts, colors=cols_f)
    scene.add(cloud)

    # Match VGGT visual_util scale computation: robust 5th/95th percentile bbox.
    # NOTE: camera target center can be overridden (e.g., computed from raw/unfiltered points)
    # so changing conf_percentile doesn't move the camera.
    lo_p = float(cfg._framing_lo_percentile)
    hi_p = float(cfg._framing_hi_percentile)
    if not (0.0 <= lo_p < hi_p <= 100.0):
        raise ValueError(f"Invalid framing percentiles: lo={lo_p}, hi={hi_p}")
    lo = np.percentile(pts, lo_p, axis=0).astype(np.float32)
    hi = np.percentile(pts, hi_p, axis=0).astype(np.float32)
    if target_center is None:
        center = np.median(pts, axis=0).astype(np.float32)
    else:
        center = np.asarray(target_center, dtype=np.float32).reshape(3)
    scene_scale = float(np.linalg.norm(hi - lo) + 1e-6)

    # For absolute distance, don't let bbox/diag affect camera distance. Otherwise use scene_scale.
    diag = 1.0 if float(cfg.cam_distance_abs) > 0.0 else scene_scale
    if cfg.view not in ("topdown", "front_oblique"):
        raise ValueError(f"Unknown VGGT render view mode: {cfg.view!r} (expected 'topdown' or 'front_oblique')")

    target = center
    # Camera distance in scene units. If cam_distance_abs > 0, use it; else fall back to factor*diag.
    if float(cfg.cam_distance_abs) > 0.0:
        base_dist = float(cfg.cam_distance_abs)
    else:
        base_dist = float(cfg.cam_distance_factor * diag)

    if cfg.view == "topdown":
        # After visual_util alignment, "front" is roughly -Z. For a more interpretable view,
        # we keep a mostly top-down camera but tilt it toward +Z (so front is "down" in image).
        h = base_dist
        z = float(h * np.tan(np.deg2rad(float(cfg.oblique_deg))))
        eye = center + np.array([0.0, h, z], dtype=np.float32)
        # "up" controls whether the front direction appears toward the top or bottom of the image.
        up = np.array([0.0, 0.0, 1.0 if cfg.front_down else -1.0], dtype=np.float32)
    else:
        # Front-ish view from a ring around the scene center (XZ plane), with independent controls:
        # - cam_distance_abs: ring radius (horizontal distance)
        # - cam_height_abs: camera height offset. When it is not set, the height is
        #   derived from pitch_down_deg instead.
        # - pitch_down_deg: tilt-down (a pure tilt knob when cam_height_abs > 0)
        dist = base_dist
        pitch = float(cfg.pitch_down_deg)
        if not (0.0 <= pitch < 90.0):
            raise ValueError(f"pitch_down_deg must be in [0, 90), got {pitch}")

        # Backward-compat default: if cam_height_abs is not set, pitch also drives elevation (old behavior).
        if float(getattr(cfg, "cam_height_abs", 0.0)) > 0.0:
            height = float(cfg.cam_height_abs)
            tilt_deg = pitch
        else:
            height = float(dist * np.tan(np.deg2rad(pitch)))
            tilt_deg = 0.0

        az = float(cfg.azimuth_deg)
        # After VGGT's visual_util.apply_scene_alignment(), the aligned scene uses an OpenGL-like convention
        # (Y up) with an additional 180° Y rotation. Empirically, placing the camera on -Z for azimuth=0
        # yields a "front view" that matches the demos more consistently than +Z.
        # azimuth around world Y axis: 0 => -Z, 90 => +X
        dx = float(np.sin(np.deg2rad(az)) * dist)
        dz = float(-np.cos(np.deg2rad(az)) * dist)
        # IMPORTANT: VGGT's scene alignment includes a Y flip (see visual_util.get_opengl_conversion_matrix),
        # so we negate elevation to make positive "height" move the camera up in the user's intuitive sense.
        eye = center + np.array([dx, -height, dz], dtype=np.float32)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Base target: look at the scene center.
    target = center

    # Optional tilt-only (front_oblique + cam_height_abs): rotate the view direction around camera-right axis.
    if cfg.view == "front_oblique" and float(getattr(cfg, "cam_height_abs", 0.0)) > 0.0:
        import numpy as np

        forward = (target - eye).astype(np.float64)
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        right = np.cross(forward, up.astype(np.float64))
        right = right / (np.linalg.norm(right) + 1e-12)
        forward_tilted = _rotate_vec_around_axis(forward, right, np.deg2rad(float(tilt_deg)))
        # Choose a target point along the tilted direction, keeping roughly the same look distance.
        target = (eye.astype(np.float64) + forward_tilted * float(base_dist)).astype(np.float32)

    cam_pose = _look_at_pose(eye, target, up)

    # Fixed perspective camera (keep render style stable; user controls pose via pitch/azimuth/distance only)
    cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
    scene.add(cam, pose=cam_pose)

    # multi-light for point clouds (helps depth perception)
    light_scale = base_dist if float(cfg.cam_distance_abs) > 0.0 else scene_scale
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=cam_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
        pose=_look_at_pose(center + np.array([light_scale, light_scale, light_scale]), target, up),
    )
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
        pose=_look_at_pose(center + np.array([-light_scale, light_scale, -light_scale]), target, up),
    )

    r = pyrender.OffscreenRenderer(viewport_width=render_size, viewport_height=render_size)
    color, _ = r.render(scene)
    r.delete()
    return color


def vggt_reconstruct_and_render_topdown(image_paths: list[str], cfg: VGGTTopdownConfig) -> str:
    """
    Run VGGT on up to cfg.max_input_views images and save a single rendered top-down image.
    Returns the saved image path.
    """
    _ensure_vggt_on_syspath()

    import numpy as np
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from visual_util import predictions_to_glb

    selected, _front0 = _select_views(image_paths, cfg.max_input_views, cfg.front_keyword)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("VGGT augmentation requires CUDA in this setup (demo_gradio style).")

    # Force EGL/OSMesa selection for headless rendering (must be set before importing OpenGL/pyrender).
    os.environ["PYOPENGL_PLATFORM"] = cfg.pyopengl_platform
    if cfg.pyopengl_platform == "egl":
        # Many headless setups need this hint.
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        if ctypes.util.find_library("EGL") is None:
            raise RuntimeError(
                "PYOPENGL_PLATFORM=egl was requested but libEGL could not be found on this system.\n"
                "Install an EGL provider (Mesa) inside the container, e.g.:\n"
                "  apt-get update && apt-get install -y libegl1 libegl1-mesa libgles2-mesa mesa-utils\n"
            )
    if cfg.pyopengl_platform == "osmesa":
        if ctypes.util.find_library("OSMesa") is None:
            raise RuntimeError(
                "PYOPENGL_PLATFORM=osmesa was requested but libOSMesa could not be found on this system.\n"
                "Install OSMesa inside the container, e.g.:\n"
                "  apt-get update && apt-get install -y libosmesa6 libgl1-mesa-dri mesa-utils\n"
            )

    ckpt = Path(cfg.ckpt_path)
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"VGGT checkpoint not found: {ckpt}")

    cache_key = str(ckpt.resolve())
    if cache_key not in _VGGT_MODEL_CACHE:
        model = VGGT()
        state = torch.load(str(ckpt), map_location="cpu")
        # Support common checkpoint wrappers without swallowing errors.
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval().to(device)
        _VGGT_MODEL_CACHE[cache_key] = model
    model = _VGGT_MODEL_CACHE[cache_key]

    if cfg.preprocess_mode not in ("crop", "pad"):
        raise ValueError(f"preprocess_mode must be 'crop' or 'pad', got {cfg.preprocess_mode!r}")
    images_tensor = load_and_preprocess_images(selected, mode=cfg.preprocess_mode).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images_tensor)

    # Some model variants may not return the input images in predictions; visual_util expects it.
    if "images" not in predictions:
        predictions["images"] = images_tensor

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Move tensors -> numpy (like demo_gradio)
    for k, v in list(predictions.items()):
        if isinstance(v, torch.Tensor):
            predictions[k] = v.detach().cpu().numpy().squeeze(0)

    # Ensure predictions include depth-derived world points for fallback
    depth_map = predictions["depth"]
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )

    # Optional absolute confidence threshold (applied before percentile-based filtering in visual_util).
    # We do this by zeroing low-confidence pixels so predictions_to_glb() masks them out.
    if cfg.conf_min and float(cfg.conf_min) > 0.0:
        if cfg.use_point_map and "world_points_conf" in predictions:
            conf = predictions["world_points_conf"]
            predictions["world_points_conf"] = (conf * (conf >= float(cfg.conf_min))).astype(conf.dtype, copy=False)
        elif (not cfg.use_point_map) and "depth_conf" in predictions:
            conf = predictions["depth_conf"]
            predictions["depth_conf"] = (conf * (conf >= float(cfg.conf_min))).astype(conf.dtype, copy=False)

    # Build aligned trimesh.Scene (demo util). We hide cameras for cleaner render.
    prediction_mode = "Predicted Pointmap" if cfg.use_point_map else "Depthmap and Camera Branch"

    # Compute a stable camera target center from *raw/unfiltered* points so that conf_percentile only
    # affects point density (render content), not the camera pose/framing.
    # We still use robust percentiles to avoid extreme outliers.
    try:
        raw_wp = predictions["world_points"] if (cfg.use_point_map and "world_points" in predictions) else predictions["world_points_from_depth"]
        raw_wp = np.asarray(raw_wp, dtype=np.float32)
        raw_pts = raw_wp.reshape(-1, 3)
        raw_pts = raw_pts[np.isfinite(raw_pts).all(axis=1)]
        if raw_pts.shape[0] >= 16:
            lo = np.percentile(raw_pts, cfg._framing_lo_percentile, axis=0).astype(np.float32)
            hi = np.percentile(raw_pts, cfg._framing_hi_percentile, axis=0).astype(np.float32)
            inlier = np.all((raw_pts >= lo) & (raw_pts <= hi), axis=1)
            inlier_pts = raw_pts[inlier]
            cam_center = np.median(inlier_pts if inlier_pts.shape[0] >= 16 else raw_pts, axis=0).astype(np.float32)
        else:
            cam_center = np.median(raw_pts, axis=0).astype(np.float32) if raw_pts.shape[0] else None
    except Exception:
        # Fail-safe: if raw center computation breaks for any reason, fall back to render-points median.
        cam_center = None

    scene_3d = predictions_to_glb(
        predictions,
        conf_thres=cfg.conf_percentile,
        show_cam=False,
        mask_black_bg=cfg.mask_black_bg,
        mask_white_bg=cfg.mask_white_bg,
        prediction_mode=prediction_mode,
    )

    # Extract point cloud from scene

    points = None
    colors = None
    for geom in scene_3d.geometry.values():
        if geom.__class__.__name__ == "PointCloud":
            points = np.asarray(geom.vertices)
            colors = np.asarray(geom.colors)[:, :3]
            break
    if points is None or colors is None:
        raise RuntimeError("Failed to extract point cloud from VGGT scene")

    color_img = _render_pointcloud_topdown(
        points, colors, render_size=cfg.render_size, cfg=cfg, target_center=cam_center
    )

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vggt_topdown_{abs(hash('|'.join(selected))) % 10**12}.png"

    from PIL import Image

    # In many headless EGL setups, the rendered framebuffer origin is bottom-left, resulting in a vertical flip.
    # Fix deterministically (no extra user knobs): only apply for PYOPENGL_PLATFORM=egl.
    if cfg.pyopengl_platform == "egl":
        color_img = color_img[::-1].copy()

    Image.fromarray(color_img).save(str(out_path))
    return str(out_path)



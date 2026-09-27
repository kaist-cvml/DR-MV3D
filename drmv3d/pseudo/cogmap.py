"""
A 10x10 bird's-eye cognitive map estimated from the images alone.

- For each object and each view, SAM 3 turns the object's name into candidate masks.
- Mask pixels are lifted into world coordinates with VGGT's point map and reduced
  to one robust median anchor per candidate.
- The per-view anchors are fused, again by median, into one world position per
  object, with candidates penalised for disagreeing across views.
- Camera positions come from the extrinsics.
- Everything is quantised onto the 10x10 grid by a robust per-axis fit, so the map
  uses the full grid.

Optionally saves debug renders: per-view tiles with the selected masks and labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import (
    _camera_center_world,
    _camera_forward_world,
    _camera_up_world,
    _normalize_depth_conf_hw,
    _preprocessed_view_to_pil,
    _quantize_dir_2d,
    _unit,
)


@dataclass(frozen=True)
class PseudoCogmapConfig:
    device: str = "cuda"
    object_names: tuple[str, ...] = ()
    max_views: int = 4

    # Filtering
    min_depth_conf: float = 0.0
    depth_trim_lo_percentile: float = 10.0
    depth_trim_hi_percentile: float = 90.0

    # Association
    assoc_topk_per_view: int = 3

    # Segmentation backend:
    # - "sam3_image": one pass per view, text prompt straight to masks and scores
    # - "sam3_track": the video predictor, propagating across views as if they
    #   were frames
    det_backend: str = "sam3_image"

    # SAM 3. If sam3_root is set, it is appended to sys.path to import `sam3`.
    sam3_root: str = ""
    # If empty, SAM3 will default to downloading weights from HuggingFace Hub.
    sam3_checkpoint: str = ""
    sam3_conf_threshold: float = 0.50
    # SAM3 video predictor options (sam3_track)
    sam3_apply_temporal_disambiguation: bool = False
    sam3_async_loading_frames: bool = False
    sam3_video_loader_type: str = "cv2"
    sam3_propagation_direction: str = "both"  # both|forward|backward
    sam3_seed_frame_idx: int = 0

    # Debug rendering (wired from engine; no new CLI flags required)
    render_detection_debug: bool = False
    render_output_dir: str | None = None
    render_size: int = 512
    sample_id: str = ""


def _world_to_cam_xyz(
    *,
    xyz_world: np.ndarray,
    ext_3x4_world_to_cam: object,
) -> np.ndarray:
    ext = np.asarray(ext_3x4_world_to_cam, dtype=np.float64).reshape(3, 4)
    R = ext[:, :3]
    t = ext[:, 3]
    return (R @ xyz_world.reshape(3) + t).astype(np.float64)


def _project_cam_to_px(
    *,
    xyz_cam: np.ndarray,
    K_3x3: object,
) -> tuple[float, float]:
    K = np.asarray(K_3x3, dtype=np.float64).reshape(3, 3)
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x, y, z = float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])
    z = z if abs(z) > 1e-9 else 1e-9
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    return float(u), float(v)


def _in_image(u: float, v: float, W: int, H: int) -> bool:
    return (0.0 <= float(u) < float(W)) and (0.0 <= float(v) < float(H))


def _robust_anchor_world_from_mask_pointmap(
    *,
    mask_hw: object,
    wp_hw3: object,
    conf_hw: object | None,
    ext_3x4_world_to_cam: object,
    min_depth_conf: float,
    stride: int = 6,
    max_points: int = 5000,
    camz_lo_percentile: float = 10.0,
    camz_hi_percentile: float = 90.0,
    min_points: int = 32,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """
    Lift a 2D mask into WORLD using per-pixel point map, then take a robust median anchor.
    Returns (anchor_world_xyz, debug_stats) or None if insufficient valid points.
    """
    m = np.asarray(mask_hw).astype(bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be HxW bool, got shape={tuple(m.shape)}")
    wp = np.asarray(wp_hw3, dtype=np.float64)
    if wp.ndim != 3 or int(wp.shape[2]) != 3:
        raise ValueError(f"wp_hw3 must be HxWx3, got shape={tuple(wp.shape)}")
    H, W, _ = wp.shape
    if m.shape[0] != H or m.shape[1] != W:
        raise ValueError(f"mask shape {tuple(m.shape)} does not match wp shape {(H, W)}")

    yy, xx = np.nonzero(m)
    n_raw = int(yy.size)
    if n_raw < int(min_points):
        return None

    stride = int(max(1, stride))
    yy = yy[::stride]
    xx = xx[::stride]
    if int(yy.size) > int(max_points):
        yy = yy[: int(max_points)]
        xx = xx[: int(max_points)]

    P = wp[yy, xx].reshape(-1, 3)
    valid = np.all(np.isfinite(P), axis=1)
    conf_mean = None
    if conf_hw is not None:
        c = np.asarray(conf_hw, dtype=np.float64)
        if c.shape != (H, W):
            raise ValueError(f"conf_hw shape {tuple(c.shape)} does not match wp shape {(H, W)}")
        cc = c[yy, xx].reshape(-1)
        valid = valid & np.isfinite(cc)
        if float(min_depth_conf) > 0.0:
            valid = valid & (cc >= float(min_depth_conf))
        if int(np.count_nonzero(valid)) > 0:
            conf_mean = float(np.mean(cc[valid]))

    P = P[valid]
    n_after_conf = int(P.shape[0])
    if n_after_conf < int(min_points):
        return None

    ext = np.asarray(ext_3x4_world_to_cam, dtype=np.float64).reshape(3, 4)
    R = ext[:, :3]
    t = ext[:, 3]
    Pc = (P @ R.T) + t[None, :]
    z = Pc[:, 2]
    z_valid = np.isfinite(z)
    z = z[z_valid]
    P = P[z_valid]
    if int(P.shape[0]) < int(min_points):
        return None

    lo = float(np.clip(float(camz_lo_percentile), 0.0, 49.0))
    hi = float(np.clip(float(camz_hi_percentile), 51.0, 100.0))
    z_lo = float(np.percentile(z, lo))
    z_hi = float(np.percentile(z, hi))
    keep = (z >= z_lo) & (z <= z_hi)
    if int(np.count_nonzero(keep)) >= int(min_points):
        Pk = P[keep]
        z_used = z[keep]
    else:
        Pk = P
        z_used = z

    anchor = np.median(Pk, axis=0).astype(np.float64)
    dbg = {
        "n_raw_mask_px": int(n_raw),
        "n_sampled": int(yy.size),
        "n_after_conf": int(n_after_conf),
        "n_after_camz": int(Pk.shape[0]),
        "conf_mean": (None if conf_mean is None else float(conf_mean)),
        "camz_lo": float(z_lo),
        "camz_hi": float(z_hi),
        "camz_med": float(np.median(z_used)),
    }
    return anchor.reshape(3), dbg


def _render_detection_debug_grid(
    *,
    view_pils: list[object],
    selected_masks: dict[tuple[str, int], object],
    selected_boxes_xyxy: dict[tuple[str, int], tuple[int, int, int, int]],
    selected_labels: dict[tuple[str, int], str],
    selected_reproj_px: dict[tuple[str, int], tuple[float, float]] | None = None,
    selected_stats: dict[tuple[str, int], str] | None = None,
    object_names: tuple[str, ...],
    obj_key_to_orig: dict[str, str],
    max_views: int,
    render_size: int,
) -> object | None:
    import math

    from PIL import Image, ImageDraw, ImageFont

    V = min(int(max_views), int(len(view_pils)))
    if V <= 0:
        return None

    cols = 2 if V > 1 else 1
    rows = int(math.ceil(V / float(cols)))
    pad = 10
    header_h = 26
    tile = int(max(160, min(384, int(render_size) // max(1, rows))))
    out_w = pad + cols * tile + (cols - 1) * pad + pad
    out_h = header_h + pad + rows * tile + (rows - 1) * pad + pad
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    hdr = "estimated cognitive map: selected masks and boxes per view"
    draw.text((pad, 4), hdr, fill=(0, 0, 0), font=font)

    _palette = [
        (220, 50, 50),
        (50, 180, 70),
        (50, 90, 200),
        (220, 140, 40),
        (150, 70, 200),
        (40, 180, 190),
        (220, 70, 160),
        (220, 200, 60),
        (140, 90, 40),
        (120, 120, 120),
        (30, 30, 30),
    ]

    def _rgb(name: str):
        return _palette[abs(hash(name)) % max(1, len(_palette))]

    alpha = 0.33

    for vi in range(V):
        rr = int(vi // cols)
        cc = int(vi % cols)
        x0 = pad + cc * (tile + pad)
        y0 = header_h + pad + rr * (tile + pad)
        base_pil = view_pils[vi].convert("RGB")
        W0, H0 = base_pil.size
        W0 = max(1, int(W0))
        H0 = max(1, int(H0))
        base = base_pil.resize((tile, tile), resample=Image.BILINEAR)
        base_np = np.asarray(base).astype(np.float32)
        overlay_np = base_np.copy()

        for obj in object_names:
            m = selected_masks.get((obj, int(vi)))
            if m is None:
                continue
            m2 = np.asarray(m).astype(bool)
            if m2.ndim != 2 or int(m2.sum()) < 8:
                continue
            m_img = Image.fromarray((m2.astype(np.uint8) * 255), mode="L").resize((tile, tile), resample=Image.NEAREST)
            mm = (np.asarray(m_img) > 127)
            col = np.asarray(_rgb(obj), dtype=np.float32).reshape(1, 1, 3)
            overlay_np[mm] = col

        out_tile = (base_np * (1.0 - alpha) + overlay_np * alpha).astype(np.uint8)
        tile_img = Image.fromarray(out_tile, mode="RGB")
        out.paste(tile_img, (x0, y0))
        draw.rectangle([(x0, y0), (x0 + tile, y0 + tile)], outline=(0, 0, 0), width=2)
        draw.text((x0 + 6, y0 + 6), f"Image {vi+1}", fill=(0, 0, 0), font=font)

        sx = float(tile) / float(W0)
        sy = float(tile) / float(H0)
        for obj in object_names:
            b = selected_boxes_xyxy.get((obj, int(vi)))
            if b is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in b)
            xx1 = x0 + int(round(float(x1) * sx))
            yy1 = y0 + int(round(float(y1) * sy))
            xx2 = x0 + int(round(float(x2) * sx))
            yy2 = y0 + int(round(float(y2) * sy))
            col = _rgb(obj)
            draw.rectangle([(xx1, yy1), (xx2, yy2)], outline=col, width=3)
            lab = selected_labels.get((obj, int(vi))) or obj_key_to_orig.get(obj, obj)
            draw.text((xx1 + 3, max(y0 + 22, yy1 + 3)), str(lab)[:28], fill=col, font=font)

            if selected_reproj_px is not None:
                rp = selected_reproj_px.get((obj, int(vi)))
                if rp is not None:
                    u_px, v_px = float(rp[0]), float(rp[1])
                    rx = x0 + int(round(u_px * sx))
                    ry = y0 + int(round(v_px * sy))
                    r = 6
                    draw.line([(rx - r, ry), (rx + r, ry)], fill=col, width=3)
                    draw.line([(rx, ry - r), (rx, ry + r)], fill=col, width=3)

            if selected_stats is not None:
                st = selected_stats.get((obj, int(vi)))
                if st:
                    draw.text((xx1 + 3, min(y0 + tile - 14, yy2 + 3)), str(st)[:40], fill=col, font=font)

    return out




def _render_segment_masks_grid(
    *,
    view_pils: list[object],
    object_names: tuple[str, ...],
    obj_key_to_orig: dict[str, str],
    cand_cache: dict[tuple[str, int], list[dict[str, Any]]],
    max_views: int,
    render_size: int,
    title: str,
) -> object | None:
    """
    Render a debug grid of the masks that enter association and lifting.

    Layout:
    - rows: object categories (text prompts)
    - cols: views (Image 1..V)

    Each cell overlays the union of all candidate masks retained for that (obj, view).
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    V = min(int(max_views), int(len(view_pils)))
    K = int(len(object_names))
    if V <= 0 or K <= 0:
        return None

    pad = 8
    header_h = 26
    label_w = 180
    tile = int(max(160, min(320, int(render_size) // max(1, min(3, K)))))
    out_w = pad + label_w + pad + V * tile + (V - 1) * pad + pad
    out_h = header_h + pad + K * tile + (K - 1) * pad + pad
    out = Image.new("RGB", (int(out_w), int(out_h)), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    draw.text((pad, 4), str(title)[:140], fill=(0, 0, 0), font=font)

    _palette = [
        (220, 50, 50),
        (50, 180, 70),
        (50, 90, 200),
        (220, 140, 40),
        (150, 70, 200),
        (40, 180, 190),
        (220, 70, 160),
        (220, 200, 60),
        (140, 90, 40),
        (120, 120, 120),
        (30, 30, 30),
    ]

    def _row_color(name: str) -> tuple[int, int, int]:
        return _palette[abs(hash(name)) % max(1, len(_palette))]

    # Column headers
    for vi in range(V):
        x0 = pad + label_w + pad + vi * (tile + pad)
        draw.text((x0 + 6, header_h - 16), f"Image {vi+1}", fill=(0, 0, 0), font=font)

    # Make masks clearly visible (user request): use a stronger overlay.
    alpha = 0.72

    for ri, obj in enumerate(object_names):
        y0 = header_h + pad + ri * (tile + pad)
        lab = obj_key_to_orig.get(obj, obj)
        col = _row_color(obj)
        draw.rectangle([(pad, y0), (pad + label_w, y0 + tile)], outline=(0, 0, 0), width=1)
        sw = 10
        draw.rectangle([(pad + 8, y0 + 8), (pad + 8 + sw, y0 + 8 + sw)], fill=col, outline=(0, 0, 0), width=1)
        draw.text((pad + 8 + sw + 6, y0 + 6), str(lab)[:34], fill=(0, 0, 0), font=font)

        for vi in range(V):
            x0 = pad + label_w + pad + vi * (tile + pad)
            base_pil = view_pils[vi].convert("RGB")
            W0, H0 = base_pil.size
            W0 = max(1, int(W0))
            H0 = max(1, int(H0))
            base = base_pil.resize((tile, tile), resample=Image.BILINEAR)
            base_np = np.asarray(base).astype(np.float32)
            overlay = base_np.copy()

            cands = cand_cache.get((obj, int(vi))) or []
            n = int(len(cands))
            if n > 0:
                # union masks
                u = None
                for c in cands[:20]:
                    m = c.get("mask")
                    if m is None:
                        continue
                    mm = np.asarray(m).astype(bool)
                    if mm.ndim != 2:
                        continue
                    u = mm if u is None else (u | mm)
                if u is not None and int(u.sum()) > 0:
                    m_img = Image.fromarray((u.astype(np.uint8) * 255), mode="L").resize((tile, tile), resample=Image.NEAREST)
                    mu = (np.asarray(m_img) > 127)
                    cc = np.asarray(col, dtype=np.float32).reshape(1, 1, 3)
                    overlay[mu] = cc
                    # Also draw a bbox around the union mask for quick inspection.
                    yy, xx = np.nonzero(mu)
                    if yy.size > 0:
                        x1 = int(np.min(xx))
                        y1 = int(np.min(yy))
                        x2 = int(np.max(xx))
                        y2 = int(np.max(yy))
                        # draw on overlay directly (so bbox is visible even with alpha blend)
                        overlay[y1 : y1 + 2, x1 : x2 + 1] = cc
                        overlay[y2 - 1 : y2 + 1, x1 : x2 + 1] = cc
                        overlay[y1 : y2 + 1, x1 : x1 + 2] = cc
                        overlay[y1 : y2 + 1, x2 - 1 : x2 + 1] = cc

            out_tile = (base_np * (1.0 - alpha) + overlay * alpha).astype(np.uint8)
            cell = Image.fromarray(out_tile, mode="RGB")
            out.paste(cell, (x0, y0))
            draw.rectangle([(x0, y0), (x0 + tile, y0 + tile)], outline=(0, 0, 0), width=1)
            # small count label in the corner
            draw.text((x0 + 6, y0 + 6), f"n={n}", fill=col, font=font)

    return out


def build_pseudo_cogmap(
    predictions: dict[str, Any],
    *,
    cfg: PseudoCogmapConfig,
) -> tuple[str, dict[str, Any]]:
    """
    Returns (text_block, meta_dict).
    """
    import os
    import sys
    import tempfile
    from pathlib import Path

    import torch

    det_backend = str(getattr(cfg, "det_backend", "sam3_image") or "sam3_image")
    if det_backend not in ("sam3_image", "sam3_track"):
        raise RuntimeError(
            f"Unsupported det_backend={det_backend!r} (supported: 'sam3_image', 'sam3_track')."
        )

    def _resolve_relpath(root: str, p: str) -> str:
        pp = Path(str(p))
        if pp.is_absolute():
            return str(pp)
        return str((Path(str(root)).resolve() / pp).resolve())




    def _ensure_sam3_on_syspath(sam3_root: str) -> None:
        s = str(sam3_root or "").strip()
        if not s:
            return
        root = Path(s).resolve()
        if not root.exists():
            raise FileNotFoundError(f"SAM3 root not found: {root}")
        p = str(root)
        if p not in sys.path:
            sys.path.insert(0, p)

    def _get_sam3_processor() -> object:
        """
        Lazy, cached SAM3 image processor.
        Uses cfg.sam3_root, cfg.sam3_checkpoint, cfg.sam3_conf_threshold.
        """
        _ensure_sam3_on_syspath(getattr(cfg, "sam3_root", ""))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        key = (
            str(getattr(cfg, "sam3_root", "")),
            str(getattr(cfg, "sam3_checkpoint", "")),
            str(cfg.device),
            float(getattr(cfg, "sam3_conf_threshold", 0.50)),
        )
        global _PSEUDO_SAM3_PROC_CACHE
        if "_PSEUDO_SAM3_PROC_CACHE" not in globals():
            _PSEUDO_SAM3_PROC_CACHE = {}  # type: ignore[assignment]
        proc = _PSEUDO_SAM3_PROC_CACHE.get(key)  # type: ignore[name-defined]
        if proc is None:
            ckpt_any = str(getattr(cfg, "sam3_checkpoint", "") or "").strip()
            ckpt_path = None
            if ckpt_any:
                ckpt_path = _resolve_relpath(getattr(cfg, "sam3_root", "") or ".", ckpt_any)
            model = build_sam3_image_model(
                device=str(cfg.device),
                eval_mode=True,
                checkpoint_path=ckpt_path,
                load_from_HF=(ckpt_path is None),
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
            proc = Sam3Processor(
                model=model,
                device=str(cfg.device),
                confidence_threshold=float(getattr(cfg, "sam3_conf_threshold", 0.50)),
            )
            _PSEUDO_SAM3_PROC_CACHE[key] = proc  # type: ignore[name-defined]
        return proc

    def _get_sam3_video_predictor() -> object:
        """
        Lazy, cached SAM3 video predictor (session-based API).
        Uses cfg.sam3_root, cfg.sam3_checkpoint, and cfg.sam3_* video options.
        """
        _ensure_sam3_on_syspath(getattr(cfg, "sam3_root", ""))
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        key = (
            str(getattr(cfg, "sam3_root", "")),
            str(getattr(cfg, "sam3_checkpoint", "")),
            bool(getattr(cfg, "sam3_async_loading_frames", False)),
            str(getattr(cfg, "sam3_video_loader_type", "cv2") or "cv2"),
            bool(getattr(cfg, "sam3_apply_temporal_disambiguation", False)),
        )
        global _PSEUDO_SAM3_VIDPRED_CACHE
        if "_PSEUDO_SAM3_VIDPRED_CACHE" not in globals():
            _PSEUDO_SAM3_VIDPRED_CACHE = {}  # type: ignore[assignment]
        pred = _PSEUDO_SAM3_VIDPRED_CACHE.get(key)  # type: ignore[name-defined]
        if pred is None:
            ckpt_any = str(getattr(cfg, "sam3_checkpoint", "") or "").strip()
            ckpt_path = None
            if ckpt_any:
                ckpt_path = _resolve_relpath(getattr(cfg, "sam3_root", "") or ".", ckpt_any)
            pred = Sam3VideoPredictor(
                checkpoint_path=ckpt_path,
                async_loading_frames=bool(getattr(cfg, "sam3_async_loading_frames", False)),
                video_loader_type=str(getattr(cfg, "sam3_video_loader_type", "cv2") or "cv2"),
                apply_temporal_disambiguation=bool(getattr(cfg, "sam3_apply_temporal_disambiguation", False)),
            )
            _PSEUDO_SAM3_VIDPRED_CACHE[key] = pred  # type: ignore[name-defined]
        return pred

    def _mask_to_xyxy(mask_hw: np.ndarray) -> tuple[int, int, int, int]:
        yy, xx = np.nonzero(mask_hw)
        if yy.size == 0:
            return (0, 0, 0, 0)
        x0 = int(np.min(xx))
        y0 = int(np.min(yy))
        x1 = int(np.max(xx))
        y1 = int(np.max(yy))
        return (x0, y0, x1, y1)

    pred_images = predictions.get("images", None)
    depth = predictions.get("depth", None)
    depth_conf = predictions.get("depth_conf", None)
    extrinsic = predictions.get("extrinsic", None)
    intrinsic = predictions.get("intrinsic", None)
    wp_all = predictions.get("world_points_from_depth", None)
    if pred_images is None or depth is None or extrinsic is None or intrinsic is None or wp_all is None:
        raise ValueError("predictions missing required keys among: images, depth, extrinsic, intrinsic, world_points_from_depth")

    # normalize object list
    obj_pairs: list[tuple[str, str]] = []
    for o in (cfg.object_names or ()):
        if not isinstance(o, str):
            continue
        orig = o.strip()
        if not orig:
            continue
        obj_pairs.append((orig.lower(), orig))
    seen = set()
    obj_pairs2: list[tuple[str, str]] = []
    for k, orig in obj_pairs:
        if k in seen:
            continue
        seen.add(k)
        obj_pairs2.append((k, orig))
    object_names = tuple([k for (k, _o) in obj_pairs2])
    obj_key_to_orig = {k: o for (k, o) in obj_pairs2}
    if not object_names:
        raise ValueError("object_names is empty (pseudo-cogmap needs the target object list).")

    ext_all = np.asarray(extrinsic)
    intr_all = np.asarray(intrinsic)
    S = int(ext_all.shape[0])
    max_views = min(int(cfg.max_views), int(S))

    depth_all = np.asarray(depth)
    conf_all = np.asarray(depth_conf) if depth_conf is not None else None
    wp_all_np = np.asarray(wp_all)

    # BEV basis from camera poses
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ups = []
    fws = []
    for i in range(max_views):
        ups.append(_unit(_camera_up_world(ext_all[i])))
        fws.append(_unit(_camera_forward_world(ext_all[i])))
    if len(ups) > 0:
        up_world = _unit(np.mean(np.stack(ups, axis=0), axis=0))
    fw_proj = []
    for fw in fws:
        fwp = fw - float(np.dot(fw, up_world)) * up_world
        if float(np.linalg.norm(fwp)) > 1e-6:
            fw_proj.append(_unit(fwp))
    north_world = fw_proj[0] if fw_proj else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right_world = _unit(np.cross(north_world, up_world))
    north_world = _unit(np.cross(up_world, right_world))

    # One PIL image per view, taken from the tensor the geometry model was fed, so
    # that mask pixel coordinates line up with the depth and point maps.
    view_pils = []
    for vi in range(max_views):
        view_pils.append(_preprocessed_view_to_pil(pred_images, vi))

    # Candidate masks, keyed by (object, view), filled by the backend below.
    cand_cache: dict[tuple[str, int], list[dict[str, Any]]] = {}

    # One pass per view: SAM 3 turns each object name into masks, boxes and scores.
    if det_backend == "sam3_image":
        sam3_proc = _get_sam3_processor()
        for vi in range(max_views):
            img_pil = view_pils[vi]
            W, H = img_pil.size

            d_hw = np.asarray(depth_all[vi], dtype=np.float32)
            if d_hw.shape[0] != H or d_hw.shape[1] != W:
                raise ValueError(f"depth shape {tuple(d_hw.shape)} does not match image {(H, W)}")
            c_hw = None
            if conf_all is not None:
                c_hw = np.asarray(_normalize_depth_conf_hw(conf_all[vi]), dtype=np.float32)

            wp_hw3 = np.asarray(wp_all_np[vi], dtype=np.float32)
            if wp_hw3.ndim != 3 or int(wp_hw3.shape[2]) != 3:
                raise ValueError(f"world_points_from_depth[{vi}] must be HxWx3, got shape={tuple(wp_hw3.shape)}")
            if int(wp_hw3.shape[0]) != int(H) or int(wp_hw3.shape[1]) != int(W):
                raise ValueError(f"world_points_from_depth[{vi}] shape {tuple(wp_hw3.shape)} does not match image {(H, W)}")

            # SAM3's perflib emits bf16 fused matmul outputs from fc1 but fc2/norm
            # keep fp32 weights → dtype mismatch on torch 2.6. Wrap all SAM3 forwards
            # in bf16 autocast so every Linear/conv op matches the fused activations.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = sam3_proc.set_image(img_pil.convert("RGB"), state=None)
            for obj in object_names:
                prompt_text = str(obj_key_to_orig.get(obj, obj))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    st2 = sam3_proc.set_text_prompt(prompt=prompt_text, state=state)
                m = st2.get("masks")
                b = st2.get("boxes")
                s = st2.get("scores")
                if m is None or b is None or s is None:
                    continue
                if isinstance(m, torch.Tensor):
                    m = m.detach().cpu().float().numpy()
                if isinstance(b, torch.Tensor):
                    b = b.detach().cpu().float().numpy()
                if isinstance(s, torch.Tensor):
                    s = s.detach().cpu().float().numpy()
                m = np.asarray(m).astype(bool)
                b = np.asarray(b, dtype=np.float32)
                s = np.asarray(s, dtype=np.float32).reshape(-1)
                if m.ndim == 4 and m.shape[1] == 1:
                    m = m[:, 0]
                if m.ndim != 3 or b.ndim != 2 or int(b.shape[1]) != 4:
                    continue
                n = int(min(int(m.shape[0]), int(b.shape[0]), int(s.shape[0])))
                if n <= 0:
                    continue

                order = np.argsort(-s[:n])
                k_use = int(min(int(n), int(max(1, cfg.assoc_topk_per_view))))
                cands: list[dict[str, Any]] = []
                for rank_i, idx_i in enumerate(order[:k_use].tolist()):
                    mask_hw = np.asarray(m[int(idx_i)]).astype(bool)
                    yy, xx = np.nonzero(mask_hw)
                    if yy.size == 0:
                        continue

                    x0f, y0f, x1f, y1f = (float(x) for x in b[int(idx_i)].reshape(4))
                    x0 = int(np.clip(np.floor(x0f), 0, W - 1))
                    y0 = int(np.clip(np.floor(y0f), 0, H - 1))
                    x1 = int(np.clip(np.ceil(x1f), 0, W - 1))
                    y1 = int(np.clip(np.ceil(y1f), 0, H - 1))
                    if x1 <= x0 or y1 <= y0:
                        x0, y0, x1, y1 = _mask_to_xyxy(mask_hw)

                    d = d_hw[yy, xx].reshape(-1)
                    valid = np.isfinite(d)
                    cm = 0.0
                    if c_hw is not None:
                        c = c_hw[yy, xx].reshape(-1)
                        valid = valid & np.isfinite(c)
                        if float(cfg.min_depth_conf) > 0.0:
                            valid = valid & (c >= float(cfg.min_depth_conf))
                        if int(np.count_nonzero(valid)) > 0:
                            cm = float(np.mean(c[valid]))
                    d = d[valid]
                    if d.size < 16:
                        continue
                    lo = float(np.clip(float(cfg.depth_trim_lo_percentile), 0.0, 49.0))
                    hi = float(np.clip(float(cfg.depth_trim_hi_percentile), 51.0, 100.0))
                    d_lo = float(np.percentile(d, lo))
                    d_hi = float(np.percentile(d, hi))
                    d2 = d[(d >= d_lo) & (d <= d_hi)]
                    if d2.size < 8:
                        d2 = d
                    z_med = float(np.median(d2))

                    lifted = _robust_anchor_world_from_mask_pointmap(
                        mask_hw=mask_hw,
                        wp_hw3=wp_hw3,
                        conf_hw=c_hw,
                        ext_3x4_world_to_cam=ext_all[vi],
                        min_depth_conf=float(cfg.min_depth_conf),
                        stride=6,
                        max_points=5000,
                        camz_lo_percentile=float(cfg.depth_trim_lo_percentile),
                        camz_hi_percentile=float(cfg.depth_trim_hi_percentile),
                        min_points=32,
                    )
                    if lifted is None:
                        continue
                    p_world, lift_dbg = lifted
                    n_keep = int(lift_dbg.get("n_after_camz", 0) or 0)
                    sam3_score = float(s[int(idx_i)])
                    qual = float(max(1e-6, cm) * (float(max(1, n_keep)) ** 0.25) * float(max(1e-3, sam3_score)))
                    cands.append(
                        {
                            "mask": mask_hw,
                            "qual": float(qual),
                            "box_rank": int(rank_i),
                            "box_xyxy": (int(x0), int(y0), int(x1), int(y1)),
                            "p_world": p_world.reshape(3),
                            "u_med": float(np.median(xx)),
                            "v_med": float(np.median(yy)),
                            "z_med": float(z_med),
                            "lift": lift_dbg,
                            "sam3_score": float(sam3_score),
                        }
                    )
                if cands:
                    cands.sort(key=lambda d: float(d.get("qual", 0.0)), reverse=True)
                    cand_cache[(obj, int(vi))] = cands[: int(max(1, cfg.assoc_topk_per_view))]

    # SAM3 tracking backend: treat multi-view as pseudo-video and propagate text prompts across views.
    # Stage-1 behavior: we only use SAM3's own tracking; we reuse pseudo association/3D lifting unchanged.
    if det_backend == "sam3_track":
        vp = _get_sam3_video_predictor()

        def _iter_instances_from_sam3_outputs(outputs: Any) -> list[tuple[int | None, np.ndarray, np.ndarray, float]]:
            """
            Normalize SAM3 video outputs into a list of (obj_id, mask_hw, box_xyxy, score).
            Supports a few common shapes without relying on private internals.
            """
            inst: list[tuple[int | None, np.ndarray, np.ndarray, float]] = []
            if outputs is None:
                return inst
            if isinstance(outputs, dict):
                # Case 0: official SAM3 video postprocessed output format
                # See sam3/model/sam3_video_inference.py::_postprocess_output
                # outputs = {
                #   "out_obj_ids": np.ndarray[int64] (N,),
                #   "out_probs": np.ndarray[float32] (N,),
                #   "out_boxes_xywh": np.ndarray[float32] (N,4) normalized [x,y,w,h] in 0..1,
                #   "out_binary_masks": np.ndarray[bool] (N,H,W),
                #   ...
                # }
                if (
                    ("out_binary_masks" in outputs)
                    and ("out_boxes_xywh" in outputs)
                    and ("out_probs" in outputs)
                    and ("out_obj_ids" in outputs)
                ):
                    m = outputs.get("out_binary_masks")
                    b = outputs.get("out_boxes_xywh")
                    s = outputs.get("out_probs")
                    obj_ids = outputs.get("out_obj_ids")
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    if isinstance(b, torch.Tensor):
                        b = b.detach().cpu().numpy()
                    if isinstance(s, torch.Tensor):
                        s = s.detach().cpu().numpy()
                    if isinstance(obj_ids, torch.Tensor):
                        obj_ids = obj_ids.detach().cpu().numpy()
                    m = np.asarray(m)
                    b = np.asarray(b)
                    s = np.asarray(s).reshape(-1)
                    obj_ids = np.asarray(obj_ids).reshape(-1)
                    if m.ndim == 4 and m.shape[1] == 1:
                        m = m[:, 0]
                    if m.ndim == 3 and b.ndim == 2 and int(b.shape[1]) == 4:
                        n = int(min(int(m.shape[0]), int(b.shape[0]), int(s.shape[0]), int(obj_ids.shape[0])))
                        for i in range(n):
                            try:
                                oid = int(obj_ids[i])
                            except Exception:
                                oid = None
                            inst.append((oid, np.asarray(m[i]).astype(bool), np.asarray(b[i], dtype=np.float32).reshape(4), float(s[i])))
                    return inst

                # Case A: dict has masks/boxes/scores arrays
                if ("masks" in outputs) and ("boxes" in outputs) and ("scores" in outputs):
                    m = outputs.get("masks")
                    b = outputs.get("boxes")
                    s = outputs.get("scores")
                    obj_ids = outputs.get("obj_ids", outputs.get("object_ids", None))
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    if isinstance(b, torch.Tensor):
                        b = b.detach().cpu().numpy()
                    if isinstance(s, torch.Tensor):
                        s = s.detach().cpu().numpy()
                    m = np.asarray(m)
                    b = np.asarray(b)
                    s = np.asarray(s).reshape(-1)
                    if m.ndim == 4 and m.shape[1] == 1:
                        m = m[:, 0]
                    if m.ndim == 3 and b.ndim == 2 and int(b.shape[1]) == 4:
                        n = int(min(int(m.shape[0]), int(b.shape[0]), int(s.shape[0])))
                        ids_list: list[int | None] = [None] * n
                        if obj_ids is not None:
                            try:
                                ids_arr = np.asarray(obj_ids).reshape(-1)
                                for i in range(min(n, int(ids_arr.shape[0]))):
                                    ids_list[i] = int(ids_arr[i])
                            except Exception:
                                pass
                        for i in range(n):
                            inst.append((ids_list[i], np.asarray(m[i]).astype(bool), np.asarray(b[i], dtype=np.float32).reshape(4), float(s[i])))
                        return inst
                # Case B: dict maps obj_id -> per-object dict
                is_objid_map = all(isinstance(k, int | np.integer | str) for k in outputs.keys())
                if is_objid_map:
                    for k, v in outputs.items():
                        try:
                            oid = int(k)
                        except Exception:
                            oid = None
                        if isinstance(v, dict) and ("mask" in v or "masks" in v):
                            m = v.get("mask", v.get("masks"))
                            b = v.get("box", v.get("boxes"))
                            s = v.get("score", v.get("scores"))
                            if m is None or b is None or s is None:
                                continue
                            if isinstance(m, torch.Tensor):
                                m = m.detach().cpu().numpy()
                            if isinstance(b, torch.Tensor):
                                b = b.detach().cpu().numpy()
                            mm = np.asarray(m)
                            if mm.ndim == 4 and mm.shape[1] == 1:
                                mm = mm[:, 0]
                            if mm.ndim == 3:
                                mm = mm[0]
                            inst.append((oid, np.asarray(mm).astype(bool), np.asarray(b, dtype=np.float32).reshape(4), float(np.asarray(s).reshape(-1)[0])))
                    return inst
            # Case C: list of dicts
            if isinstance(outputs, list):
                for v in outputs:
                    if not isinstance(v, dict):
                        continue
                    m = v.get("mask", v.get("masks"))
                    b = v.get("box", v.get("boxes"))
                    s = v.get("score", v.get("scores"))
                    oid = v.get("obj_id", v.get("object_id", None))
                    if m is None or b is None or s is None:
                        continue
                    if isinstance(m, torch.Tensor):
                        m = m.detach().cpu().numpy()
                    if isinstance(b, torch.Tensor):
                        b = b.detach().cpu().numpy()
                    mm = np.asarray(m)
                    if mm.ndim == 4 and mm.shape[1] == 1:
                        mm = mm[:, 0]
                    if mm.ndim == 3:
                        mm = mm[0]
                    try:
                        oid2 = int(oid) if oid is not None else None
                    except Exception:
                        oid2 = None
                    inst.append((oid2, np.asarray(mm).astype(bool), np.asarray(b, dtype=np.float32).reshape(4), float(np.asarray(s).reshape(-1)[0])))
            return inst

        with tempfile.TemporaryDirectory(prefix="pseudo_sam3_vid_") as td:
            # Write frames as JPEG sequence
            for vi in range(max_views):
                out_path = os.path.join(td, f"{vi:05d}.jpg")
                view_pils[vi].save(out_path, format="JPEG", quality=95)

            # Start session
            resp = vp.handle_request({"type": "start_session", "resource_path": td})
            session_id = resp.get("session_id")
            if not session_id:
                raise RuntimeError("SAM3 start_session did not return session_id")

            seed_idx = int(np.clip(int(getattr(cfg, "sam3_seed_frame_idx", 0) or 0), 0, max(0, max_views - 1)))

            # IMPORTANT:
            # SAM3 video "add_prompt(text=...)" resets semantic prompt state internally, so multiple
            # sequential text prompts do NOT accumulate. If we add prompts for all objects first,
            # only the *last* prompt survives for propagation.
            #
            # Therefore we run (add_prompt -> propagate) sequentially per object and fill cand_cache.
            direction = str(getattr(cfg, "sam3_propagation_direction", "both") or "both")
            for obj in object_names:
                prompt_text = str(obj_key_to_orig.get(obj, obj))
                resp2 = vp.handle_request(
                    {
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": int(seed_idx),
                        "text": prompt_text,
                    }
                )
                # Use the immediate output on the prompted frame too (helps short sequences)
                inst0 = _iter_instances_from_sam3_outputs(resp2.get("outputs"))
                if inst0:
                    # run the same candidate construction logic for the seed frame
                    msg0 = {"frame_index": int(seed_idx), "outputs": resp2.get("outputs")}
                    for msg in [msg0]:
                        vi = int(msg.get("frame_index", -1))
                        if vi < 0 or vi >= int(max_views):
                            continue
                        instances = _iter_instances_from_sam3_outputs(msg.get("outputs"))
                        if not instances:
                            continue
                        # Prepare per-view geometry arrays
                        img_pil = view_pils[vi]
                        W, H = img_pil.size
                        d_hw = np.asarray(depth_all[vi], dtype=np.float32)
                        c_hw = None
                        if conf_all is not None:
                            c_hw = np.asarray(_normalize_depth_conf_hw(conf_all[vi]), dtype=np.float32)
                        wp_hw3 = np.asarray(wp_all_np[vi], dtype=np.float32)
                        triplets = [(m, b, float(s)) for (_oid, m, b, s) in instances]
                        triplets.sort(key=lambda t: float(t[2]), reverse=True)
                        triplets = triplets[: int(max(1, cfg.assoc_topk_per_view))]
                        cands: list[dict[str, Any]] = []
                        for rank_i, (mask_hw, box_xyxy, score) in enumerate(triplets):
                            mask_hw = np.asarray(mask_hw).astype(bool)
                            yy, xx = np.nonzero(mask_hw)
                            if yy.size == 0:
                                continue
                            box_arr = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
                            x0f, y0f, x1f, y1f = (float(x) for x in box_arr)
                            if (
                                float(box_arr[0]) >= -1e-3
                                and float(box_arr[1]) >= -1e-3
                                and float(box_arr[2]) >= 0.0
                                and float(box_arr[3]) >= 0.0
                                and float(box_arr[0]) <= 1.5
                                and float(box_arr[1]) <= 1.5
                                and float(box_arr[2]) <= 1.5
                                and float(box_arr[3]) <= 1.5
                                and float(box_arr[0] + box_arr[2]) <= 1.5
                                and float(box_arr[1] + box_arr[3]) <= 1.5
                            ):
                                xn, yn, wn, hn = (float(x) for x in box_arr)
                                x0f = xn * float(W)
                                y0f = yn * float(H)
                                x1f = (xn + wn) * float(W)
                                y1f = (yn + hn) * float(H)
                            x0 = int(np.clip(np.floor(x0f), 0, W - 1))
                            y0 = int(np.clip(np.floor(y0f), 0, H - 1))
                            x1 = int(np.clip(np.ceil(x1f), 0, W - 1))
                            y1 = int(np.clip(np.ceil(y1f), 0, H - 1))
                            if x1 <= x0 or y1 <= y0:
                                x0, y0, x1, y1 = _mask_to_xyxy(mask_hw)
                            d = d_hw[yy, xx].reshape(-1)
                            valid = np.isfinite(d)
                            cm = 0.0
                            if c_hw is not None:
                                c = c_hw[yy, xx].reshape(-1)
                                valid = valid & np.isfinite(c)
                                if float(cfg.min_depth_conf) > 0.0:
                                    valid = valid & (c >= float(cfg.min_depth_conf))
                                if int(np.count_nonzero(valid)) > 0:
                                    cm = float(np.mean(c[valid]))
                            d = d[valid]
                            if d.size < 16:
                                continue
                            lo = float(np.clip(float(cfg.depth_trim_lo_percentile), 0.0, 49.0))
                            hi = float(np.clip(float(cfg.depth_trim_hi_percentile), 51.0, 100.0))
                            d_lo = float(np.percentile(d, lo))
                            d_hi = float(np.percentile(d, hi))
                            d2 = d[(d >= d_lo) & (d <= d_hi)]
                            if d2.size < 8:
                                d2 = d
                            z_med = float(np.median(d2))
                            lifted = _robust_anchor_world_from_mask_pointmap(
                                mask_hw=mask_hw,
                                wp_hw3=wp_hw3,
                                conf_hw=c_hw,
                                ext_3x4_world_to_cam=ext_all[vi],
                                min_depth_conf=float(cfg.min_depth_conf),
                                stride=6,
                                max_points=5000,
                                camz_lo_percentile=float(cfg.depth_trim_lo_percentile),
                                camz_hi_percentile=float(cfg.depth_trim_hi_percentile),
                                min_points=32,
                            )
                            if lifted is None:
                                continue
                            p_world, lift_dbg = lifted
                            n_keep = int(lift_dbg.get("n_after_camz", 0) or 0)
                            qual = float(max(1e-6, cm) * (float(max(1, n_keep)) ** 0.25) * float(max(1e-3, score)))
                            cands.append(
                                {
                                    "mask": mask_hw,
                                    "qual": float(qual),
                                    "box_rank": int(rank_i),
                                    "box_xyxy": (int(x0), int(y0), int(x1), int(y1)),
                                    "p_world": p_world.reshape(3),
                                    "u_med": float(np.median(xx)),
                                    "v_med": float(np.median(yy)),
                                    "z_med": float(z_med),
                                    "lift": lift_dbg,
                                    "sam3_score": float(score),
                                }
                            )
                        if cands:
                            cands.sort(key=lambda d: float(d.get("qual", 0.0)), reverse=True)
                            cand_cache[(obj, int(vi))] = cands[: int(max(1, cfg.assoc_topk_per_view))]

                # Propagate across views for this object prompt
                for msg in vp.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                        "start_frame_index": int(seed_idx),
                        "max_frame_num_to_track": int(max_views),
                    }
                ):
                    vi = int(msg.get("frame_index", -1))
                    if vi < 0 or vi >= int(max_views):
                        continue
                    instances = _iter_instances_from_sam3_outputs(msg.get("outputs"))
                    if not instances:
                        continue
                    # Prepare per-view geometry arrays
                    img_pil = view_pils[vi]
                    W, H = img_pil.size
                    d_hw = np.asarray(depth_all[vi], dtype=np.float32)
                    c_hw = None
                    if conf_all is not None:
                        c_hw = np.asarray(_normalize_depth_conf_hw(conf_all[vi]), dtype=np.float32)
                    wp_hw3 = np.asarray(wp_all_np[vi], dtype=np.float32)
                    triplets = [(m, b, float(s)) for (_oid, m, b, s) in instances]
                    triplets.sort(key=lambda t: float(t[2]), reverse=True)
                    triplets = triplets[: int(max(1, cfg.assoc_topk_per_view))]
                    cands: list[dict[str, Any]] = []
                    for rank_i, (mask_hw, box_xyxy, score) in enumerate(triplets):
                        mask_hw = np.asarray(mask_hw).astype(bool)
                        yy, xx = np.nonzero(mask_hw)
                        if yy.size == 0:
                            continue
                        box_arr = np.asarray(box_xyxy, dtype=np.float32).reshape(4)
                        x0f, y0f, x1f, y1f = (float(x) for x in box_arr)
                        # SAM3 video outputs use normalized XYWH boxes ("out_boxes_xywh") in 0..1.
                        # If we detect that format, convert to pixel-space XYXY here for consistent downstream logic.
                        # Heuristic: all coords in [0, ~1] and (x+w) / (y+h) stay within bounds.
                        if (
                            float(box_arr[0]) >= -1e-3
                            and float(box_arr[1]) >= -1e-3
                            and float(box_arr[2]) >= 0.0
                            and float(box_arr[3]) >= 0.0
                            and float(box_arr[0]) <= 1.5
                            and float(box_arr[1]) <= 1.5
                            and float(box_arr[2]) <= 1.5
                            and float(box_arr[3]) <= 1.5
                            and float(box_arr[0] + box_arr[2]) <= 1.5
                            and float(box_arr[1] + box_arr[3]) <= 1.5
                        ):
                            xn, yn, wn, hn = (float(x) for x in box_arr)
                            x0f = xn * float(W)
                            y0f = yn * float(H)
                            x1f = (xn + wn) * float(W)
                            y1f = (yn + hn) * float(H)
                        x0 = int(np.clip(np.floor(x0f), 0, W - 1))
                        y0 = int(np.clip(np.floor(y0f), 0, H - 1))
                        x1 = int(np.clip(np.ceil(x1f), 0, W - 1))
                        y1 = int(np.clip(np.ceil(y1f), 0, H - 1))
                        if x1 <= x0 or y1 <= y0:
                            x0, y0, x1, y1 = _mask_to_xyxy(mask_hw)

                        d = d_hw[yy, xx].reshape(-1)
                        valid = np.isfinite(d)
                        cm = 0.0
                        if c_hw is not None:
                            c = c_hw[yy, xx].reshape(-1)
                            valid = valid & np.isfinite(c)
                            if float(cfg.min_depth_conf) > 0.0:
                                valid = valid & (c >= float(cfg.min_depth_conf))
                            if int(np.count_nonzero(valid)) > 0:
                                cm = float(np.mean(c[valid]))
                        d = d[valid]
                        if d.size < 16:
                            continue
                        lo = float(np.clip(float(cfg.depth_trim_lo_percentile), 0.0, 49.0))
                        hi = float(np.clip(float(cfg.depth_trim_hi_percentile), 51.0, 100.0))
                        d_lo = float(np.percentile(d, lo))
                        d_hi = float(np.percentile(d, hi))
                        d2 = d[(d >= d_lo) & (d <= d_hi)]
                        if d2.size < 8:
                            d2 = d
                        z_med = float(np.median(d2))

                        lifted = _robust_anchor_world_from_mask_pointmap(
                            mask_hw=mask_hw,
                            wp_hw3=wp_hw3,
                            conf_hw=c_hw,
                            ext_3x4_world_to_cam=ext_all[vi],
                            min_depth_conf=float(cfg.min_depth_conf),
                            stride=6,
                            max_points=5000,
                            camz_lo_percentile=float(cfg.depth_trim_lo_percentile),
                            camz_hi_percentile=float(cfg.depth_trim_hi_percentile),
                            min_points=32,
                        )
                        if lifted is None:
                            continue
                        p_world, lift_dbg = lifted
                        n_keep = int(lift_dbg.get("n_after_camz", 0) or 0)
                        qual = float(max(1e-6, cm) * (float(max(1, n_keep)) ** 0.25) * float(max(1e-3, score)))
                        cands.append(
                            {
                                "mask": mask_hw,
                                "qual": float(qual),
                                "box_rank": int(rank_i),
                                "box_xyxy": (int(x0), int(y0), int(x1), int(y1)),
                                "p_world": p_world.reshape(3),
                                "u_med": float(np.median(xx)),
                                "v_med": float(np.median(yy)),
                                "z_med": float(z_med),
                                "lift": lift_dbg,
                                "sam3_score": float(score),
                            }
                        )
                    if cands:
                        cands.sort(key=lambda d: float(d.get("qual", 0.0)), reverse=True)
                        cand_cache[(obj, int(vi))] = cands[: int(max(1, cfg.assoc_topk_per_view))]

            # Close session
            vp.handle_request({"type": "close_session", "session_id": session_id})

    # Final selection caches for debug render
    mask_cache: dict[tuple[str, int], object] = {}
    box_cache: dict[tuple[str, int], tuple[int, int, int, int]] = {}
    label_cache: dict[tuple[str, int], str] = {}
    reproj_cache: dict[tuple[str, int], tuple[float, float]] = {}
    stats_cache: dict[tuple[str, int], str] = {}

    # Association per object: pick candidates closest to global median world point
    obj_world_pts: dict[str, list[np.ndarray]] = {o: [] for o in object_names}
    assoc_debug: dict[str, Any] = {}
    vis_debug: dict[str, Any] = {}
    for obj in object_names:
        # Visibility-aware association: brute force over (K+1)^V combinations (K<=3, V<=4).
        # Allow "None" per view to drop a detection (helps reject false positives).
        per_view_opts: list[list[dict[str, Any] | None]] = []
        for vi in range(max_views):
            cands = list(cand_cache.get((obj, int(vi)), []))
            # None option first (meaning: treat as not detected in this view)
            per_view_opts.append([None] + cands)

        # Early exit if no candidates anywhere
        if not any(len(opts) > 1 for opts in per_view_opts):
            continue

        # penalties (tuned to be simple/robust)
        P_BEHIND_DETECTED = 50.0   # if selected but fused point is behind camera: very unlikely
        P_INFRONT_MISSED = 2.0     # if not selected but fused point is in-frame and in-front: could be occluded, weak penalty
        P_OUTSIDE_DETECTED = 5.0   # if selected but fused point projects out of frame: likely wrong detection
        P_BOX_MISS = 4.0           # if selected but fused projection not inside selected box
        W_REPROJ = 12.0            # reprojection error weight (normalized by image diag)
        W_DIST = 1.0               # world consistency weight
        W_QUAL = 0.1               # quality reward weight

        # image sizes for FOV checks
        view_wh = []
        for vi in range(max_views):
            W, H = view_pils[vi].size
            view_wh.append((int(W), int(H)))

        best_score = None
        best_choice: list[dict[str, Any] | None] | None = None
        best_fused: np.ndarray | None = None
        best_view_debug: list[dict[str, Any]] | None = None

        def _iter_choices(i: int, cur: list[dict[str, Any] | None], options=per_view_opts):
            if i == max_views:
                yield list(cur)
                return
            for opt in options[i]:
                cur.append(opt)
                yield from _iter_choices(i + 1, cur)
                cur.pop()

        for choice in _iter_choices(0, []):
            chosen = [c for c in choice if c is not None]
            if len(chosen) == 0:
                continue
            Pw = np.stack([np.asarray(c["p_world"], dtype=np.float64).reshape(3) for c in chosen], axis=0)
            fused = np.median(Pw, axis=0)

            # consistency term (sum distances to fused)
            dist_term = float(np.sum(np.linalg.norm(Pw - fused[None, :], axis=1)))
            qual_term = float(np.sum([float(c.get("qual", 0.0) or 0.0) for c in chosen]))

            score = W_DIST * dist_term - W_QUAL * qual_term

            per_view_dbg = []
            for vi in range(max_views):
                sel = choice[vi]
                xyz_cam = _world_to_cam_xyz(xyz_world=fused, ext_3x4_world_to_cam=ext_all[vi])
                z = float(xyz_cam[2])
                u_px, v_px = _project_cam_to_px(xyz_cam=xyz_cam, K_3x3=intr_all[vi])
                in_front = (z > 1e-6)
                in_frame = bool(in_front and _in_image(u_px, v_px, view_wh[vi][0], view_wh[vi][1]))

                # penalties
                pen = 0.0
                if sel is not None and z <= 0.0:
                    pen += P_BEHIND_DETECTED
                if sel is None and in_frame:
                    pen += P_INFRONT_MISSED
                if sel is not None and (in_front and (not in_frame)):
                    pen += P_OUTSIDE_DETECTED
                reproj_err = None
                box_contains = None
                if sel is not None:
                    du = float(u_px) - float(sel.get("u_med", 0.0))
                    dv = float(v_px) - float(sel.get("v_med", 0.0))
                    reproj_err = float((du * du + dv * dv) ** 0.5)
                    diag = float((view_wh[vi][0] ** 2 + view_wh[vi][1] ** 2) ** 0.5)
                    diag = diag if diag > 1e-6 else 1.0
                    pen += W_REPROJ * (reproj_err / diag)
                    bx = sel.get("box_xyxy")
                    if bx is not None:
                        x1, y1, x2, y2 = (int(v) for v in bx)
                        box_contains = bool(
                            (float(u_px) >= float(x1))
                            and (float(u_px) <= float(x2))
                            and (float(v_px) >= float(y1))
                            and (float(v_px) <= float(y2))
                        )
                        if in_front and (not box_contains):
                            pen += P_BOX_MISS

                score += pen
                per_view_dbg.append(
                    {
                        "view": int(vi),
                        "selected": bool(sel is not None),
                        "z_cam": float(z),
                        "u_px": float(u_px),
                        "v_px": float(v_px),
                        "in_frame": bool(in_frame),
                        "penalty": float(pen),
                        "box_rank": (None if sel is None else int(sel.get("box_rank", -1))),
                        "qual": (None if sel is None else float(sel.get("qual", 0.0))),
                        "reproj_err_px": reproj_err,
                        "box_contains": box_contains,
                    }
                )

            if best_score is None or float(score) < float(best_score):
                best_score = float(score)
                best_choice = list(choice)
                best_fused = fused
                best_view_debug = per_view_dbg

        if best_choice is None or best_fused is None:
            continue

        picks = {}
        for vi in range(max_views):
            c = best_choice[vi]
            if c is None:
                continue
            picks[int(vi)] = c

        for vi, c in picks.items():
            obj_world_pts[obj].append(np.asarray(c["p_world"], dtype=np.float64).reshape(3))
            mask_cache[(obj, int(vi))] = c["mask"]
            box_cache[(obj, int(vi))] = tuple(int(x) for x in c["box_xyxy"])  # type: ignore[assignment]
            label_cache[(obj, int(vi))] = f"{obj_key_to_orig.get(obj, obj)} (rank {int(c.get('box_rank', -1))}, q={float(c.get('qual', 0.0)):.3f})"

        if best_view_debug is not None:
            for d in best_view_debug:
                vi = int(d.get("view", -1))
                if vi < 0 or vi >= max_views:
                    continue
                if not bool(d.get("selected", False)):
                    continue
                reproj_cache[(obj, vi)] = (float(d.get("u_px", 0.0)), float(d.get("v_px", 0.0)))
                zc = d.get("z_cam")
                inf = d.get("in_frame")
                err = d.get("reproj_err_px")
                if zc is not None and inf is not None and err is not None:
                    stats_cache[(obj, vi)] = f"z={float(zc):.2f} in={bool(inf)} err={float(err):.1f}px"

        assoc_debug[obj] = {
            "n_views_with_cands": int(sum(1 for i in range(max_views) if cand_cache.get((obj, int(i))))),
            "n_views_picked": int(len(picks)),
            "best_score": float(best_score) if best_score is not None else None,
        }
        vis_debug[obj] = {
            "fused_world": [float(best_fused[0]), float(best_fused[1]), float(best_fused[2])],
            "per_view": best_view_debug or [],
        }

    objects_built = [o for o in object_names if len(obj_world_pts.get(o) or []) > 0]
    if not objects_built:
        empty = {"objects": [], "views": []}
        desc = "[Cognitive Map Format]\n- Empty pseudo cogmap (no objects built)\n"
        return desc + "\n" + str(empty), {
            "cogmap": empty,
            "cogmap_description": desc,
            "pseudo_debug": {"association": assoc_debug},
        }

    # Fuse object world positions (median)
    obj_world = {o: np.median(np.stack(obj_world_pts[o], axis=0), axis=0) for o in objects_built}

    # Camera centers
    cam_centers = [np.asarray(_camera_center_world(ext_all[i]), dtype=np.float64).reshape(3) for i in range(max_views)]

    # Rotation-degeneracy detection (cheating-free: uses only VGGT output).
    # When all views share the same camera position (e.g. MindCube `rotation_*`
    # items), VGGT's multi-view solver is ill-posed and returns spurious
    # baselines. Detect when camera position spread is much smaller than the
    # object cloud spread, then collapse all view positions to their mean.
    if len(cam_centers) >= 2 and len(objects_built) >= 2:
        cam_arr = np.stack(cam_centers, axis=0)              # (V, 3)
        obj_arr = np.stack([obj_world[o] for o in objects_built], axis=0)  # (N, 3)
        cam_spread = float(np.linalg.norm(cam_arr.std(axis=0)))
        obj_spread = float(np.linalg.norm(obj_arr.std(axis=0))) + 1e-6
        if cam_spread / obj_spread < 0.10:
            mean_center = cam_arr.mean(axis=0)
            cam_centers = [mean_center.copy() for _ in range(max_views)]

    # Origin: anchor object (most picked) if available, else median of objects
    view_hits = {o: int(sum(1 for _ in obj_world_pts.get(o, []))) for o in objects_built}
    anchor = sorted(objects_built, key=lambda o: (-int(view_hits.get(o, 0)), o))[0]
    origin_world = np.asarray(obj_world.get(anchor), dtype=np.float64).reshape(3)

    # Project to BEV u,v
    obj_uv = {}
    for o in objects_built:
        d = obj_world[o] - origin_world
        obj_uv[o] = (float(np.dot(d, right_world)), float(np.dot(d, north_world)))
    cam_uv = []
    for c in cam_centers:
        d = c - origin_world
        cam_uv.append((float(np.dot(d, right_world)), float(np.dot(d, north_world))))

    # Fit-to-grid per axis (objects+cameras), centered at anchor
    uv_fit = list(obj_uv.values()) + [uv for uv in cam_uv if np.isfinite(uv[0]) and np.isfinite(uv[1])]
    us = np.asarray([u for (u, _v) in uv_fit], dtype=np.float64)
    vs = np.asarray([v for (_u, v) in uv_fit], dtype=np.float64)
    med_u, med_v = 0.0, 0.0  # anchor-centered in BEV already
    u10, u90 = np.percentile(us, [10.0, 90.0])
    v10, v90 = np.percentile(vs, [10.0, 90.0])
    half_u = float(max(abs(u90 - med_u), abs(med_u - u10), 1e-6))
    half_v = float(max(abs(v90 - med_v), abs(med_v - v10), 1e-6))
    scale_u = half_u / 4.0
    scale_v = half_v / 4.0

    def _to_grid(u: float, v: float) -> list[int]:
        x = int(round(((float(u) - med_u) / max(1e-6, scale_u)) + 5.0))
        y = int(round(5.0 - ((float(v) - med_v) / max(1e-6, scale_v))))
        return [max(0, min(9, x)), max(0, min(9, y))]

    objects_out = [{"name": obj_key_to_orig.get(o, o), "position": _to_grid(*obj_uv[o])} for o in objects_built]

    # Cameras: position from cam_uv, facing from rotation projected onto ground
    views_out: list[dict[str, Any]] = []
    for i in range(max_views):
        u, v = cam_uv[i]
        fw = _unit(_camera_forward_world(ext_all[i]))
        fwp = fw - float(np.dot(fw, up_world)) * up_world
        fr = float(np.dot(fwp, right_world))
        fn = float(np.dot(fwp, north_world))
        facing = _quantize_dir_2d(fr, fn)
        views_out.append({"name": f"Image {i+1}", "position": _to_grid(float(u), float(v)), "facing": facing})

    cogmap = {"objects": objects_out, "views": views_out}
    desc = (
        "[Cognitive Map Format]\n"
        "- The map uses a 10x10 grid where [0,0] is at the top-left corner and [9,9] is at the bottom-right corner in the bird's view\n"
        "- The map was estimated from the images themselves, so positions are approximate.\n"
    )
    import json
    txt = desc + "\n" + json.dumps(cogmap, ensure_ascii=False, indent=2)

    # Optional detection debug render
    pseudo_det_render_path = None
    pseudo_sam_render_path = None
    if bool(cfg.render_detection_debug) and isinstance(cfg.render_output_dir, str) and str(cfg.render_output_dir).strip():
        import hashlib
        import os

        os.makedirs(str(cfg.render_output_dir), exist_ok=True)
        sid = str(cfg.sample_id or "")
        key = hashlib.sha1(sid.encode("utf-8")).hexdigest()[:12] if sid else "unknown"
        img = _render_detection_debug_grid(
            view_pils=view_pils,
            selected_masks=mask_cache,
            selected_boxes_xyxy=box_cache,
            selected_labels=label_cache,
            selected_reproj_px=reproj_cache,
            selected_stats=stats_cache,
            object_names=object_names,
            obj_key_to_orig=obj_key_to_orig,
            max_views=max_views,
            render_size=int(cfg.render_size),
        )
        if img is not None:
            pseudo_det_render_path = os.path.join(str(cfg.render_output_dir), f"pseudo_det_{key}_{int(cfg.render_size)}.png")
            img.save(pseudo_det_render_path)

        # Extra debug render: the masks that enter association, per object per view.
        title3 = f"SAM 3 masks ({det_backend}; before association)"

        img3 = _render_segment_masks_grid(
            view_pils=view_pils,
            object_names=object_names,
            obj_key_to_orig=obj_key_to_orig,
            cand_cache=cand_cache,
            max_views=max_views,
            render_size=int(cfg.render_size),
            title=title3,
        )
        if img3 is not None:
            pseudo_sam_render_path = os.path.join(
                str(cfg.render_output_dir), f"pseudo_sam_{key}_{int(cfg.render_size)}.png"
            )
            img3.save(pseudo_sam_render_path)

    meta = {
        "cogmap": cogmap,
        "cogmap_description": desc,
        # Marks which backend produced the map.
        "det_backend": str(det_backend),
        "pseudo_det_render_path": pseudo_det_render_path,
        "pseudo_sam_render_path": pseudo_sam_render_path,
        "pseudo_debug": {
            "association": assoc_debug,
            "visibility": vis_debug,
            "anchor_object": obj_key_to_orig.get(anchor, anchor),
            "origin_world": [float(origin_world[0]), float(origin_world[1]), float(origin_world[2])],
            "right_world": [float(right_world[0]), float(right_world[1]), float(right_world[2])],
            "north_world": [float(north_world[0]), float(north_world[1]), float(north_world[2])],
            "up_world": [float(up_world[0]), float(up_world[1]), float(up_world[2])],
            "scale_u": float(scale_u),
            "scale_v": float(scale_v),
        },
    }
    return txt, meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PseudoCogmapConfig",
    "build_pseudo_cogmap",
]

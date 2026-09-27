"""
Geometry helpers shared by the estimated-cognitive-map pipeline: camera pose
extraction from VGGT extrinsics, depth-confidence normalisation, and direction
quantisation onto the grid's four compass directions.
"""

from __future__ import annotations

from typing import Any


def _preprocessed_view_to_pil(pred_images: Any, view_idx: int) -> object:
    import numpy as np
    from PIL import Image

    arr = pred_images[view_idx]
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected image shape for view {view_idx}: {arr.shape}")
    if arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Unexpected image channels for view {view_idx}: {arr.shape}")
    img_u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(img_u8, mode="RGB")


def _normalize_depth_conf_hw(conf_hw_any: object) -> object:
    import numpy as np

    c = np.asarray(conf_hw_any)
    if c.ndim != 2:
        return c.astype(np.float32)
    cf = c.astype(np.float32, copy=False)
    mx = float(np.nanmax(cf)) if cf.size > 0 else 0.0
    if mx > 1.5:
        # Likely 0..255 or similar
        cf = cf / max(mx, 1e-6)
    return np.clip(cf, 0.0, 1.0).astype(np.float32, copy=False)


def _unit(v: object, eps: float = 1e-12) -> object:
    import numpy as np

    x = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n < eps:
        return np.zeros((3,), dtype=np.float64)
    return (x / n).astype(np.float64)


def _camera_center_world(ext_3x4_world_to_cam: object) -> object:
    import numpy as np

    ext = np.asarray(ext_3x4_world_to_cam, dtype=np.float64).reshape(3, 4)
    R = ext[:, :3]
    t = ext[:, 3]
    return (-R.T @ t).astype(np.float64)


def _camera_forward_world(ext_3x4_world_to_cam: object) -> object:
    import numpy as np

    ext = np.asarray(ext_3x4_world_to_cam, dtype=np.float64).reshape(3, 4)
    R = ext[:, :3]
    # camera +z axis in world
    return (R.T[:, 2]).astype(np.float64)


def _camera_up_world(ext_3x4_world_to_cam: object) -> object:
    import numpy as np

    ext = np.asarray(ext_3x4_world_to_cam, dtype=np.float64).reshape(3, 4)
    R = ext[:, :3]
    # OpenCV y is down, so up is -y_cam
    return (-(R.T[:, 1])).astype(np.float64)


def _quantize_dir_2d(v_right: float, v_north: float) -> str:
    # north maps to "up" in grid, right maps to "right"
    ar = abs(float(v_right))
    an = abs(float(v_north))
    if an >= ar:
        return "up" if float(v_north) >= 0 else "down"
    return "right" if float(v_right) >= 0 else "left"


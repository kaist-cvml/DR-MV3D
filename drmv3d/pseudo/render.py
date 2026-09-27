"""
Bird's-eye renderer for a cognitive map: a 10x10 grid with the objects, the view
markers and their facing arrows. For inspecting a map by eye; nothing in the
pipeline feeds a rendered map to the model.

Both map formats are accepted: {"objects": [...], "views": [...]}, and the flat
{name: {"position": [x, y], "facing": ...}, ...}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BEVCogMapRenderConfig:
    render_size: int = 512
    grid_size: int = 10
    pad: int = 24
    bg_rgb: tuple[int, int, int] = (255, 255, 255)
    grid_rgb: tuple[int, int, int] = (210, 210, 210)
    axis_rgb: tuple[int, int, int] = (60, 60, 60)
    obj_radius_px: int = 14
    view_radius_px: int = 11
    font_size: int = 16
    show_labels: bool = True
    show_legend: bool = True


_PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("RED", (220, 50, 50)),
    ("GREEN", (50, 180, 70)),
    ("BLUE", (50, 90, 200)),
    ("ORANGE", (220, 140, 40)),
    ("PURPLE", (150, 70, 200)),
    ("CYAN", (40, 180, 190)),
    ("MAGENTA", (220, 70, 160)),
    ("YELLOW", (220, 200, 60)),
    ("BROWN", (140, 90, 40)),
    ("GRAY", (120, 120, 120)),
    ("BLACK", (30, 30, 30)),
]


def _get_font(size: int):
    # Import lazily so this file can be imported in environments without pillow.
    from PIL import ImageFont

    try:
        # Many envs don't have system fonts; PIL default is fine.
        return ImageFont.truetype("DejaVuSans.ttf", int(size))
    except Exception:
        return ImageFont.load_default()


def _as_xy(x: Any) -> tuple[float, float] | None:
    try:
        if isinstance(x, list | tuple) and len(x) >= 2:
            return float(x[0]), float(x[1])
    except Exception:
        return None
    return None


def _normalize_facing(facing: Any) -> str | None:
    if facing is None:
        return None
    if isinstance(facing, list):
        if not facing:
            return None
        facing = facing[0]
    if not isinstance(facing, str):
        return None
    s = facing.strip().lower()
    return s or None


def _iter_objects_and_views(cogmap: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns normalized (objects, views) lists.
    """
    objects: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []

    if not isinstance(cogmap, dict):
        return objects, views

    # Complex format
    if isinstance(cogmap.get("objects"), list):
        for o in cogmap.get("objects") or []:
            if isinstance(o, dict):
                objects.append(o)
    if isinstance(cogmap.get("views"), list):
        for v in cogmap.get("views") or []:
            if isinstance(v, dict):
                views.append(v)

    if objects or views:
        return objects, views

    # Simple format: {name: {...}}
    for k, v in cogmap.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if not isinstance(v, dict):
            continue
        objects.append({"name": k.strip(), "position": v.get("position"), "facing": v.get("facing")})
    return objects, views


def _arrow_vector(facing: str | None) -> tuple[float, float]:
    """
    Return a small direction vector in pixel space.
    Facing semantics follow the cogmap convention:
    - up: decreasing y
    - right: increasing x
    - down: increasing y
    - left: decreasing x
    For inner/outer we draw a small circle marker only (no arrow).
    """
    if not facing:
        return (0.0, 0.0)
    if facing == "up":
        return (0.0, -1.0)
    if facing == "right":
        return (1.0, 0.0)
    if facing == "down":
        return (0.0, 1.0)
    if facing == "left":
        return (-1.0, 0.0)
    # inner/outer or unknown
    return (0.0, 0.0)


def render_bev_cogmap_image(
    cogmap: dict[str, Any],
    *,
    cfg: BEVCogMapRenderConfig | None = None,
    object_color_map: dict[str, tuple[int, int, int]] | None = None,
    view_color_map: dict[str, tuple[int, int, int]] | None = None,
) -> object:
    """
    Returns a PIL.Image RGB.
    """
    from PIL import Image, ImageDraw

    cfg = cfg or BEVCogMapRenderConfig()
    W = int(cfg.render_size)
    n = int(cfg.grid_size)

    img = Image.new("RGB", (W, W), cfg.bg_rgb)
    draw = ImageDraw.Draw(img)
    font = _get_font(int(cfg.font_size))

    pad = int(cfg.pad)
    cell = float((W - 2 * pad) / float(n))

    # Grid lines
    for i in range(n + 1):
        x = pad + i * cell
        draw.line([(x, pad), (x, W - pad)], fill=cfg.grid_rgb, width=1)
        y = pad + i * cell
        draw.line([(pad, y), (W - pad, y)], fill=cfg.grid_rgb, width=1)

    # Axis labels
    draw.text((pad, 4), "BEV grid (0,0)=top-left, (9,9)=bottom-right", fill=cfg.axis_rgb, font=font)

    def _rgb_for_name(name: str, is_view: bool) -> tuple[int, int, int]:
        if is_view and isinstance(view_color_map, dict) and name in view_color_map:
            return tuple(view_color_map[name])
        if (not is_view) and isinstance(object_color_map, dict) and name in object_color_map:
            return tuple(object_color_map[name])
        idx = abs(hash(name)) % len(_PALETTE)
        return _PALETTE[idx][1]

    objects, views = _iter_objects_and_views(cogmap)

    def _xy_to_px(xy: tuple[float, float]) -> tuple[float, float]:
        x, y = float(xy[0]), float(xy[1])
        # clamp to grid
        x = max(0.0, min(float(n - 1), x))
        y = max(0.0, min(float(n - 1), y))
        cx = pad + (x + 0.5) * cell
        cy = pad + (y + 0.5) * cell
        return cx, cy

    # Objects
    for o in objects:
        nm = str(o.get("name") or "").strip()
        if not nm:
            continue
        pos = _as_xy(o.get("position"))
        if pos is None:
            continue
        facing = _normalize_facing(o.get("facing"))
        rgb = _rgb_for_name(nm, False)
        cx, cy = _xy_to_px(pos)
        r = int(cfg.obj_radius_px)
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=rgb, outline=(0, 0, 0), width=2)
        if bool(cfg.show_labels):
            draw.text((cx + r + 3, cy - r), nm, fill=(0, 0, 0), font=font)

        # Facing arrow (if directional)
        vx, vy = _arrow_vector(facing)
        if abs(vx) > 0.0 or abs(vy) > 0.0:
            L = max(10, int(r * 1.4))
            x2 = cx + vx * L
            y2 = cy + vy * L
            draw.line([(cx, cy), (x2, y2)], fill=(0, 0, 0), width=3)
            # arrow head
            ah = 6
            if vx == 0.0 and vy == -1.0:  # up
                tri = [(x2, y2), (x2 - ah, y2 + ah), (x2 + ah, y2 + ah)]
            elif vx == 1.0 and vy == 0.0:  # right
                tri = [(x2, y2), (x2 - ah, y2 - ah), (x2 - ah, y2 + ah)]
            elif vx == 0.0 and vy == 1.0:  # down
                tri = [(x2, y2), (x2 - ah, y2 - ah), (x2 + ah, y2 - ah)]
            else:  # left
                tri = [(x2, y2), (x2 + ah, y2 - ah), (x2 + ah, y2 + ah)]
            draw.polygon(tri, fill=(0, 0, 0))
        elif facing in ("inner", "outer"):
            # inner/outer: draw a small ring marker
            rr = max(4, int(r * 0.35))
            draw.ellipse([(cx - rr, cy - rr), (cx + rr, cy + rr)], outline=(0, 0, 0), width=2)

    # Views
    for v in views:
        nm = str(v.get("name") or "").strip()
        if not nm:
            continue
        pos = _as_xy(v.get("position"))
        if pos is None:
            continue
        facing = _normalize_facing(v.get("facing"))
        rgb = _rgb_for_name(nm, True)
        cx, cy = _xy_to_px(pos)
        r = int(cfg.view_radius_px)
        draw.rectangle([(cx - r, cy - r), (cx + r, cy + r)], fill=rgb, outline=(0, 0, 0), width=2)
        if bool(cfg.show_labels):
            draw.text((cx + r + 3, cy - r), nm, fill=(0, 0, 0), font=font)

        vx, vy = _arrow_vector(facing)
        if abs(vx) > 0.0 or abs(vy) > 0.0:
            L = max(10, int(r * 1.6))
            x2 = cx + vx * L
            y2 = cy + vy * L
            draw.line([(cx, cy), (x2, y2)], fill=(0, 0, 0), width=3)

    # Legend
    if bool(cfg.show_legend):
        # bottom-left legend box
        legend = "Legend: circle=object, square=view; arrow=facing"
        draw.text((pad, W - pad + 2 - int(cfg.font_size)), legend, fill=(0, 0, 0), font=font)

    return img






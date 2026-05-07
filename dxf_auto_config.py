from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ezdxf

from cad_import import DXFImportConfig, FloorRegion

_INSUNITS_SCALE = {
    0: 1.0,     # Unitless
    1: 0.0254,  # Inches
    2: 0.3048,  # Feet
    4: 0.001,   # Millimeters
    5: 0.01,    # Centimeters
    6: 1.0,     # Meters
}

CN_MAP = {"首": 1, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
EXIT_KEYWORDS = (
    "安全出口", "疏散出口", "楼梯出口", "安全门", "逃生出口",
    "EMERGENCY EXIT", "FIRE EXIT", "EXIT",
)
STRUCT_ENTITY_TYPES = {"LINE", "LWPOLYLINE", "ARC", "POLYLINE", "CIRCLE"}
STRUCT_LAYER_SKIP_PREFIX = (
    "PUB_DIM", "PUB_TAB", "PUB_TEXT", "A_TEXT", "DIM_", "TK_", "图框", "AXIS",
    "T天花", "D铺砖", "FUR", "WINDOW", "COLUMN",
)


def _read_doc(dxf_path: str) -> "ezdxf.document.Drawing":
    try:
        return ezdxf.readfile(dxf_path)
    except Exception:
        return ezdxf.readfile(dxf_path, encoding="gbk")


def _get_text_content(ent) -> str:
    try:
        if ent.dxftype() == "MTEXT":
            raw = ""
            for attr in ("text", "plain_mtext"):
                val = getattr(ent, attr, None)
                if val is None:
                    continue
                raw = val() if callable(val) else str(val)
                if raw:
                    break
            return re.sub(r"\{[^}]*\}|\\[A-Za-z0-9;]+", "", raw).strip()
        return (ent.dxf.text or "").strip()
    except Exception:
        return ""


def _iter_text_entities(msp) -> Iterable[Tuple[float, float, str]]:
    for ent in msp:
        if ent.dxftype() not in ("TEXT", "MTEXT"):
            continue
        try:
            x = float(ent.dxf.insert.x)
            y = float(ent.dxf.insert.y)
        except Exception:
            continue
        text = _get_text_content(ent)
        if text:
            yield x, y, text


def _parse_floor_text(text: str) -> Optional[int]:
    t = text.strip().upper()
    for pat in (
        r"\b(?:FLOOR|LEVEL)\s*([0-9]{1,2})\b",
        r"\b([0-9]{1,2})\s*F\b",
        r"\bF\s*([0-9]{1,2})\b",
    ):
        m = re.search(pat, t)
        if m:
            return int(m.group(1))

    m = re.search(r"(首|[一二三四五六七八九十])\s*层", text)
    if m:
        return CN_MAP.get(m.group(1))

    return None


def _estimate_bbox(msp) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []
    for ent in msp:
        tp = ent.dxftype()
        try:
            if tp == "LINE":
                xs.extend([float(ent.dxf.start.x), float(ent.dxf.end.x)])
                ys.extend([float(ent.dxf.start.y), float(ent.dxf.end.y)])
            elif tp == "LWPOLYLINE":
                pts = list(ent.get_points("xy"))
                xs.extend(float(p[0]) for p in pts)
                ys.extend(float(p[1]) for p in pts)
            elif tp == "POLYLINE":
                pts = [(float(p.x), float(p.y)) for p in ent.points()]
                xs.extend(p[0] for p in pts)
                ys.extend(p[1] for p in pts)
            elif tp == "CIRCLE":
                cx, cy, r = float(ent.dxf.center.x), float(ent.dxf.center.y), float(ent.dxf.radius)
                xs.extend([cx - r, cx + r])
                ys.extend([cy - r, cy + r])
            elif tp in ("TEXT", "MTEXT", "POINT"):
                x = float(ent.dxf.insert.x if tp != "POINT" else ent.dxf.location.x)
                y = float(ent.dxf.insert.y if tp != "POINT" else ent.dxf.location.y)
                xs.append(x)
                ys.append(y)
        except Exception:
            continue
    if not xs or not ys:
        return None
    return min(xs), max(xs), min(ys), max(ys)


def _detect_floor_regions(msp, scale: float) -> List[FloorRegion]:
    floor_marks: Dict[int, List[float]] = defaultdict(list)
    for x, _, text in _iter_text_entities(msp):
        floor = _parse_floor_text(text)
        if floor is not None and floor > 0:
            floor_marks[floor].append(x)

    if not floor_marks:
        return []

    bbox = _estimate_bbox(msp)
    if bbox is None:
        return []
    x0, x1, y0, y1 = bbox
    yr = y1 - y0
    pad = max(1.0, yr * 0.1)
    y_min = (y0 - pad) * scale
    y_max = (y1 + pad) * scale

    ordered = sorted((sum(xs) / len(xs), fl) for fl, xs in floor_marks.items())
    regions: List[FloorRegion] = []
    for i, (x, fl) in enumerate(ordered):
        left = x0 if i == 0 else (ordered[i - 1][0] + x) / 2.0
        right = x1 if i == len(ordered) - 1 else (x + ordered[i + 1][0]) / 2.0
        regions.append(FloorRegion(
            floor=fl,
            x_min=left * scale,
            x_max=right * scale,
            y_min=y_min,
            y_max=y_max,
        ))
    return regions


def _detect_structure_layers(msp) -> List[str]:
    layer_count: Dict[str, int] = defaultdict(int)
    total = 0
    for ent in msp:
        if ent.dxftype() not in STRUCT_ENTITY_TYPES:
            continue
        layer = (ent.dxf.layer or "").strip()
        up = layer.upper()
        if any(up.startswith(prefix) for prefix in STRUCT_LAYER_SKIP_PREFIX):
            continue
        layer_count[layer] += 1
        total += 1

    if total == 0:
        return []

    items = sorted(layer_count.items(), key=lambda kv: (-kv[1], kv[0]))
    out: List[str] = []
    covered = 0
    for layer, cnt in items:
        ratio = cnt / total
        if ratio < 0.005:
            continue
        out.append(layer)
        covered += cnt
        if covered / total >= 0.85:
            break
    return out or [items[0][0]]


def _floor_by_regions(x: float, y: float, regions: Sequence[FloorRegion]) -> int:
    for region in regions:
        if region.contains(x, y):
            return region.floor
    return 1


def _detect_exits(msp, scale: float, regions: Sequence[FloorRegion]) -> List[Tuple[int, float, float, str]]:
    exits: List[Tuple[int, float, float, str]] = []
    kws = [k.upper() for k in EXIT_KEYWORDS]
    for x, y, text in _iter_text_entities(msp):
        t = text.upper()
        if not any(k in t for k in kws):
            continue
        sx, sy = x * scale, y * scale
        exits.append((_floor_by_regions(sx, sy, regions), sx, sy, text))
    return exits


def auto_detect_config(dxf_path: str, verbose: bool = False) -> DXFImportConfig:
    try:
        doc = _read_doc(dxf_path)
    except Exception as exc:
        print(f"  ⚠ 自动检测失败，使用默认参数: {exc}")
        return DXFImportConfig()

    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    scale = _INSUNITS_SCALE.get(insunits, 1.0)
    msp = doc.modelspace()

    try:
        floor_regions = _detect_floor_regions(msp, scale)
    except Exception as exc:
        print(f"  ⚠ floor_regions 自动检测失败，已回退为空: {exc}")
        floor_regions = []

    try:
        structure_layers = _detect_structure_layers(msp)
    except Exception as exc:
        print(f"  ⚠ structure_layers 自动检测失败，已回退为空: {exc}")
        structure_layers = []

    try:
        exit_positions = _detect_exits(msp, scale, floor_regions)
    except Exception as exc:
        print(f"  ⚠ exit_positions 自动检测失败，已回退为空: {exc}")
        exit_positions = []

    cfg = DXFImportConfig(
        scale=scale,
        floor_regions=floor_regions,
        structure_layers=structure_layers,
        exit_positions=exit_positions,
    )
    if verbose:
        print("  自动检测结果:")
        print(f"    scale={cfg.scale} (INSUNITS={insunits})")
        print(f"    floor_regions={len(cfg.floor_regions)}")
        print(f"    structure_layers={cfg.structure_layers}")
        print(f"    exit_positions={len(cfg.exit_positions)}")
    return cfg


def print_dxf_info(dxf_path: str) -> None:
    doc = _read_doc(dxf_path)
    msp = doc.modelspace()
    insunits = int(doc.header.get("$INSUNITS", 0) or 0)
    scale = _INSUNITS_SCALE.get(insunits, 1.0)

    etype_count: Dict[str, int] = defaultdict(int)
    layer_count: Dict[str, int] = defaultdict(int)
    for ent in msp:
        etype_count[ent.dxftype()] += 1
        layer_count[(ent.dxf.layer or "").strip()] += 1

    print("DXF 基本信息")
    print("-----------")
    print(f"INSUNITS={insunits}  建议 scale={scale}")
    print("\n实体类型统计:")
    for k, v in sorted(etype_count.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k:12s} {v}")

    print("\n图层统计 TOP 30:")
    for i, (layer, cnt) in enumerate(sorted(layer_count.items(), key=lambda kv: (-kv[1], kv[0]))[:30], 1):
        print(f"  {i:>2d}. {layer or '<EMPTY>'} : {cnt}")

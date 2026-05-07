"""
端到端演示：DXF → 建筑拓扑图（存储）→ 输入火灾/位置 → 规划逃生路径 → 2D/3D 可视化
===================================================================================

两阶段流程
----------
  阶段一（平时）：解析 DXF，构建并保存建筑拓扑图
  阶段二（火灾）：输入当前位置和火灾位置，规划最优逃生路径

用法
----
  # 自动检测所有参数（推荐首次使用）
  python import_from_dxf.py building.dxf --auto

  # 先查看图层信息，再手动微调
  python import_from_dxf.py building.dxf --show-layers

  # 自动检测 + 覆盖部分参数
  python import_from_dxf.py building.dxf --auto --min-room-area 8.0

  # 阶段一：仅解析 DXF，输出图像（不指定火灾位置）
  python import_from_dxf.py building.dxf --scale 0.001

  # 阶段二：指定当前位置和火灾节点，规划路径
  python import_from_dxf.py building.dxf --scale 0.001 \\
         --start F2_S1 --fire-node F2_C3

  # 仅做质量校验
  python import_from_dxf.py building.dxf --validate-only --log-level DEBUG

针对 building.dxf 的完整命令示例
----------------------------------
  python import_from_dxf.py building.dxf \\
    --scale 0.001 \\
    --min-room-area 8.0 \\
    --floor-region "1,99,208,-210,55" \\
    --floor-region "2,208,272,-210,55" \\
    --floor-region "3,272,337,-210,55" \\
    --floor-region "4,337,401,-210,55" \\
    --floor-region "5,401,466,-210,55" \\
    --structure-layer 2 --structure-layer STAIR \\
    --structure-layer WALL --structure-layer 6 \\
    --exit-pos "1,111.2,-62.0,北楼梯出口" \\
    --exit-pos "1,175.6,-62.0,南楼梯出口"

参数说明
--------
  dxf_path              DXF 文件路径

  --scale FLOAT         坐标缩放（毫米→米填 0.001，默认 1.0）
  --floor-height FLOAT  层高（米，默认 3.0）
  --merge-tol FLOAT     节点合并距离阈值（米，默认 1.5）
  --min-room-area FLOAT 闭合多边形最小面积（m²，默认 2.0；台阶踏步约 3m²，
                        建议设为 8.0 以过滤；含大房间的图纸可保持 2.0）
  --max-poly-area FLOAT 闭合多边形最大面积（m²，默认 500.0；超过则为幅面边框）

  --floor-region STR    楼层分区，格式："楼层号,x_min,x_max,y_min,y_max"
                        可多次指定，用于多平面平铺 DXF（图层名无楼层号时）
                        坐标已含 scale 换算（即米）
  --structure-layer STR 白名单图层名，可多次指定（不指定=接受全部图层）
  --extra-stair STR     强制视为楼梯图层的图层名，可多次指定（默认 STAIR）
  --exit-pos STR        手动出口节点，格式："楼层,x,y,标签"，可多次指定
                        坐标单位：米（scale 换算后）

  --start NODE_ID       疏散起始节点 ID（默认自动选最低层第一个非出口节点）
  --fire-node NODE_ID   着火节点 ID，可多次指定
  --fire-edge A,B       着火边（节点对），可多次指定

  --output-dir DIR      输出目录（默认 ./output）
  --validate-only       仅执行质量校验，不生成图像
  --log-level LEVEL     日志级别：DEBUG | INFO | WARNING（默认 INFO）

输出文件
--------
  <output-dir>/cad_escape_3d.png   三维逃生路径图
  <output-dir>/cad_escape_2d.png   各层平面逃生路径图
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
from typing import List

# ── 将脚本所在目录加入 sys.path，支持从任意目录运行 ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cad_import import cad_to_building, DXFImportConfig, FloorRegion
from cad_validation import validate_building
from fire_escape_system import (
    NODE_EXIT,
    visualize_3d,
    visualize_all_floors_2d,
)


# ============================================================
# 命令行解析
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="import_from_dxf.py",
        description="从 DXF 文件构建建筑拓扑图，并（可选）规划逃生路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("dxf_path", help="DXF 文件路径")

    # ── DXF 解析参数 ────────────────────────────────────────────
    g = p.add_argument_group("DXF 解析参数")
    g.add_argument("--scale", type=float, default=1.0,
                   help="坐标缩放（毫米→米: 0.001，默认 1.0）")
    g.add_argument("--floor-height", type=float, default=3.0, dest="floor_height",
                   help="层高（米，默认 3.0）")
    g.add_argument("--merge-tol", type=float, default=1.5, dest="merge_tol",
                   help="节点合并距离（米，默认 1.5）")
    g.add_argument("--min-room-area", type=float, default=2.0, dest="min_room_area",
                   help="闭合多边形最小面积 m²（默认 2.0；建议 8.0 过滤台阶踏步）")
    g.add_argument("--max-poly-area", type=float, default=500.0, dest="max_poly_area",
                   help="闭合多边形最大面积 m²（默认 500.0）")
    g.add_argument("--floor-region", action="append", default=[],
                   dest="floor_regions", metavar="F,XMIN,XMAX,YMIN,YMAX",
                   help='楼层分区，格式 "楼层,x_min,x_max,y_min,y_max"（米）')
    g.add_argument("--structure-layer", action="append", default=[],
                   dest="structure_layers", metavar="LAYER",
                   help="白名单图层名（可多次指定；不指定=接受所有图层）")
    g.add_argument("--extra-stair", action="append", default=["STAIR"],
                   dest="extra_stair_layers", metavar="LAYER",
                   help="强制楼梯图层名（默认 STAIR）")
    g.add_argument("--exit-pos", action="append", default=[],
                   dest="exit_positions", metavar="F,X,Y,LABEL",
                   help='手动出口，格式 "楼层,x,y,标签"（米）')

    # ── 火灾/疏散参数（阶段二） ──────────────────────────────────
    g2 = p.add_argument_group("火灾/疏散参数（阶段二，可选）")
    g2.add_argument("--start", default=None, metavar="NODE_ID",
                    help="疏散起始节点 ID（默认自动选取）")
    g2.add_argument("--fire-node", action="append", default=[],
                    dest="fire_nodes", metavar="NODE_ID",
                    help="着火节点 ID，可多次指定")
    g2.add_argument("--fire-edge", action="append", default=[],
                    dest="fire_edges", metavar="A,B",
                    help="着火边 '节点A,节点B'，可多次指定")

    # ── 输出参数 ────────────────────────────────────────────────
    g3 = p.add_argument_group("输出参数")
    g3.add_argument("--auto", action="store_true",
                    help="自动从 DXF 提取 scale/floor_regions/structure_layers/exit_positions")
    g3.add_argument("--show-layers", action="store_true", dest="show_layers",
                    help="打印 DXF 图层统计后退出")
    g3.add_argument("--output-dir", default=os.path.join(_HERE, "output"),
                    dest="output_dir", help="输出图像目录（默认 ./output）")
    g3.add_argument("--validate-only", action="store_true",
                    help="仅执行质量校验，不生成图像")
    g3.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    dest="log_level")
    return p


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s  %(name)s: %(message)s",
    )

    # ── 检查文件 ─────────────────────────────────────────────────
    if not os.path.isfile(args.dxf_path):
        print(f"  ✗ 文件不存在: {args.dxf_path}")
        return 1

    if args.show_layers:
        from dxf_auto_config import print_dxf_info
        print_dxf_info(args.dxf_path)
        return 0

    # ── 阶段一：解析 DXF，构建建筑拓扑图 ─────────────────────────
    print(f"\n[1/4] 解析 DXF 文件: {args.dxf_path}")
    cli_opts = _collect_cli_opts(sys.argv[1:])
    if args.auto:
        from dxf_auto_config import auto_detect_config
        cfg = auto_detect_config(args.dxf_path, verbose=True)
        _apply_manual_overrides(cfg, args, cli_opts)
    else:
        cfg = _build_config(args)
    cfg.building_name = os.path.splitext(os.path.basename(args.dxf_path))[0]
    _print_config_summary(cfg)

    try:
        building = cad_to_building(args.dxf_path, cfg)
    except Exception as exc:
        print(f"  ✗ 解析失败: {exc}")
        logging.exception("DXF 解析异常")
        return 1

    node_count  = len(building.all_nodes)
    edge_count  = len(building.all_edges)
    floor_list  = sorted(building.floors.keys())
    print(f"  节点数: {node_count}  边数: {edge_count}  楼层: {floor_list}")

    if node_count == 0:
        print("  ✗ 未提取到任何节点，请检查 DXF 图层命名和 --floor-region 配置。")
        _print_naming_hint()
        return 1

    # ── 阶段一结束：如只做校验则在此返回 ────────────────────────
    print("\n[2/4] 执行质量校验 ...")
    report = validate_building(building, stair_xy_tolerance=cfg.node_merge_tolerance * 2)
    report.print_report()

    if args.validate_only:
        return 0 if report.is_valid else 1

    if not report.is_valid:
        print("  ⚠ 存在 ERROR 级别问题，路径规划结果可能不正确，仍将继续执行。")

    # ── 阶段二：输入当前位置和火灾位置，规划路径 ─────────────────
    start_id = _resolve_start(args.start, building)
    if start_id is None:
        print("  ✗ 找不到可用起始节点（需要至少一个非出口节点）。")
        return 1

    fire_nodes      = set(args.fire_nodes)
    fire_edge_pairs = _parse_fire_edges(args.fire_edges)

    print(f"\n[3/4] 路径规划")
    print(f"  起点: {start_id}")
    if fire_nodes:
        print(f"  着火节点: {', '.join(sorted(fire_nodes))}")
    if fire_edge_pairs:
        print(f"  着火边:   {fire_edge_pairs}")

    if not fire_nodes and not fire_edge_pairs:
        print("  (未指定火灾位置，将规划到最近出口的路径)")

    path = building.find_escape_route(
        start_id,
        fire_node_ids   = fire_nodes,
        fire_edge_pairs = fire_edge_pairs,
    )

    if path:
        _print_path(path, building)
    else:
        print("  ✗ 未找到可用逃生路径（所有出口被封锁或不连通）。")

    # ── 可视化 ───────────────────────────────────────────────────
    print(f"\n[4/4] 生成可视化图像 → {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Arial Unicode MS", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    try:
        path_3d = os.path.join(args.output_dir, "cad_escape_3d.png")
        path_2d = os.path.join(args.output_dir, "cad_escape_2d.png")

        fig3d = visualize_3d(building, escape_path=path,
                             fire_zones=fire_nodes, start_id=start_id)
        fig2d = visualize_all_floors_2d(building, escape_path=path,
                                        fire_zones=fire_nodes, start_id=start_id)

        fig3d.savefig(path_3d, dpi=150, bbox_inches="tight",
                      facecolor=fig3d.get_facecolor())
        fig2d.savefig(path_2d, dpi=150, bbox_inches="tight",
                      facecolor=fig2d.get_facecolor())
        plt.close("all")
        print(f"  3D 图: {path_3d}")
        print(f"  2D 图: {path_2d}")
    except Exception as exc:
        print(f"  ✗ 可视化失败: {exc}")
        logging.exception("可视化异常")
        return 1

    return 0


# ============================================================
# 配置构建
# ============================================================

def _build_config(args) -> DXFImportConfig:
    """从命令行参数构建 DXFImportConfig。"""
    regions = _parse_floor_regions(args.floor_regions)
    exits = _parse_exit_positions(args.exit_positions)

    return DXFImportConfig(
        scale                 = args.scale,
        floor_height          = args.floor_height,
        node_merge_tolerance  = args.merge_tol,
        min_room_area         = args.min_room_area,
        max_polygon_area_m2   = args.max_poly_area,
        floor_regions         = regions,
        structure_layers      = args.structure_layers,
        extra_stair_layers    = args.extra_stair_layers,
        exit_positions        = exits,
        building_name         = os.path.splitext(os.path.basename(args.dxf_path))[0],
    )


def _collect_cli_opts(argv: List[str]) -> set:
    opts = set()
    for a in argv:
        if not a.startswith("--"):
            continue
        name = a.split("=", 1)[0]
        opts.add(name)
    return opts


def _parse_floor_regions(values: List[str]) -> List[FloorRegion]:
    regions: List[FloorRegion] = []
    for s in values:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 5:
            print(f"  ⚠ 忽略无效 --floor-region '{s}'（格式: 楼层,x_min,x_max,y_min,y_max）")
            continue
        try:
            regions.append(FloorRegion(
                floor=int(parts[0]),
                x_min=float(parts[1]), x_max=float(parts[2]),
                y_min=float(parts[3]), y_max=float(parts[4]),
            ))
        except ValueError:
            print(f"  ⚠ 忽略无法解析的 --floor-region '{s}'")
    return regions


def _parse_exit_positions(values: List[str]) -> List[tuple]:
    exits: List[tuple] = []
    for s in values:
        parts = s.split(",", 3)
        if len(parts) < 3:
            print(f"  ⚠ 忽略无效 --exit-pos '{s}'（格式: 楼层,x,y,标签）")
            continue
        try:
            exits.append((
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                parts[3].strip() if len(parts) > 3 else "出口",
            ))
        except ValueError:
            print(f"  ⚠ 忽略无法解析的 --exit-pos '{s}'")
    return exits


def _apply_manual_overrides(cfg: DXFImportConfig, args, cli_opts: set) -> None:
    if "--scale" in cli_opts:
        cfg.scale = args.scale
    if "--floor-height" in cli_opts:
        cfg.floor_height = args.floor_height
    if "--merge-tol" in cli_opts:
        cfg.node_merge_tolerance = args.merge_tol
    if "--min-room-area" in cli_opts:
        cfg.min_room_area = args.min_room_area
    if "--max-poly-area" in cli_opts:
        cfg.max_polygon_area_m2 = args.max_poly_area
    if args.floor_regions:
        cfg.floor_regions = _parse_floor_regions(args.floor_regions)
    if args.structure_layers:
        cfg.structure_layers = args.structure_layers
    if "--extra-stair" in cli_opts:
        cfg.extra_stair_layers = args.extra_stair_layers
    if args.exit_positions:
        cfg.exit_positions = _parse_exit_positions(args.exit_positions)


# ============================================================
# 辅助函数
# ============================================================

def _resolve_start(start_arg, building) -> str | None:
    if start_arg:
        if building.get_node(start_arg) is None:
            print(f"  ⚠ 指定的起点 '{start_arg}' 不存在，尝试自动选取。")
        else:
            return start_arg
    min_floor = min(building.floors.keys())
    candidates = sorted(
        nid for nid, n in building.all_nodes.items()
        if n.floor == min_floor and n.type != NODE_EXIT
    )
    if not candidates:
        candidates = sorted(
            nid for nid, n in building.all_nodes.items() if n.type != NODE_EXIT)
    if candidates:
        print(f"  自动选取起始节点: {candidates[0]}")
        return candidates[0]
    return None


def _parse_fire_edges(fire_edge_args: list) -> list:
    pairs = []
    for fe in fire_edge_args:
        parts = [p.strip() for p in fe.split(",")]
        if len(parts) == 2 and parts[0] and parts[1]:
            pairs.append((parts[0], parts[1]))
        else:
            print(f"  ⚠ 忽略无效的着火边参数 '{fe}'（格式: 节点A,节点B）")
    return pairs


def _print_path(path: list, building) -> None:
    ICON = {"room": "[房]", "corridor": "[廊]", "stair": "[梯]", "exit": "[出口]"}
    sep = "─" * 52
    print(f"  最优逃生路径（共 {len(path)} 步）:")
    print(f"  {sep}")
    for i, nid in enumerate(path):
        n    = building.get_node(nid)
        icon = ICON.get(n.type, "[?]")
        arrow = " →" if i < len(path) - 1 else ""
        print(f"  {i+1:2d}. {icon} [F{n.floor}] {n.label}{arrow}")
    print(f"  {sep}")
    exit_node = building.get_node(path[-1])
    print(f"  出口: {exit_node.label}  (楼层 F{exit_node.floor})")


def _print_config_summary(cfg: DXFImportConfig) -> None:
    print(f"  配置: scale={cfg.scale}  merge_tol={cfg.node_merge_tolerance}m"
          f"  min_room={cfg.min_room_area}m²  max_poly={cfg.max_polygon_area_m2}m²")
    if cfg.floor_regions:
        print(f"  楼层分区: {len(cfg.floor_regions)} 个区域"
              f"  [{', '.join(f'F{r.floor}(X:{r.x_min:.0f}~{r.x_max:.0f})' for r in cfg.floor_regions)}]")
    else:
        print("  楼层识别: 图层名正则（未指定 --floor-region）")
    if cfg.structure_layers:
        print(f"  白名单图层: {cfg.structure_layers}")
    if cfg.exit_positions:
        print(f"  手动出口: {len(cfg.exit_positions)} 个")


def _print_naming_hint() -> None:
    print("""
  DXF 图层命名约定提示
  ─────────────────────────────────────────────────────
  标准多层 DXF（图层名含楼层号）：
    楼层识别：F1, F-1, FLOOR1, L1, 1F 等
    类型识别：exit/出口, stair/楼梯, room/房间, corridor/走廊

  多平面平铺 DXF（国内常见，图层名不含楼层号）：
    → 使用 --floor-region "楼层,x_min,x_max,y_min,y_max" 手动分区
    → 使用 --structure-layer 指定要处理的图层（过滤标注/家具等）
    → 使用 --exit-pos "楼层,x,y,标签" 手动指定出口位置
    → 先用 --log-level DEBUG 运行查看节点坐标，再填写坐标参数

  building.dxf 参考命令：
    python import_from_dxf.py building.dxf --scale 0.001 --min-room-area 8.0 \\
      --floor-region "1,99,208,-210,55" --floor-region "2,208,272,-210,55" \\
      --floor-region "3,272,337,-210,55" --floor-region "4,337,401,-210,55" \\
      --floor-region "5,401,466,-210,55" \\
      --structure-layer 2 --structure-layer STAIR --structure-layer WALL \\
      --exit-pos "1,111.2,-62.0,北楼梯出口" --exit-pos "1,175.6,-62.0,南楼梯出口"
  ─────────────────────────────────────────────────────
""")


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    sys.exit(main())

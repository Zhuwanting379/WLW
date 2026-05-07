"""
使用预设测试数据运行逃生路线规划，并将图片保存到 output 文件夹。
"""

import os
import matplotlib
matplotlib.use("Agg")   # 非交互式后端，直接保存文件

import matplotlib.pyplot as plt

# 确保 fire_escape_system 可被导入（同目录）
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fire_escape_system import (
    Building, Floor, Node,
    NODE_ROOM, NODE_CORRIDOR, NODE_STAIR, NODE_EXIT,
    visualize_3d, visualize_all_floors_2d,
)

# ── 输出目录 ───────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_test_building() -> Building:
    """按用户提供的测试数据构建三层建筑"""
    building = Building("测试建筑")

    # ── 第 1 层 ────────────────────────────────────────────
    f1 = Floor(1)
    f1.add_node(Node("F1_RoomA",    NODE_ROOM,     1, -5,  -3, "房间A"))
    f1.add_node(Node("F1_RoomB",    NODE_ROOM,     1,  5,  -3, "房间B"))
    f1.add_node(Node("F1_Corridor", NODE_CORRIDOR, 1,  0,  -3, "1层走廊"))
    f1.add_node(Node("F1_Stair",    NODE_STAIR,    1,  0,   2, "楼梯口"))
    f1.add_node(Node("F1_Exit",     NODE_EXIT,     1,  6,  -3, "安全出口"))

    f1.connect("F1_RoomA",    "F1_Corridor")
    f1.connect("F1_RoomB",    "F1_Corridor")
    f1.connect("F1_Corridor", "F1_Stair")
    f1.connect("F1_Corridor", "F1_Exit")
    building.add_floor(f1)

    # ── 第 2 层 ────────────────────────────────────────────
    f2 = Floor(2)
    f2.add_node(Node("F2_RoomC",    NODE_ROOM,     2, -4,  -2, "房间C"))
    f2.add_node(Node("F2_Stair",    NODE_STAIR,    2,  0,   2, "楼梯口"))
    f2.add_node(Node("F2_Corridor", NODE_CORRIDOR, 2, -4,   2, "2层走廊"))

    f2.connect("F2_RoomC",    "F2_Corridor")
    f2.connect("F2_Corridor", "F2_Stair")
    building.add_floor(f2)

    # ── 第 3 层 ────────────────────────────────────────────
    f3 = Floor(3)
    f3.add_node(Node("F3_RoomD", NODE_ROOM,  3,  3,  4, "房间D"))
    f3.add_node(Node("F3_Stair", NODE_STAIR, 3,  0,  2, "楼梯口"))

    f3.connect("F3_RoomD", "F3_Stair")
    building.add_floor(f3)

    # ── 跨层楼梯 & 邻接表 ─────────────────────────────────
    building.connect_stairs()
    building.build_graph()
    return building


def main():
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Arial Unicode MS", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    building = build_test_building()

    start_id       = "F3_RoomD"
    fire_node_ids  = {"F2_RoomC"}
    fire_edge_pairs = []

    # ── 路径规划 ─────────────────────────────────────────
    path = building.find_escape_route(start_id, fire_node_ids, fire_edge_pairs)

    sep = "=" * 58
    print(sep)
    if path:
        print("  [成功] 找到最优逃生路径！\n")
        for i, nid in enumerate(path):
            n    = building.get_node(nid)
            icon = {"room":"[房]","corridor":"[廊]",
                    "stair":"[梯]","exit":"[出口]"}.get(n.type, "[?]")
            arrow = " ->" if i < len(path) - 1 else ""
            print(f"  {i+1:2d}. {icon} [F{n.floor}] {n.label}{arrow}")
        print(f"\n  出口：{building.get_node(path[-1]).label}")
    else:
        print("  [警告] 无法找到逃生路径！所有路径均被封锁。")
    print(sep)

    # ── 可视化 & 保存 ─────────────────────────────────────
    fig3d   = visualize_3d(
        building, escape_path=path,
        fire_zones=fire_node_ids, start_id=start_id)
    fig2d   = visualize_all_floors_2d(
        building, escape_path=path,
        fire_zones=fire_node_ids, start_id=start_id)

    path_3d = os.path.join(OUTPUT_DIR, "escape_route_3d.png")
    path_2d = os.path.join(OUTPUT_DIR, "escape_route_2d.png")

    fig3d.savefig(path_3d, dpi=150, bbox_inches="tight",
                  facecolor=fig3d.get_facecolor())
    fig2d.savefig(path_2d, dpi=150, bbox_inches="tight",
                  facecolor=fig2d.get_facecolor())

    plt.close("all")
    print(f"\n[OK] 图片已保存：")
    print(f"     3D 图：{path_3d}")
    print(f"     2D 图：{path_2d}")


if __name__ == "__main__":
    main()

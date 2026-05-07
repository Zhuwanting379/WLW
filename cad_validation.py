"""
建筑拓扑图导入质量校验模块
============================

对 cad_import.cad_to_building() 返回的 Building 执行静态检查，
输出包含警告和错误的 ValidationReport。

检查项
------
- EXIT001   : 无安全出口（ERROR）
- STAIR001  : 楼梯节点跨层 XY 坐标偏差过大，无法自动连通（WARNING）
- CONN001   : 图不完全连通，存在不可达节点（WARNING）
- REACH001  : 存在楼层无法到达任何安全出口（ERROR/WARNING）
- ORPHAN01  : 存在孤立节点（无任何连边）（WARNING）
- FLOOR001  : 楼层编号不连续（WARNING）
- EDGE001   : 自环边或重复边（WARNING）

用法
----
    from cad_validation import validate_building
    report = validate_building(building)
    report.print_report()
    if not report.is_valid:
        print("存在严重问题，请检查 DXF 图层命名和连接关系。")
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from fire_escape_system import Building, Node, NODE_EXIT, NODE_STAIR

logger = logging.getLogger(__name__)


# ============================================================
# 报告数据结构
# ============================================================

@dataclass
class ValidationIssue:
    """单条校验结论。"""
    code:    str   # 问题代码，如 "EXIT001"
    level:   str   # "ERROR" | "WARNING" | "INFO"
    message: str   # 人类可读的简要描述
    detail:  str = ""  # 可选技术细节


@dataclass
class ValidationReport:
    """全部校验结论的容器，提供过滤和打印功能。"""
    issues: List[ValidationIssue] = field(default_factory=list)

    # ── 添加结论 ──────────────────────────────────────────────────

    def add_error(self, code: str, message: str, detail: str = "") -> None:
        self.issues.append(ValidationIssue(code, "ERROR", message, detail))

    def add_warning(self, code: str, message: str, detail: str = "") -> None:
        self.issues.append(ValidationIssue(code, "WARNING", message, detail))

    def add_info(self, code: str, message: str, detail: str = "") -> None:
        self.issues.append(ValidationIssue(code, "INFO", message, detail))

    # ── 过滤属性 ──────────────────────────────────────────────────

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "WARNING"]

    @property
    def is_valid(self) -> bool:
        """无 ERROR 级别问题时返回 True。"""
        return len(self.errors) == 0

    # ── 格式化输出 ────────────────────────────────────────────────

    def print_report(self) -> None:
        """向标准输出打印格式化报告。"""
        _SEP  = "=" * 58
        _SEP2 = "─" * 58
        _ICON = {"ERROR": "✗", "WARNING": "⚠", "INFO": "✓"}

        print(_SEP)
        print("  建筑拓扑图校验报告")
        print(_SEP)

        non_info = [i for i in self.issues if i.level != "INFO"]
        if not non_info:
            print("  ✓ 所有检查均通过，未发现问题。")
        else:
            for issue in self.issues:
                icon = _ICON.get(issue.level, "?")
                print(f"  {icon} [{issue.level}] {issue.code}: {issue.message}")
                if issue.detail:
                    print(f"       ↳ {issue.detail}")

        print(_SEP2)
        print(f"  ERROR: {len(self.errors)}   WARNING: {len(self.warnings)}")
        print(_SEP)

    def to_dict(self) -> dict:
        """序列化为字典，便于日志记录或 JSON 输出。"""
        return {
            "is_valid": self.is_valid,
            "errors":   len(self.errors),
            "warnings": len(self.warnings),
            "issues":   [
                {"code": i.code, "level": i.level, "message": i.message, "detail": i.detail}
                for i in self.issues
            ],
        }


# ============================================================
# 校验器
# ============================================================

class GraphValidator:
    """对 Building 执行系列静态质量检查。"""

    def __init__(self, stair_xy_tolerance: float = 1.0) -> None:
        """
        参数
        ----
        stair_xy_tolerance : float
            楼梯节点跨层 XY 坐标偏差允许上限（米）。
            超过此值时警告楼梯可能无法自动连通。
        """
        self.stair_tol = stair_xy_tolerance

    # ── 主入口 ────────────────────────────────────────────────────

    def validate(self, building: Building) -> ValidationReport:
        """
        对已调用 connect_stairs() 和 build_graph() 的 Building 执行全部检查。

        返回 ValidationReport，包含 ERROR / WARNING / INFO 三级结论。
        """
        rpt = ValidationReport()
        self._check_exits(building, rpt)
        self._check_stair_alignment(building, rpt)
        self._check_connectivity(building, rpt)
        self._check_exit_reachability(building, rpt)
        self._check_orphan_nodes(building, rpt)
        self._check_floor_continuity(building, rpt)
        self._check_edge_sanity(building, rpt)

        # 统计信息（INFO）
        rpt.add_info(
            "STATS",
            f"节点总数={len(building.all_nodes)}, "
            f"边总数={len(building.all_edges)}, "
            f"楼层数={len(building.floors)}",
        )
        return rpt

    # ── 各项检查 ──────────────────────────────────────────────────

    def _check_exits(self, building: Building, rpt: ValidationReport) -> None:
        """EXIT001: 建筑必须至少有一个 exit 节点。"""
        exits = [n for n in building.all_nodes.values() if n.type == NODE_EXIT]
        if not exits:
            rpt.add_error(
                "EXIT001",
                "建筑中未找到任何安全出口节点（type=exit）。",
                "请确认 DXF 图层命名包含 exit/出口 等关键词，或手动添加 exit 节点。",
            )
        else:
            ids_str = ", ".join(n.id for n in exits[:10])
            rpt.add_info("EXIT001", f"安全出口: {len(exits)} 个", ids_str)

    def _check_stair_alignment(self, building: Building, rpt: ValidationReport) -> None:
        """STAIR001: 楼梯节点跨层 XY 坐标偏差不应超过阈值。"""
        # 以坐标四舍五入（1位小数）为楼梯井标识符，容忍微小浮点差异
        stair_groups: Dict[str, List[Node]] = defaultdict(list)
        for node in building.all_nodes.values():
            if node.type == NODE_STAIR:
                key = f"{node.x:.1f},{node.y:.1f}"
                stair_groups[key].append(node)

        for members in stair_groups.values():
            if len(members) < 2:
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    d = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                    if d > self.stair_tol:
                        rpt.add_warning(
                            "STAIR001",
                            f"楼梯 {a.id}(F{a.floor}) 与 {b.id}(F{b.floor}) XY 偏差 {d:.2f}m"
                            f" > 阈值 {self.stair_tol}m，跨层边可能未生成。",
                            "楼梯节点应位于同一竖向井道（XY 坐标相同），"
                            "可调整 stair_xy_tolerance 或在 DXF 中对齐坐标。",
                        )

    def _check_connectivity(self, building: Building, rpt: ValidationReport) -> None:
        """CONN001: 图（当前 build_graph 结果）应整体连通。"""
        all_ids = list(building.all_nodes.keys())
        if not all_ids:
            return

        visited: Set[str] = set()
        queue = [all_ids[0]]
        visited.add(all_ids[0])
        while queue:
            curr = queue.pop()
            for nbr, _, _ in building._graph.get(curr, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

        unreachable = set(all_ids) - visited
        if unreachable:
            rpt.add_warning(
                "CONN001",
                f"图不完全连通：{len(unreachable)} 个节点与主图不相连。",
                "节点: " + ", ".join(sorted(unreachable)[:20]),
            )
        else:
            rpt.add_info("CONN001", "图连通性正常，所有节点均可互达。")

    def _check_exit_reachability(self, building: Building, rpt: ValidationReport) -> None:
        """REACH001: 每个楼层的节点应能到达至少一个安全出口。"""
        exits = {nid for nid, n in building.all_nodes.items() if n.type == NODE_EXIT}
        if not exits:
            return  # EXIT001 已报告，此处跳过

        # 从所有出口节点做反向 BFS（图无向，正向等价）
        reachable: Set[str] = set(exits)
        queue = list(exits)
        while queue:
            curr = queue.pop()
            for nbr, _, _ in building._graph.get(curr, []):
                if nbr not in reachable:
                    reachable.add(nbr)
                    queue.append(nbr)

        for fnum, fl in building.floors.items():
            floor_ids = set(fl.nodes.keys())
            unreachable = floor_ids - reachable
            if unreachable == floor_ids:
                rpt.add_error(
                    "REACH001",
                    f"第 {fnum} 层所有节点均无法到达任何安全出口。",
                    "请检查该层是否存在楼梯节点，且楼梯节点 XY 坐标与相邻层对齐。",
                )
            elif unreachable:
                rpt.add_warning(
                    "REACH001",
                    f"第 {fnum} 层有 {len(unreachable)} 个节点无法到达安全出口。",
                    "节点: " + ", ".join(sorted(unreachable)[:10]),
                )

    def _check_orphan_nodes(self, building: Building, rpt: ValidationReport) -> None:
        """ORPHAN01: 发现没有任何边连接的孤立节点。"""
        connected_ids: Set[str] = set()
        for edge in building.all_edges:
            connected_ids.add(edge.node_a)
            connected_ids.add(edge.node_b)

        orphans = [nid for nid in building.all_nodes if nid not in connected_ids]
        if orphans:
            rpt.add_warning(
                "ORPHAN01",
                f"发现 {len(orphans)} 个孤立节点（无任何边连接）。",
                "节点: " + ", ".join(orphans[:20]),
            )
        else:
            rpt.add_info("ORPHAN01", "无孤立节点。")

    def _check_floor_continuity(self, building: Building, rpt: ValidationReport) -> None:
        """FLOOR001: 楼层编号应连续，无跳层。"""
        floor_nums = sorted(building.floors.keys())
        if len(floor_nums) < 2:
            return
        for i in range(len(floor_nums) - 1):
            if floor_nums[i + 1] - floor_nums[i] > 1:
                rpt.add_warning(
                    "FLOOR001",
                    f"楼层编号不连续：F{floor_nums[i]} 之后直接跳到 F{floor_nums[i+1]}。",
                    "可能是 DXF 图层命名导致楼层号跳跃，请检查图层名是否含多个数字。",
                )

    def _check_edge_sanity(self, building: Building, rpt: ValidationReport) -> None:
        """EDGE001: 检测自环边和重复边。"""
        seen: Set[Tuple[str, str]] = set()
        self_loops = 0
        duplicates = 0

        for edge in building.all_edges:
            if edge.node_a == edge.node_b:
                self_loops += 1
                continue
            key = (min(edge.node_a, edge.node_b), max(edge.node_a, edge.node_b))
            if key in seen:
                duplicates += 1
            else:
                seen.add(key)

        if self_loops:
            rpt.add_warning("EDGE001", f"发现 {self_loops} 条自环边（起终点相同）。")
        if duplicates:
            rpt.add_warning("EDGE001", f"发现 {duplicates} 条重复边（相同节点对出现多次）。")
        if not self_loops and not duplicates:
            rpt.add_info("EDGE001", "无自环或重复边。")


# ============================================================
# 便捷函数
# ============================================================

def validate_building(
    building: Building,
    stair_xy_tolerance: float = 1.0,
) -> ValidationReport:
    """
    快捷接口：创建 GraphValidator 并对建筑执行全部检查。

    参数
    ----
    building            : 已调用 connect_stairs() 和 build_graph() 的建筑对象
    stair_xy_tolerance  : 楼梯跨层 XY 坐标偏差阈值（米，默认 1.0）

    返回
    ----
    ValidationReport
    """
    validator = GraphValidator(stair_xy_tolerance=stair_xy_tolerance)
    return validator.validate(building)
"""
========================================================
建筑火灾动态逃生路线规划系统
Building Fire Dynamic Escape Route Planning System
========================================================
 
【节点类型说明】
  room     房间     - 普通功能空间（办公室、卧室等）
  corridor 走廊     - 走廊中的关键节点（交叉口、转折点）
  stair    楼梯口   - 楼层间竖向通道入口
                     同坐标(x,y)的 stair 节点自动跨层连通
  exit     安全出口 - 建筑对外的疏散门，Dijkstra 的目标节点
 
【火灾位置说明】
  火灾可发生在两类位置：
  1. 节点：直接封锁该节点的所有相邻边
  2. 边（走廊中段）：封锁指定的 node_a--node_b 连线
  输入时先选择类型，再按提示输入节点ID或边的两端节点ID。
 
【可视化图例】
  房间   (room)     - 蓝色  圆形
  走廊   (corridor) - 黄色  圆形
  楼梯口 (stair)    - 橙色  圆形
  安全出口(exit)    - 绿色  菱形
  当前位置(you)     - 绿色  五角星
  火灾节点(fire)    - 红色  三角形
  火灾边  (fire)    - 红色  虚线
  逃生路径           - 亮绿色 粗线
 
========================================================
"""

import math
import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.path   as mpath
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 3D 箭头补丁（投影到屏幕坐标，外观与 2D annotate 一致） ─
class _Arrow3D(FancyArrowPatch):
    """将三维线段渲染为 FancyArrowPatch，支持与 2D 相同的 arrowstyle。"""
    def __init__(self, p1, p2, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._p1 = p1  # (x, y, z)
        self._p2 = p2

    def do_3d_projection(self):
        x1, y1, z1 = self._p1
        x2, y2, z2 = self._p2
        ax = self.axes
        xs, ys, zs = proj3d.proj_transform(
            [x1, x2], [y1, y2], [z1, z2], ax.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)

# ============================================================
# 节点类型常量 & 可视化配色
# ============================================================

# 节点类型标识符（与 Node.type 字段对应）
NODE_ROOM     = "room"       # 普通房间
NODE_CORRIDOR = "corridor"   # 走廊（连接通道）
NODE_STAIR    = "stair"      # 楼梯口（楼层间通道入口，本身不是出口）
NODE_EXIT     = "exit"       # 安全出口（逃生终点，通常只设在 1 层）

# 每层建筑高度（米），用于将楼层编号转换为 3D Z 轴坐标
FLOOR_HEIGHT  = 3.0

# 节点类型 -> 散点颜色（正常状态下的默认配色）
COLOR_MAP = {
    NODE_ROOM     : "#4A90D9",  # 蓝色   - 普通房间
    NODE_CORRIDOR : "#F5DE60",  # 黄色   - 走廊通道
    NODE_STAIR    : "#FF9D00",  # 橙色   - 楼梯口（跨层入口）
    NODE_EXIT     : "#01CF0C",  # 绿色   - 安全出口（逃生终点）
}
COLOR_FIRE       = "#E74C3C"    # 红色   - 火灾危险节点（红色三角标记）
COLOR_FIRE_EDGE  = "#FF6B6B"    # 浅红色 - 火灾边（走廊中段着火）
COLOR_PATH       = "#2ECC71"    # 亮绿色 - 最优逃生路径高亮
COLOR_START      = "#00FF44"    # 纯绿色 - 当前所在位置（绿色五角星标记）
COLOR_EDGE       = "#888888"    # 灰色   - 普通通道连线
COLOR_EDGE_STAIR = "#FF9D00"    # 橙色   - 楼梯跨层连线

# ============================================================
# 数据结构
# ============================================================

@dataclass
class Node:
    """
    建筑图中的一个逻辑节点（房间、走廊、楼梯口或安全出口）。

    Attributes:
        id    : 全局唯一标识符（用户自定义字符串）
        type  : 节点类型，取値为 NODE_ROOM / NODE_CORRIDOR / NODE_STAIR / NODE_EXIT
        floor : 所在楼层编号（从 1 开始）
        x, y  : 节点在平面图中的 2D 坐标（单位：米）
        label : 可读显示名称（用于可视化标注）
    """
    id    : str
    type  : str   # room / corridor / stair / exit
    floor : int
    x     : float
    y     : float
    label : str = ""

    @property
    def z(self) -> float:
        """根据楼层号计算 3D Z 轴高度（米），公式：(floor-1) × FLOOR_HEIGHT"""
        return (self.floor - 1) * FLOOR_HEIGHT

    @property
    def pos3d(self) -> Tuple[float, float, float]:
        """返回节点的三维坐标 (x, y, z)，供 matplotlib 3D 绘图使用"""
        return (self.x, self.y, self.z)

    def dist_to(self, other: "Node") -> float:
        """计算到另一节点的三维欧几里得距离，用作边的默认权重（含楼层高度差）"""
        return math.sqrt(
            (self.x-other.x)**2 + (self.y-other.y)**2 + (self.z-other.z)**2
        )


@dataclass
class Edge:
    """
    建筑拓扑图中的无向边，表示两节点间的可通行路径。
 
    node_a, node_b : 连接的两端节点 id
    weight         : 路径代价（欧几里得距离），由 Floor.connect 自动计算
    blocked        : True 时 Dijkstra 不可经过（节点火灾或边火灾均可触发）
    is_stair       : True 表示跨层楼梯边，2D 图中不绘制
    fire_on_edge   : True 表示火灾直接发生在该边上（走廊中段着火）
                     区别于节点火灾（fire_on_edge=False 时封锁来自节点）
    """
    node_a      : str
    node_b      : str
    weight      : float = 0.0
    blocked     : bool  = False
    is_stair    : bool  = False
    fire_on_edge: bool  = False   # 边火灾标记，用于可视化时用红色虚线绘制


class Floor:
    """
    表示建筑的一个楼层，持有该层的所有节点和边。
    楼层仅用于数据组织；图的连通性在 Building 层统一管理。
    """
    def __init__(self, floor_num: int):
        self.floor_num = floor_num
        self.nodes: Dict[str, Node] = {}  # 本层节点字典 {id -> Node}
        self.edges: List[Edge]      = []  # 本层平面内的边列表（不含跨层楼梯边）

    def add_node(self, node: Node):
        """向本层添加一个节点"""
        self.nodes[node.id] = node

    def connect(self, id_a: str, id_b: str):
        """按两端点的三维距离自动计算权重，并在本层添加一条无向边"""
        a, b = self.nodes[id_a], self.nodes[id_b]
        self.edges.append(Edge(id_a, id_b, weight=a.dist_to(b)))


class Building:
    """
    多层建筑的完整拓扑图，提供路径规划接口。
 
    职责：
    - 聚合所有 Floor，维护全局节点和边的索引
    - 自动识别楼梯井并添加跨层边（connect_stairs）
    - 支持节点火灾和边火灾两种封锁方式（set_fire）
    - 使用 Dijkstra 算法寻找最短逃生路径（find_escape_route）
 
    内部结构：
    _nodes : 全局节点字典 {id -> Node}
    _edges : 全局边列表（含跨层楼梯边）
    _graph : 邻接表 {id -> [(邻居id, 权重, Edge)]}，每次重建时排除 blocked 边
    """
 
    def __init__(self, name: str = "建筑"):
        self.name    = name
        self.floors  : Dict[int, Floor] = {}
        self._nodes  : Dict[str, Node]  = {}
        self._edges  : List[Edge]       = []
        self._graph  : Dict[str, List[Tuple[str, float, Edge]]] = defaultdict(list)
 
    # ── 构建接口 ───────────────────────────────────────────
 
    def add_floor(self, floor: Floor) -> None:
        """将一层的节点和边并入全局索引。add_floor 后需调用 connect_stairs + build_graph。"""
        self.floors[floor.floor_num] = floor
        self._nodes.update(floor.nodes)
        self._edges.extend(floor.edges)
 
    def connect_stairs(self) -> None:
        """
        识别各层楼梯井（同 (x,y) 坐标的 stair 节点）并添加跨层边。
        仅连接相邻楼层（楼层差=1），不跳层。权重 = FLOOR_HEIGHT。
        """
        stair_map: Dict[Tuple[float, float], List[Node]] = defaultdict(list)
        for node in self._nodes.values():
            if node.type == NODE_STAIR:
                stair_map[(node.x, node.y)].append(node)
 
        for shaft in stair_map.values():
            shaft.sort(key=lambda n: n.floor)
            for i in range(len(shaft) - 1):
                a, b = shaft[i], shaft[i + 1]
                self._edges.append(Edge(a.id, b.id, weight=FLOOR_HEIGHT, is_stair=True))
 
    def build_graph(self) -> None:
        """依据当前所有边的 blocked 状态重建邻接表。封锁状态变更后必须调用。"""
        self._graph.clear()
        for edge in self._edges:
            if not edge.blocked:
                self._graph[edge.node_a].append((edge.node_b, edge.weight, edge))
                self._graph[edge.node_b].append((edge.node_a, edge.weight, edge))
 
    def reset_fire(self) -> None:
        """清除所有边的 blocked 和 fire_on_edge 标记"""
        for edge in self._edges:
            edge.blocked      = False
            edge.fire_on_edge = False
 
    def set_fire(
        self,
        fire_node_ids : Set[str],
        fire_edge_pairs: List[Tuple[str, str]]
    ) -> None:
        """
        标记火灾区域并重建图。支持两种火灾类型：
 
        1. 节点火灾（fire_node_ids）：
           封锁所有端点在集合中的边。
           火灾节点本身变为孤岛，Dijkstra 无法经过。
 
        2. 边火灾（fire_edge_pairs）：
           直接封锁走廊中段的某条边，同时设置 fire_on_edge=True 用于可视化。
           此时两端节点本身仍可通行（只是这段走廊不可通）。
 
        参数：
            fire_node_ids  : 着火节点 id 集合
            fire_edge_pairs: [(node_a, node_b), ...] 着火边端点对列表
        """
        self.reset_fire()
 
        # --- 封锁节点火灾相关的边 ---
        for edge in self._edges:
            if edge.node_a in fire_node_ids or edge.node_b in fire_node_ids:
                edge.blocked = True
 
        # --- 封锁边火灾 ---
        # 将 fire_edge_pairs 转换为集合（双向匹配）
        fire_edge_set = set()
        for a, b in fire_edge_pairs:
            fire_edge_set.add((a, b))
            fire_edge_set.add((b, a))   # 无向边，双向都加入
 
        for edge in self._edges:
            if (edge.node_a, edge.node_b) in fire_edge_set:
                edge.blocked      = True
                edge.fire_on_edge = True
 
        self.build_graph()
 
    # ── Dijkstra 路径规划 ──────────────────────────────────
 
    def find_escape_route(
        self,
        start_id       : str,
        fire_node_ids  : Optional[Set[str]]          = None,
        fire_edge_pairs: Optional[List[Tuple[str,str]]] = None
    ) -> Optional[List[str]]:
        """
        Dijkstra 寻找从 start_id 到最近 exit 节点的最短路径。
 
        算法：最小堆优先队列 + 惰性删除。
        一旦弹出 exit 节点立即终止并回溯路径（多出口时自动选最近的）。
 
        返回节点 id 序列（含起终点），无法到达任何出口时返回 None。
        """
        self.set_fire(
            fire_node_ids   or set(),
            fire_edge_pairs or []
        )
 
        exits = {nid for nid, n in self._nodes.items() if n.type == NODE_EXIT}
        if not exits:
            return None
 
        dist: Dict[str, float]         = {nid: float("inf") for nid in self._nodes}
        prev: Dict[str, Optional[str]] = {nid: None for nid in self._nodes}
        dist[start_id] = 0.0
        heap: List[Tuple[float, str]]  = [(0.0, start_id)]
 
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue          # 惰性删除：跳过过期条目
            if u in exits:
                # 回溯前驱链，还原完整路径
                path: List[str] = []
                cur: Optional[str] = u
                while cur is not None:
                    path.append(cur)
                    cur = prev[cur]
                path.reverse()
                return path
            for v, w, _ in self._graph[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
 
        return None   # 所有路径均被封锁
 
    # ── 属性访问 ───────────────────────────────────────────
 
    def get_node(self, nid: str) -> Optional[Node]:
        return self._nodes.get(nid)
 
    @property
    def all_nodes(self) -> Dict[str, Node]:
        return self._nodes
 
    @property
    def all_edges(self) -> List[Edge]:
        return self._edges

# ============================================================
# 交互式建筑输入
# ============================================================

SEP  = "=" * 58
SEP2 = "─" * 48

def _ask(prompt: str, default: str = "") -> str:
    val = input(prompt).strip()
    return val if val else default

def _ask_int(prompt: str, default: int = 0) -> int:
    while True:
        try:
            val = input(prompt).strip()
            return int(val) if val else default
        except ValueError:
            print("  [错误] 请输入整数。")

def _ask_float(prompt: str, default: float = 0.0) -> float:
    while True:
        try:
            val = input(prompt).strip()
            return float(val) if val else default
        except ValueError:
            print("  [错误] 请输入数字。")

def _ask_type(prompt: str) -> str:
    type_map = {"1": NODE_ROOM, "2": NODE_CORRIDOR,
                "3": NODE_STAIR, "4": NODE_EXIT}
    while True:
        val = input(prompt).strip()
        if val in type_map:
            return type_map[val]
        print("  [错误] 请输入 1/2/3/4。")


def input_building() -> Building:
    """
    交互式引导用户逐层输入建筑结构。

    stair 与 exit 的区别（重要）：
      stair  楼梯口 —— 连接上下楼层的通道入口。
             每层楼梯井对应一个 stair 节点，须与其他楼层
             同坐标的 stair 节点保持 x,y 一致，
             系统才会自动将它们跨层连通。
             stair 本身不是逃生终点。
      exit   安全出口 —— 建筑对外的疏散门。
             Dijkstra 以 exit 为终点，找到 exit 即视为逃生成功。
             通常只设在 1 层（地面层）。
    """
    print(SEP)
    print("  *** 建筑火灾动态逃生路线规划系统 ***")
    print(SEP)
    print("""
【节点类型说明】
  1  room      房间       普通功能空间
  2  corridor  走廊       连接各房间的通道节点
  3  stair     楼梯口     楼层间竖向通道入口
                          [*] 各层同坐标的 stair 会被自动跨层连通
                          [*] 本身不是出口，需走到 exit 才算逃生成功
  4  exit      安全出口   对外疏散门（逃生终点），通常只设在1层
""")

    name       = _ask("建筑名称（回车默认 '我的建筑'）：", "我的建筑")
    num_floors = _ask_int("楼层总数（2~10）：", 3)
    num_floors = max(2, min(10, num_floors))
    building   = Building(name)

    all_node_ids: Set[str] = set()   # 全局节点id，防止重复

    for f in range(1, num_floors + 1):
        print(f"\n{SEP2}")
        print(f"  第 {f} 层  ——  输入节点")
        print(SEP2)
        if f == 1:
            print("  提示：1层通常需要 exit 节点作为逃生终点。")
        else:
            print(f"  提示：若本层有楼梯口(stair)，其 x,y 须与其他楼层对应"
                  f" stair 相同，才能自动跨层连通。")

        fl = Floor(f)
        num_nodes = _ask_int(f"  第{f}层节点数量：", 1)

        for i in range(num_nodes):
            print(f"\n  节点 {i+1}/{num_nodes}")
            while True:
                nid = _ask(f"    节点ID（唯一字符串，如 F{f}_R1）：")
                if not nid:
                    print("    [错误] ID 不能为空。")
                elif nid in all_node_ids:
                    print(f"    [错误] ID '{nid}' 已存在，请换一个。")
                else:
                    break
            all_node_ids.add(nid)

            ntype = _ask_type("    类型 [1=room  2=corridor  3=stair  4=exit]：")
            x     = _ask_float("    x 坐标：")
            y     = _ask_float("    y 坐标：")
            label = _ask(f"    显示名称（回车使用ID '{nid}'）：", nid)
            fl.add_node(Node(id=nid, type=ntype, floor=f, x=x, y=y, label=label))

        print(f"\n{SEP2}")
        print(f"  第 {f} 层  ——  输入通道连线（边）")
        print(f"  本层节点：{list(fl.nodes.keys())}")
        print(SEP2)
        num_edges = _ask_int(f"  第{f}层边数量：", 0)

        for i in range(num_edges):
            print(f"\n  边 {i+1}/{num_edges}")
            while True:
                a = _ask("    节点A ID：")
                b = _ask("    节点B ID：")
                if a not in fl.nodes:
                    print(f"    [错误] '{a}' 不在本层节点中。")
                elif b not in fl.nodes:
                    print(f"    [错误] '{b}' 不在本层节点中。")
                elif a == b:
                    print("    [错误] 两端点不能相同。")
                else:
                    fl.connect(a, b)
                    print(f"    [OK] 已连接 {a} -- {b}")
                    break

        building.add_floor(fl)

    building.connect_stairs()
    building.build_graph()
    print(f"\n[OK] 建筑 '{name}' 构建完成："
          f"{len(building.all_nodes)} 个节点，{len(building.all_edges)} 条边")
    return building


def input_simulation(building: Building) -> Tuple[str, Set[str]]:
    """交互式输入用户当前位置与火灾危险区域"""
    nodes = building.all_nodes
    print(f"\n{SEP}")
    print("  模拟参数输入")
    print(SEP)
    print("\n所有节点：")
    for nid, n in sorted(nodes.items(), key=lambda x: (x[1].floor, x[0])):
        print(f"  [F{n.floor}] {nid:20s}  {n.label:20s}  type={n.type:9s}  ({n.x}, {n.y})")

    print()
    while True:
        start_id = _ask("您当前所在节点ID：")
        if start_id in nodes:
            break
        print(f"  [错误] '{start_id}' 不存在，请重新输入。")

    # 火灾位置输入
    print(f"""
火灾位置输入（每行一条，回车结束）：
  输入格式：
    n <节点ID>          -- 节点着火（封锁该节点所有边）
    e <节点A> <节点B>   -- 走廊中段着火（仅封锁这条边，两端节点仍可通）
  示例：
    n F2_C3             -- F2_C3 节点着火
    e F2_C1 F2_C2       -- F2_C1 和 F2_C2 之间的走廊中段着火
""")
 
    fire_node_ids  : Set[str]           = set()
    fire_edge_pairs: List[Tuple[str,str]] = []
 
    # 预建边索引方便校验（双向）
    edge_set = set()
    for e in building.all_edges:
        edge_set.add((e.node_a, e.node_b))
        edge_set.add((e.node_b, e.node_a))
 
    while True:
        raw = input("  火灾输入（回车结束）：").strip()
        if not raw:
            break
        parts = raw.split()
        if not parts:
            continue
 
        if parts[0].lower() == "n":
            # 节点火灾
            if len(parts) < 2:
                print("  [错误] 格式：n <节点ID>")
                continue
            fid = parts[1]
            if fid not in nodes:
                print(f"  [跳过] '{fid}' 节点不存在。")
            elif fid == start_id:
                print("  [跳过] 起点不能是火灾节点。")
            else:
                fire_node_ids.add(fid)
                print(f"  [OK] 节点着火：{nodes[fid].label}")
 
        elif parts[0].lower() == "e":
            # 边火灾
            if len(parts) < 3:
                print("  [错误] 格式：e <节点A> <节点B>")
                continue
            a, b = parts[1], parts[2]
            if a not in nodes:
                print(f"  [跳过] '{a}' 不存在。")
            elif b not in nodes:
                print(f"  [跳过] '{b}' 不存在。")
            elif (a, b) not in edge_set:
                print(f"  [跳过] '{a}' 和 '{b}' 之间没有边。")
            else:
                fire_edge_pairs.append((a, b))
                print(f"  [OK] 边着火：{nodes[a].label} -- {nodes[b].label}")
        else:
            print("  [错误] 请以 'n' 或 'e' 开头。")
 
    return start_id, fire_node_ids, fire_edge_pairs


# ============================================================
# 可视化
# ============================================================

def _node_style(
    node: Node, fire_zones: Set[str], path_set: Set[str], start_id: str
) -> Tuple[str, int, str]:
    """
    根据节点的当前状态返回 (颜色, 大小, 标记形状) 三元组，用于散点图绘制。

    优先级（从高到低）：
      起点(绿色五角星) > 火灾(红色三角) > 出口(绿色菱形)
      > 路径节点(亮绿圆) > 普通节点(类型对应色圆)
    """
    nid = node.id
    if nid == start_id:
        return COLOR_START, 250, "*"   # 绿色五角星 - 当前位置
    if nid in fire_zones:
        return COLOR_FIRE, 200, "^"    # 红色三角   - 火灾区域
    if node.type == NODE_EXIT:
        return COLOR_MAP[NODE_EXIT], 200, "D"  # 绿色菱形 - 安全出口
    if nid in path_set:
        return COLOR_PATH, 120, "o"    # 亮绿圆     - 逃生路径节点
    return COLOR_MAP.get(node.type, "#FFFFFF"), 70, "o"  # 普通节点（按类型着色）


# ── 可视化辅助函数 ────────────────────────────────────────────

def _dark_axes(ax, is_3d: bool = False) -> None:
    """统一应用深色主题到坐标轴"""
    ax.set_facecolor("#0d1b2a")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    if not is_3d:
        for sp in ax.spines.values():
            sp.set_color("#2a4060")
    if is_3d:
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = True
            pane.set_facecolor((0.02, 0.04, 0.10, 0.98))
            pane.set_edgecolor("#1a3050")


def _floor_outline_3d(ax, building: Building, global_limits: Tuple[float, float, float, float]) -> None:
    """
    global_limits: (x_min, x_max, y_min, y_max) 全局统一边界
    """
    x_min, x_max, y_min, y_max = global_limits
    margin = 0.8
    x0, x1 = x_min - margin, x_max + margin
    y0, y1 = y_min - margin, y_max + margin
    for fnum, floor in building.floors.items():
        z = (fnum - 1) * FLOOR_HEIGHT
        # 绘制矩形边线
        corners = [(x0,y0,z),(x1,y0,z),(x1,y1,z),(x0,y1,z),(x0,y0,z)]
        ax.plot([c[0] for c in corners],
                [c[1] for c in corners],
                [c[2] for c in corners],
                color="#1e4080", linewidth=1.2, alpha=0.5)
        # 半透明填充
        verts = [[(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]]
        poly = Poly3DCollection(verts, alpha=0.06,
                                facecolor="#2060a0", edgecolor="none")
        ax.add_collection3d(poly)
        # 楼层编号已由 z 轴刻度标注，此处不再额外绘制文字


def _make_legend_handles() -> List:
    """
    构建图例句柄列表，使用真实 marker（圆形/菱形/星形/三角），
    与节点在图中的实际形状完全一致。
    """
    def node_h(color, marker, label):
        return mlines.Line2D(
            [], [], linestyle="None",
            marker=marker, markersize=10,
            markerfacecolor=color,
            markeredgecolor="#000000" if marker != "*" else "none",
            markeredgewidth=0.6,
            label=label
        )

    def line_h(color, lw, ls, label):
        return mlines.Line2D([], [], color=color, linewidth=lw,
                             linestyle=ls, label=label)

    return [
        node_h(COLOR_MAP[NODE_ROOM],     "o", "Room (房间)"),
        node_h(COLOR_MAP[NODE_CORRIDOR], "o", "Corridor (走廊)"),
        node_h(COLOR_MAP[NODE_STAIR],    "o", "Stair (楼梯口)"),
        node_h(COLOR_MAP[NODE_EXIT],     "D", "Exit (安全出口)"),
        node_h(COLOR_START,              "*", "Your Location (当前位置)"),
        node_h(COLOR_FIRE,               "^", "Fire (火灾)"),
        line_h(COLOR_PATH,       3.5, "-",  "Escape Path (逃生路径)"),
        line_h(COLOR_EDGE_STAIR, 1.5, ":",  "Stairwell (楼梯跨层)"),
    ]


def visualize_3d(
    building    : Building,
    escape_path : Optional[List[str]] = None,
    fire_zones  : Optional[Set[str]]  = None,
    start_id    : Optional[str]       = None,
):
    """
    生成建筑的三维可视化图。
    - 每层绘制半透明矩形楼板，清晰区分楼层边界
    - 边按类别分批渲染：路径(绿色粗线) / 火灾(红色虚线) / 楼梯(橙色点线) / 普通(灰色)
    - 节点形状含义：圆形(room/corridor/stair) / 菱形(exit) / 星形(当前位置) / 三角(火灾)
    - 图例使用真实 marker 形状
    """
    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor("#080e1a")
    ax  = fig.add_subplot(111, projection="3d")
    _dark_axes(ax, is_3d=True)

    nodes      = building.all_nodes
    fire_zones = fire_zones or set()
    path_set   = set(escape_path) if escape_path else set()

    # 计算所有节点的全局 x,y 范围
    all_x = [n.x for n in building.all_nodes.values()]
    all_y = [n.y for n in building.all_nodes.values()]
    global_x_min, global_x_max = min(all_x), max(all_x)
    global_y_min, global_y_max = min(all_y), max(all_y)
    global_limits = (global_x_min, global_x_max, global_y_min, global_y_max)

    # ── 半透明楼板轮廓 ────────────────────────────────────
    _floor_outline_3d(ax, building, global_limits)

    # ── 边：按类型分 4 组渲染 ────────────────────────────
    seg_normal, seg_stair, seg_path, seg_fire = [], [], [], []
    for edge in building.all_edges:
        na = nodes.get(edge.node_a)
        nb = nodes.get(edge.node_b)
        if na is None or nb is None:
            continue
        seg = [na.pos3d, nb.pos3d]
        on_path = edge.node_a in path_set and edge.node_b in path_set
        if on_path:
            seg_path.append(seg)
        elif edge.fire_on_edge:
            seg_fire.append(seg)
        elif edge.is_stair:
            seg_stair.append(seg)
        else:
            seg_normal.append(seg)

    if seg_normal:
        ax.add_collection3d(Line3DCollection(
            seg_normal, colors=COLOR_EDGE, linewidths=0.9, alpha=0.55))
    if seg_stair:
        ax.add_collection3d(Line3DCollection(
            seg_stair, colors=COLOR_EDGE_STAIR,
            linewidths=2.2, alpha=0.85, linestyles="dotted"))
    if seg_fire:
        ax.add_collection3d(Line3DCollection(
            seg_fire, colors=COLOR_FIRE_EDGE,
            linewidths=2.5, alpha=1.0, linestyles="dashed"))
    # ── 火灾边中心三角标记 ────────────────────────────────
    for edge in building.all_edges:
        _na = nodes.get(edge.node_a)
        _nb = nodes.get(edge.node_b)
        if _na and _nb and edge.fire_on_edge:
            ax.scatter(
                (_na.x + _nb.x) / 2,
                (_na.y + _nb.y) / 2,
                (_na.z + _nb.z) / 2,
                c=COLOR_FIRE, s=200, marker="^",
                depthshade=False, zorder=7,
                edgecolors="#000000", linewidths=0.6)
    if seg_path:
        ax.add_collection3d(Line3DCollection(
            seg_path, colors=COLOR_PATH, linewidths=3.5, alpha=1.0))

    # ── 逃生路径方向箭头（线段中点，指向下一节点） ────────
    if escape_path and len(escape_path) > 1:
        for i in range(len(escape_path) - 1):
            na = nodes.get(escape_path[i])
            nb = nodes.get(escape_path[i + 1])
            if na is None or nb is None:
                continue
            dx = nb.x - na.x
            dy = nb.y - na.y
            dz = nb.z - na.z
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            if length < 1e-6:
                continue
            arrow_len = min(length * 0.38, 1.4)   # 箭头长度：线段的38%，上限1.4m
            ux, uy, uz = dx / length, dy / length, dz / length
            # 箭头起点 = 线段中点向后偏移半个箭头长度
            sx = (na.x + nb.x) / 2 - ux * arrow_len / 2
            sy = (na.y + nb.y) / 2 - uy * arrow_len / 2
            sz = (na.z + nb.z) / 2 - uz * arrow_len / 2
            ex = sx + ux * arrow_len
            ey = sy + uy * arrow_len
            ez = sz + uz * arrow_len
            arr = _Arrow3D(
                (sx, sy, sz), (ex, ey, ez),
                arrowstyle="-|>",
                color=COLOR_PATH,
                lw=2.0,
                mutation_scale=18,
                alpha=1.0,
                zorder=9,
            )
            ax.add_artist(arr)

    # ── 节点散点（按形状分类绘制）────────────────────────
    for nid, node in nodes.items():
        c, s, m = _node_style(node, fire_zones, path_set, start_id or "")
        # ── 路径节点先画实心圆盖帽，遮住路径线段端点 ──
        if nid in path_set:
            ax.scatter(
                *node.pos3d, c=COLOR_PATH, s=55, marker="o",
                depthshade=False, zorder=4, edgecolors="none", linewidths=0,
            )
        ax.scatter(
            *node.pos3d, c=c, s=s, marker=m,
            depthshade=False, zorder=6,
            edgecolors="#000000" if m != "*" else "none",
            linewidths=0.6
        )
        # ── 节点类型中文注释（高 zorder，不被连线遮盖） ────
        _TYPE_CN_3D = {"room": "房间", "corridor": "走廊",
                       "stair": "楼梯口", "exit": "安全出口"}
        if nid == (start_id or ""):
            _ann3d = "当前位置"
        elif nid in fire_zones:
            _ann3d = "火灾危险"
        else:
            _ann3d = _TYPE_CN_3D.get(node.type, node.type)
        ax.text(
            node.x + 0.25, node.y, node.z + 0.45,
            _ann3d, color=c, fontsize=7, ha="left", va="bottom",
            zorder=200,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#080e1a",
                      edgecolor="none", alpha=0.80),
        )

    # ── 坐标轴范围（防止节点贴边/裁剪） ─────────────────
    margin = 1.5
    all_x = [n.x for n in nodes.values()]
    all_y = [n.y for n in nodes.values()]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    floor_nums = sorted(building.floors.keys())
    ax.set_zlim(-0.5, (max(floor_nums) - 1) * FLOOR_HEIGHT + 1.5)

    # ── 坐标轴样式 ────────────────────────────────────────
    ax.set_xlabel("X (m)", color="#7799bb", labelpad=8)
    ax.set_ylabel("Y (m)", color="#7799bb", labelpad=8)
    ax.set_zlabel("Height (m)", color="#7799bb", labelpad=8)
    ax.xaxis.line.set_color("#1a3050")
    ax.yaxis.line.set_color("#1a3050")
    ax.zaxis.line.set_color("#1a3050")
    ax.grid(False)

    ax.set_zticks([(f - 1) * FLOOR_HEIGHT for f in floor_nums])
    ax.set_zticklabels([f"F{f}" for f in floor_nums], color="#aaaaaa")
    ax.view_init(elev=28, azim=-55)

    start_label = nodes[start_id].label if (start_id and start_id in nodes) else "?"
    ax.set_title(
        f"{building.name}  |  逃生路线规划  |  起点: {start_label}",
        fontsize=13, color="white", pad=16, fontweight="bold")

    # ── 图例（锚定在 3D 坐标轴内左上角） ─────────────────
    _leg3d = ax.legend(
        handles=_make_legend_handles(),
        loc="upper left", fontsize=9,
        facecolor="#091525", edgecolor="#2a5080",
        labelcolor="white", framealpha=0.96,
        handlelength=1.0, handleheight=1.0,
        handletextpad=0.5,
        borderpad=0.7, labelspacing=0.4,
        markerscale=1.0,
    )
    _leg3d.set_zorder(200)   # 确保图例文字不被任何三维连线遮盖
    return fig


def visualize_all_floors_2d(
    building    : Building,
    escape_path : Optional[List[str]] = None,
    fire_zones  : Optional[Set[str]]  = None,
    start_id    : Optional[str]       = None,
):
    """
    生成各楼层的 2D 平面图，按楼层横向排列拼接在同一画布中。
    - 逃生路径：亮绿色粗线（含光晕）
    - 火灾边：红色虚线
    - 普通边：灰色细线
    - 节点形状：圆形(room/corridor/stair) / 菱形(exit) / 星形(当前位置) / 三角(火灾)
    - 图例使用真实 marker 形状
    """
    num_floors = len(building.floors)
    fig, axes  = plt.subplots(1, num_floors,
                              figsize=(7.0 * num_floors, 7.5),
                              squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("#080e1a")

    fire_zones = fire_zones or set()
    path_set   = set(escape_path) if escape_path else set()
    all_nodes  = building.all_nodes

    # 预先将非楼梯边按楼层分组
    floor_edge_groups = defaultdict(list)
    for edge in building.all_edges:
        if edge.is_stair:
            continue
        na = all_nodes.get(edge.node_a)
        nb = all_nodes.get(edge.node_b)
        if na and nb:
            floor_edge_groups[na.floor].append((edge, na, nb))

    for idx, fnum in enumerate(sorted(building.floors.keys())):
        ax = axes[idx]
        _dark_axes(ax)

        # ── 绘制边（三类） ─────────────────────────────────
        for edge, na, nb in floor_edge_groups.get(fnum, []):
            on_path = edge.node_a in path_set and edge.node_b in path_set
            if on_path:
                # 逃生路径：与 3D 保持一致，无荧光，实线
                ax.plot([na.x, nb.x], [na.y, nb.y],
                        color=COLOR_PATH, linewidth=2.8, alpha=1.0,
                        solid_capstyle="round", zorder=3)
            elif edge.fire_on_edge:
                # 火灾边：红色虚线 + 中点三角标记
                ax.plot([na.x, nb.x], [na.y, nb.y],
                        color=COLOR_FIRE_EDGE, linewidth=2.2,
                        linestyle="--", alpha=1.0, zorder=3)
                mx, my = (na.x + nb.x) / 2, (na.y + nb.y) / 2
                ax.scatter(mx, my, c=COLOR_FIRE, s=150, marker="^",
                           zorder=6, edgecolors="#000000", linewidths=0.6)
            else:
                # 普通边：灰色细线
                ax.plot([na.x, nb.x], [na.y, nb.y],
                        color=COLOR_EDGE, linewidth=0.9,
                        alpha=0.55, zorder=1)

        # ── 逃生路径方向箭头（仅同层线段，画在线段中点） ─
        if escape_path and len(escape_path) > 1:
            for i in range(len(escape_path) - 1):
                pna = all_nodes.get(escape_path[i])
                pnb = all_nodes.get(escape_path[i + 1])
                if pna is None or pnb is None:
                    continue
                if pna.floor != fnum or pnb.floor != fnum:
                    continue
                seg_dx = pnb.x - pna.x
                seg_dy = pnb.y - pna.y
                seg_len = math.sqrt(seg_dx**2 + seg_dy**2)
                if seg_len < 1e-6:
                    continue
                ux2, uy2 = seg_dx / seg_len, seg_dy / seg_len
                alen = min(seg_len * 0.35, 1.2)
                mx2 = (pna.x + pnb.x) / 2
                my2 = (pna.y + pnb.y) / 2
                ax.annotate(
                    "",
                    xy=(mx2 + ux2 * alen / 2, my2 + uy2 * alen / 2),
                    xytext=(mx2 - ux2 * alen / 2, my2 - uy2 * alen / 2),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color=COLOR_PATH,
                        lw=2.0,
                        mutation_scale=18,
                    ),
                    zorder=8,
                )

        # ── 绘制节点 ──────────────────────────────────────
        floor_nodes = [(nid, node) for nid, node in all_nodes.items()
                       if node.floor == fnum]

        # 计算本层坐标范围用于设置轴范围
        fx = [n.x for _, n in floor_nodes] or [0]
        fy = [n.y for _, n in floor_nodes] or [0]
        pad = max(1.5, (max(fx) - min(fx)) * 0.18, (max(fy) - min(fy)) * 0.18)

        # 收集标签坐标以做简单防重叠（记录已用位置）
        label_positions: List[Tuple[float, float]] = []

        for nid, node in floor_nodes:
            c, s, m = _node_style(node, fire_zones, path_set, start_id or "")
            # ── 路径节点先画实心圆盖帽，遮住路径线段端点 ──
            if nid in path_set:
                ax.scatter(
                    node.x, node.y, c=COLOR_PATH, s=55, marker="o",
                    zorder=4, edgecolors="none", linewidths=0,
                )
            ax.scatter(
                node.x, node.y, c=c, s=s, marker=m, zorder=5,
                edgecolors="#000000" if m != "*" else "none",
                linewidths=0.6
            )
            # ── 节点类型中文注释（所有节点，高 zorder 不被连线遮盖） ──
            _TYPE_CN_2D = {"room": "房间", "corridor": "走廊",
                           "stair": "楼梯口", "exit": "安全出口"}
            if nid == (start_id or ""):
                _ann2d = "当前位置"
            elif nid in fire_zones:
                _ann2d = "火灾危险"
            else:
                _ann2d = _TYPE_CN_2D.get(node.type, node.type)
            ax.annotate(
                _ann2d,
                xy=(node.x, node.y),
                xytext=(7, 7),                  # 固定像素点偏移，与坐标尺度无关
                textcoords="offset points",
                fontsize=6, color=c, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", facecolor="#080e1a",
                          edgecolor="none", alpha=0.80),
                zorder=10,
            )
            
        ax.set_title(f"Floor {fnum}  第{fnum}层",
                     color="white", fontweight="bold", fontsize=12, pad=8)
        ax.set_xlabel("X (m)", color="#7799bb", fontsize=8)
        ax.set_ylabel("Y (m)", color="#7799bb", fontsize=8)
        ax.set_xlim(min(fx) - pad, max(fx) + pad)
        ax.set_ylim(min(fy) - pad, max(fy) + pad)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, color="#1a3050", alpha=0.5, linewidth=0.5)

    # ── 共享图例（底部居中，含真实 marker） ──────────────
    fig.legend(
        handles=_make_legend_handles(),
        loc="lower center", ncol=4, fontsize=8.5,
        facecolor="#091525", edgecolor="#2a5080",
        labelcolor="white", framealpha=0.96,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=1.0, handletextpad=0.5,
        columnspacing=1.5, borderpad=0.8,
    )
    fig.suptitle(f"{building.name}  —  各层平面图",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    return fig


# ============================================================
# 主流程
# ============================================================

def run():
    try:
        plt.rcParams["font.sans-serif"] = [
            "SimHei", "Arial Unicode MS", "WenQuanYi Micro Hei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 1. 输入建筑
    building = input_building()

    # 2. 输入模拟参数
    start_id, fire_zones, fire_edge_pairs = input_simulation(building)

    # 3. 路径规划
    print(f"\n{SEP}")
    print("  正在计算最优逃生路径（Dijkstra）...")
    print(SEP2)
    path = building.find_escape_route(start_id, fire_zones, fire_edge_pairs)

    if path:
        print("\n[成功] 找到最优逃生路径！\n")
        for i, nid in enumerate(path):
            n    = building.get_node(nid)
            icon = {"room":"[房间]","corridor":"[走廊]","stair":"[楼梯]","exit":"[出口]"}.get(n.type,"[?]")
            arrow = " ->" if i < len(path)-1 else ""
            print(f"  {i+1:2d}. {icon} [F{n.floor}] {n.label}{arrow}")
        print(f"\n  [出口] 已到达安全出口：{building.get_node(path[-1]).label}")
    else:
        print("\n[警告] 无法找到逃生路径！所有路径均被火灾封锁！")
        print("   建议：寻找最近窗口或等待救援。")
    print(SEP)

    # 4. 可视化
    print("\n正在生成可视化图表...")
    visualize_3d(building, escape_path=path,
                 fire_zones=fire_zones, start_id=start_id)
    visualize_all_floors_2d(building, escape_path=path,
                            fire_zones=fire_zones, start_id=start_id)
    plt.show()
    print("[OK] 可视化完成。")


if __name__ == "__main__":
    run()
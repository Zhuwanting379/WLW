"""
DXF → Building 转换模块（v3）
==============================
git 我将要接着测试
我已经创建了一个new banch
是的我又来测试了

将 AutoCAD DXF 文件解析并转换为 fire_escape_system.py 中的 Building 对象。
本模块只负责"构建建筑拓扑图"（两阶段流程的第一阶段），
不涉及火灾位置和当前位置——这两个参数在发生火灾后由
building.find_escape_route(start, fire_nodes, fire_edges) 传入。

支持两类 DXF 文件
------------------
A) 图层名含楼层号的标准多层 DXF（FLOOR1 / F-2 / L3 ...）
   → 直接从图层名提取楼层，无需额外配置

B) 「多平面平铺」DXF（国内建筑 CAD 常见格式）
   → 所有楼层平面图排列在同一个 modelspace 的不同 XY 区域，
     图层名是绘图分类号（'1','2','WALL' 等），不含楼层信息
   → 需通过 DXFImportConfig.floor_regions 手动指定每层的 XY 矩形范围

节点密集问题根因及解决方案
----------------------------
国内 CAD 建筑图纸常见如下情况导致节点密集堆叠：
  1. 所有楼层平面图并排在同一 modelspace → 用 floor_regions 分区
  2. 墙线 LINE 数量极大 → LINE 仅在楼梯/出口/走廊图层生成节点
  3. 窗/家具/标注/幅面等无关图层 → 黑/白名单过滤
  4. 台阶踏步等小面积闭合多边形 → min_room_area 阈值过滤
  5. 幅面边框等超大闭合多边形 → max_polygon_area_m2 阈值过滤

针对 building.dxf 的推荐配置（改造后首层～五层）
--------------------------------------------------
    from cad_import import DXFImportConfig, FloorRegion, cad_to_building

    cfg = DXFImportConfig(
        scale          = 0.001,    # DXF 单位：毫米
        min_room_area  = 8.0,      # 过滤台阶踏步格（约 3m²）
        floor_regions  = [
            # 坐标已换算为米（scale=0.001 后）
            # 标题位置确认：首层 X≈144m，二层 X≈208m，依次间距 65m
            FloorRegion(floor=1, x_min=99,  x_max=208, y_min=-210, y_max=55),
            FloorRegion(floor=2, x_min=208, x_max=272, y_min=-210, y_max=55),
            FloorRegion(floor=3, x_min=272, x_max=337, y_min=-210, y_max=55),
            FloorRegion(floor=4, x_min=337, x_max=401, y_min=-210, y_max=55),
            FloorRegion(floor=5, x_min=401, x_max=466, y_min=-210, y_max=55),
        ],
        structure_layers   = ['2', 'STAIR', 'WALL', '6', '8', '1'],
        extra_stair_layers = ['STAIR'],
        # 首层楼梯间（位于外墙处）可作为疏散出口
        exit_positions = [
            (1, 111.2, -62.0,  "北楼梯出口"),
            (1, 175.6, -62.0,  "南楼梯出口"),
        ],
    )
    building = cad_to_building("building.dxf", cfg)
    # building 对象存储完整拓扑图，可序列化保存

    # —— 发生火灾后 ——
    path = building.find_escape_route(
        start_id       = "F3_S1",
        fire_node_ids  = {"F3_C2"},
        fire_edge_pairs= [],
    )
"""
from __future__ import annotations


import re
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

try:
    import ezdxf
except ImportError:
    raise ImportError("ezdxf 未安装，请先执行: pip install ezdxf")

from fire_escape_system import (
    Building, Floor, Node, Edge,
    NODE_ROOM, NODE_CORRIDOR, NODE_STAIR, NODE_EXIT,
)

logger = logging.getLogger(__name__)


# ============================================================
# 楼层区域定义
# ============================================================

@dataclass
class FloorRegion:
    """
    用 XY 矩形框划定「多平面平铺」DXF 中某一楼层平面图的范围。
    坐标单位：米（已含 scale 缩放）。

    当一个实体的代表坐标（质心或首顶点）落在矩形内时，
    该实体归属此楼层。多个矩形重叠时取第一个匹配。

    Parameters
    ----------
    floor              : 楼层编号（从 1 开始）
    x_min/x_max        : X 坐标范围（米）
    y_min/y_max        : Y 坐标范围（米）
    """
    floor: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


# ============================================================
# 导入配置
# ============================================================

@dataclass
class DXFImportConfig:
    """
    DXF 导入配置。

    最常用参数
    ----------
    scale              : DXF 坐标单位→米。毫米图纸填 0.001
    floor_regions      : 楼层分区列表，为空则退回图层名正则识别楼层
    structure_layers   : 白名单图层（空=接受全部）
    exit_positions     : 手动出口坐标 [(floor, x_m, y_m, label), ...]
    min_room_area      : 闭合多边形最小面积（m²），小于此值丢弃
    max_polygon_area_m2: 闭合多边形最大面积（m²），超过此值丢弃（幅面边框）
    node_merge_tolerance: 节点合并距离（米）
    """

    # ── 坐标缩放 ─────────────────────────────────────────────────
    scale: float = 1.0

    # ── 楼层分区（多平面平铺 DXF） ───────────────────────────────
    floor_regions: List[FloorRegion] = field(default_factory=list)

    # ── 手动出口位置 ─────────────────────────────────────────────
    # [(floor, x_m, y_m, label), ...]
    exit_positions: List[Tuple] = field(default_factory=list)

    # ── 图层过滤 ─────────────────────────────────────────────────
    structure_layers: List[str] = field(default_factory=list)  # 白名单（空=全部）
    ignored_layers: List[str] = field(default_factory=lambda: [  # 黑名单（前缀匹配）
        "PUB_DIM", "PUB_TAB", "PUB_TEXT", "PUB_TITLE",
        "DIM_ELEV", "DIM_IDEN", "DIM_LEAD", "DIM_SYMB",
        "A_TEXT", "TK_", "图框", "AXIS",
        "T天花", "D铺砖", "fur", "WINDOW", "COLUMN", "BALCONY", "DOTE",
        "1拆除", "1砌筑", "1门槛石", "1救援窗",
    ])
    extra_stair_layers: List[str] = field(default_factory=lambda: ["STAIR"])

    # ── 图层名楼层正则（仅 floor_regions 为空时使用） ────────────
    floor_layer_patterns: List[str] = field(default_factory=lambda: [
        r"floor[-_\s]?(\d+)", r"fl[-_\s]?(\d+)", r"level[-_\s]?(\d+)",
        r"(?<![a-z])f[-_](\d+)", r"(?<![a-z])f(\d+)(?![a-z])",
        r"(\d+)[-_]?f(?![a-z])", r"l[-_](\d+)",
        r"(?<![a-z])l(\d+)(?![a-z])", r"(\d+)[-_]?l(?![a-z])", r"[-_](\d+)$",
    ])

    # ── 语义关键词 ───────────────────────────────────────────────
    exit_keywords:     List[str] = field(default_factory=lambda: [
        "exit", "escape", "evacuate", "emergency", "出口", "安全门", "疏散", "逃生"])
    stair_keywords:    List[str] = field(default_factory=lambda: [
        "stair", "stairs", "staircase", "stairway", "楼梯", "阶梯", "台阶"])
    room_keywords:     List[str] = field(default_factory=lambda: [
        "room", "office", "toilet", "wc", "bathroom", "lab",
        "房间", "房", "室", "办公", "卧室", "厕所", "实验"])
    corridor_keywords: List[str] = field(default_factory=lambda: [
        "corridor", "hallway", "hall", "passage", "aisle", "lobby",
        "走廊", "通道", "大厅", "过道"])
    door_to_exit_keywords: List[str] = field(default_factory=lambda: [
        "大门", "正门", "主入口", "入口", "楼门", "gate", "entrance", "main door"])
    door_exit_floor: int = 1

    # ── 几何阈值 ─────────────────────────────────────────────────
    node_merge_tolerance:  float = 0.5
    min_room_area:         float = 2.0    # m²，小于此值丢弃（墙厚矩形/台阶踏步）
    max_polygon_area_m2:   float = 500.0  # m²，大于此值丢弃（幅面边框/用地红线）
    max_circle_stair_r:    float = 3.0    # 圆半径 ≤ 此值 → 楼梯节点

    # ── 走廊-房间连接 ────────────────────────────────────────────
    room_corridor_connect_dist: float = 8.0

    # ── 建筑参数 ─────────────────────────────────────────────────
    floor_height:   float = 3.0
    building_name:  str   = "CAD导入建筑"
    default_floor:  int   = 1


# ============================================================
# 中间数据结构
# ============================================================

@dataclass
class _RawNode:
    x: float; y: float; floor: int; ntype: str
    label: str = ""; source: str = ""

@dataclass
class _RawEdge:
    ax: float; ay: float; bx: float; by: float; floor: int
    ntype: str = NODE_CORRIDOR


# ============================================================
# 纯工具函数
# ============================================================

def _d2(ax, ay, bx, by) -> float:
    return math.sqrt((ax-bx)**2 + (ay-by)**2)

def _poly_area(pts) -> float:
    a = 0.0
    n = len(pts)
    for i in range(n):
        x0,y0=pts[i]; x1,y1=pts[(i+1)%n]
        a += x0*y1 - x1*y0
    return abs(a)/2.0

def _poly_centroid(pts) -> Tuple[float,float]:
    n = len(pts)
    if not n: return 0.0, 0.0
    cx=cy=area=0.0
    for i in range(n):
        x0,y0=pts[i]; x1,y1=pts[(i+1)%n]
        c=x0*y1-x1*y0; area+=c; cx+=(x0+x1)*c; cy+=(y0+y1)*c
    area/=2.0
    if abs(area)<1e-10:
        return sum(p[0] for p in pts)/n, sum(p[1] for p in pts)/n
    return cx/(6.0*area), cy/(6.0*area)

def _project(px,py,ax,ay,bx,by) -> Tuple[float,float,float]:
    dx,dy=bx-ax,by-ay; sq=dx*dx+dy*dy
    if sq<1e-12: return ax,ay,0.0
    t=max(0.0,min(1.0,((px-ax)*dx+(py-ay)*dy)/sq))
    return ax+t*dx,ay+t*dy,t


# ============================================================
# 图层分类工具
# ============================================================

def _layer_floor(layer:str, cfg:DXFImportConfig, x:float, y:float) -> int:
    if cfg.floor_regions:
        for r in cfg.floor_regions:
            if r.contains(x, y): return r.floor
        return cfg.default_floor
    for pat in cfg.floor_layer_patterns:
        m = re.search(pat, layer, re.IGNORECASE)
        if m:
            try:
                v=int(m.group(1))
                if v>0: return v
            except: pass
    return cfg.default_floor

def _layer_type(layer:str, cfg:DXFImportConfig) -> Optional[str]:
    ll = layer.lower()
    for sl in cfg.extra_stair_layers:
        if sl.lower()==ll: return NODE_STAIR
    for kw in cfg.exit_keywords:
        if kw in ll: return NODE_EXIT
    for kw in cfg.stair_keywords:
        if kw in ll: return NODE_STAIR
    for kw in cfg.room_keywords:
        if kw in ll: return NODE_ROOM
    for kw in cfg.corridor_keywords:
        if kw in ll: return NODE_CORRIDOR
    return None

def _layer_skip(layer:str, cfg:DXFImportConfig) -> bool:
    ll = layer.lower()
    for ign in cfg.ignored_layers:
        if ll.startswith(ign.lower()): return True
    if cfg.structure_layers:
        return not any(ll==s.lower() for s in cfg.structure_layers)
    return False


# ============================================================
# DXF 解析器
# ============================================================

class DXFParser:
    """
    从 DXF modelspace 提取 _RawNode / _RawEdge。

    关键过滤规则（解决节点密集问题）
    ---------------------------------
    ① 黑/白名单跳过无关图层
    ② 闭合 LWPOLYLINE：面积在 [min_room_area, max_polygon_area_m2] 内才生成节点
    ③ 开放 LWPOLYLINE：仅走廊/楼梯/出口图层生成节点+边
    ④ LINE：仅楼梯/出口/走廊图层生成节点+边（墙线 LINE 不生成节点）
    ⑤ CIRCLE：按半径分类楼梯/房间
    """

    def __init__(self, cfg:DXFImportConfig):
        self.cfg = cfg
        self.raw_nodes: List[_RawNode] = []
        self.raw_edges: List[_RawEdge] = []
        self._labels: List[Tuple[float,float,str,str]] = []

    def _s(self, v): return v * self.cfg.scale

    def _floor(self, layer, x, y):
        return _layer_floor(layer, self.cfg, x, y)

    def parse(self, path:str):
        try:
            doc = ezdxf.readfile(path)
        except Exception:
            doc = ezdxf.readfile(path, encoding="gbk")
        msp = doc.modelspace()

        handlers = {
            "LWPOLYLINE": self._lwpoly,
            "LINE":       self._line,
            "POLYLINE":   self._polyline,
            "CIRCLE":     self._circle,
            "POINT":      self._point,
            "TEXT":       self._textent,
            "MTEXT":      self._textent,
        }
        etype_cnt: Dict[str,int] = defaultdict(int)
        skip_cnt:  Dict[str,int] = defaultdict(int)

        for ent in msp:
            et = ent.dxftype()
            etype_cnt[et] += 1
            h = handlers.get(et)
            if h is None: continue
            lyr = ent.dxf.layer
            if _layer_skip(lyr, self.cfg):
                skip_cnt[lyr] += 1; continue
            ntype = _layer_type(lyr, self.cfg)
            try: h(ent, lyr, ntype)
            except Exception as e:
                logger.debug("实体解析异常 %s@%s: %s", et, lyr, e)

        logger.debug("实体统计: %s", dict(sorted(etype_cnt.items())))
        logger.debug("跳过图层 TOP10: %s",
                     dict(sorted(skip_cnt.items(), key=lambda x:-x[1])[:10]))
        logger.debug("解析完毕: nodes=%d edges=%d",
                     len(self.raw_nodes), len(self.raw_edges))

        self._attach_labels()
        self._reclassify_exits()
        self._connect_rooms()
        logger.debug("后处理: nodes=%d edges=%d",
                     len(self.raw_nodes), len(self.raw_edges))

    # ── 实体处理器 ───────────────────────────────────────────────

    def _lwpoly(self, ent, lyr, ntype):
        raw_pts = list(ent.get_points("xy"))
        if len(raw_pts) < 2: return
        pts = [(self._s(p[0]), self._s(p[1])) for p in raw_pts]
        closed = ent.closed or (
            len(pts)>=3 and _d2(*pts[0],*pts[-1]) < self.cfg.node_merge_tolerance)

        if closed and len(pts)>=3:
            cx,cy = _poly_centroid(pts)
            fl    = self._floor(lyr, cx, cy)
            area  = _poly_area(pts)
            # ── 面积双向过滤（核心！）──────────────────────────
            if area < self.cfg.min_room_area:
                logger.debug("丢弃小多边形 %.2fm² @%s", area, lyr); return
            if area > self.cfg.max_polygon_area_m2:
                logger.debug("丢弃超大多边形 %.0fm² @%s", area, lyr); return
            resolved = ntype or NODE_ROOM
            self.raw_nodes.append(_RawNode(cx,cy,fl,resolved,source="LWPOLY_c"))
        else:
            # 开放折线：仅走廊/楼梯/出口图层
            resolved = ntype or NODE_CORRIDOR
            if resolved not in (NODE_CORRIDOR, NODE_STAIR, NODE_EXIT): return
            fl = self._floor(lyr, pts[0][0], pts[0][1])
            for p in pts:
                self.raw_nodes.append(_RawNode(p[0],p[1],fl,resolved,source="LWPOLY_o"))
            for i in range(len(pts)-1):
                self.raw_edges.append(
                    _RawEdge(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fl,resolved))

    def _line(self, ent, lyr, ntype):
        ax=self._s(ent.dxf.start.x); ay=self._s(ent.dxf.start.y)
        bx=self._s(ent.dxf.end.x);   by=self._s(ent.dxf.end.y)
        if _d2(ax,ay,bx,by)<1e-9: return
        resolved = ntype or NODE_CORRIDOR
        # ── LINE 过滤策略（核心！）──────────────────────────────
        # 墙线(ROOM/None图层) LINE 不生成节点，避免数千个密集端点
        # 只有走廊/楼梯/出口图层的 LINE 才生成节点+边
        if resolved not in (NODE_CORRIDOR, NODE_STAIR, NODE_EXIT): return
        fl = self._floor(lyr, ax, ay)
        self.raw_nodes.append(_RawNode(ax,ay,fl,resolved,source="LINE"))
        self.raw_nodes.append(_RawNode(bx,by,fl,resolved,source="LINE"))
        self.raw_edges.append(_RawEdge(ax,ay,bx,by,fl,resolved))

    def _polyline(self, ent, lyr, ntype):
        try:
            pts = [(self._s(p.x),self._s(p.y)) for p in ent.points()]
        except AttributeError:
            try:
                pts = [(self._s(v.dxf.location.x),self._s(v.dxf.location.y))
                       for v in ent.vertices
                       if hasattr(v,'dxf') and hasattr(v.dxf,'location')]
            except: return
        if len(pts)<2: return
        resolved = ntype or NODE_CORRIDOR
        if resolved not in (NODE_CORRIDOR, NODE_STAIR, NODE_EXIT): return
        fl = self._floor(lyr, pts[0][0], pts[0][1])
        for p in pts:
            self.raw_nodes.append(_RawNode(p[0],p[1],fl,resolved,source="POLY"))
        for i in range(len(pts)-1):
            self.raw_edges.append(
                _RawEdge(pts[i][0],pts[i][1],pts[i+1][0],pts[i+1][1],fl,resolved))

    def _circle(self, ent, lyr, ntype):
        cx=self._s(ent.dxf.center.x); cy=self._s(ent.dxf.center.y)
        r=self._s(ent.dxf.radius)
        if ntype is None:
            ntype = NODE_STAIR if r<=self.cfg.max_circle_stair_r else NODE_ROOM
        fl = self._floor(lyr, cx, cy)
        self.raw_nodes.append(_RawNode(cx,cy,fl,ntype,source="CIRCLE"))

    def _point(self, ent, lyr, ntype):
        x=self._s(ent.dxf.location.x); y=self._s(ent.dxf.location.y)
        fl = self._floor(lyr, x, y)
        self.raw_nodes.append(_RawNode(x,y,fl,ntype or NODE_CORRIDOR,source="POINT"))

    def _textent(self, ent, lyr, *_):
        try:
            if ent.dxftype()=="MTEXT":
                x=self._s(ent.dxf.insert.x); y=self._s(ent.dxf.insert.y)
                text=""
                for attr in ("text","plain_mtext"):
                    val=getattr(ent,attr,None)
                    if val is None: continue
                    text=val() if callable(val) else str(val)
                    if text: break
                text=re.sub(r"\{[^}]*\}|\\[A-Za-z0-9;]+","",text).strip()
            else:
                x=self._s(ent.dxf.insert.x); y=self._s(ent.dxf.insert.y)
                text=(ent.dxf.text or "").strip()
            if text: self._labels.append((x,y,lyr,text))
        except: pass

    # ── 后处理 ───────────────────────────────────────────────────

    def _attach_labels(self):
        """TEXT/MTEXT 关联到同楼层最近节点。"""
        for lx,ly,llyr,text in self._labels:
            lfl = _layer_floor(llyr, self.cfg, lx, ly)
            best=-1; best_d=float("inf")
            for i,rn in enumerate(self.raw_nodes):
                if rn.floor!=lfl: continue
                d=_d2(lx,ly,rn.x,rn.y)
                if d<best_d: best_d=d; best=i
            if best>=0 and not self.raw_nodes[best].label:
                self.raw_nodes[best].label=text

    def _reclassify_exits(self):
        """含大门/入口关键词的首层节点 → exit。"""
        kws=[kw.lower() for kw in self.cfg.door_to_exit_keywords if kw]
        if not kws: return
        cnt=0
        for rn in self.raw_nodes:
            if rn.floor!=self.cfg.door_exit_floor or rn.ntype==NODE_EXIT or not rn.label: continue
            ll=rn.label.lower()
            if any(kw and kw in ll for kw in kws):
                rn.ntype=NODE_EXIT; cnt+=1
        if cnt: logger.info("  标注→exit 重分类: %d 个", cnt)

    def _connect_rooms(self):
        """为每个 ROOM 节点在最近走廊线段上插入连接点（门洞节点）。"""
        max_d=self.cfg.room_corridor_connect_dist
        if max_d<=0: return
        segs_by_fl: Dict[int,List[Tuple[int,_RawEdge]]] = defaultdict(list)
        for i,seg in enumerate(self.raw_edges):
            if seg.ntype==NODE_CORRIDOR:
                segs_by_fl[seg.floor].append((i,seg))
        if not any(segs_by_fl.values()): return

        new_nodes:List[_RawNode]=[]; new_edges:List[_RawEdge]=[]; split:Set[int]=set()
        room_cnt=0
        for rn in self.raw_nodes:
            if rn.ntype!=NODE_ROOM: continue
            segs=segs_by_fl.get(rn.floor,[])
            if not segs: continue
            best_d=max_d; best_proj=None; best_t=0.0
            best_si=None; best_seg=None
            for si,seg in segs:
                qx,qy,t=_project(rn.x,rn.y,seg.ax,seg.ay,seg.bx,seg.by)
                d=_d2(rn.x,rn.y,qx,qy)
                if d<best_d:
                    best_d=d; best_proj=(qx,qy); best_t=t; best_si=si; best_seg=seg
            if best_proj is None: continue
            px,py=best_proj
            if _d2(rn.x,rn.y,px,py)<1e-6: continue
            room_cnt+=1
            new_nodes.append(_RawNode(px,py,rn.floor,NODE_CORRIDOR,source="JCT"))
            new_edges.append(_RawEdge(rn.x,rn.y,px,py,rn.floor,NODE_CORRIDOR))
            EPS=0.05
            if best_seg is not None and EPS<best_t<1.0-EPS:
                split.add(best_si)
                if _d2(best_seg.ax,best_seg.ay,px,py)>1e-6:
                    new_edges.append(_RawEdge(best_seg.ax,best_seg.ay,px,py,rn.floor,NODE_CORRIDOR))
                if _d2(px,py,best_seg.bx,best_seg.by)>1e-6:
                    new_edges.append(_RawEdge(px,py,best_seg.bx,best_seg.by,rn.floor,NODE_CORRIDOR))
        if split:
            self.raw_edges=[s for i,s in enumerate(self.raw_edges) if i not in split]
        self.raw_nodes.extend(new_nodes); self.raw_edges.extend(new_edges)
        if room_cnt: logger.info("  房间-走廊连接: %d 个", room_cnt)


# ============================================================
# 节点合并器（Union-Find）
# ============================================================

class _NodeMerger:
    _PRI = {NODE_EXIT:3, NODE_STAIR:2, NODE_ROOM:1, NODE_CORRIDOR:0}

    def __init__(self, tol:float): self.tol=tol

    def merge(self, raw:List[_RawNode]) -> List[_RawNode]:
        n=len(raw)
        if not n: return []
        par=list(range(n))

        def find(i):
            while par[i]!=i: par[i]=par[par[i]]; i=par[i]
            return i
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: par[rb]=ra

        buckets:Dict[int,List[int]]=defaultdict(list)
        for i,rn in enumerate(raw): buckets[rn.floor].append(i)
        for ids in buckets.values():
            for ii in range(len(ids)):
                for jj in range(ii+1,len(ids)):
                    ia,ib=ids[ii],ids[jj]
                    if find(ia)==find(ib): continue
                    if _d2(raw[ia].x,raw[ia].y,raw[ib].x,raw[ib].y)<self.tol:
                        union(ia,ib)

        groups:Dict[int,List[int]]=defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        out=[]
        for mi in groups.values():
            ms=[raw[i] for i in mi]
            mx=sum(m.x for m in ms)/len(ms); my=sum(m.y for m in ms)/len(ms)
            mt=max(ms,key=lambda m:self._PRI.get(m.ntype,0)).ntype
            lb=[m.label for m in ms if m.label]
            out.append(_RawNode(mx,my,ms[0].floor,mt,max(lb,key=len) if lb else ""))
        return out


# ============================================================
# 主入口
# ============================================================

def cad_to_building(
    dxf_path: str,
    cfg: Optional[DXFImportConfig] = None,
) -> Building:
    """
    从 DXF 文件构建 Building 拓扑对象（两阶段流程第一阶段）。

    返回的 Building 只包含建筑结构拓扑，不含任何火灾/位置信息。
    火灾发生后调用方自行传入：
        path = building.find_escape_route(start, fire_node_ids, fire_edge_pairs)

    Parameters
    ----------
    dxf_path : str   DXF 文件路径
    cfg      : DXFImportConfig  导入配置，None 时使用默认配置

    Returns
    -------
    Building  已完成 connect_stairs() + build_graph() 的建筑对象
    """
    if cfg is None: cfg = DXFImportConfig()
    logger.info("=== 解析 DXF: %s ===", dxf_path)

    # ── 1. 解析 ──────────────────────────────────────────────────
    parser = DXFParser(cfg)
    parser.parse(dxf_path)
    logger.info("步骤1 原始: nodes=%d edges=%d",
                len(parser.raw_nodes), len(parser.raw_edges))
    if not parser.raw_nodes:
        logger.warning("未提取到节点，返回空 Building。%s", _empty_hint(cfg))
        return Building(cfg.building_name)

    # ── 2. 节点合并 ───────────────────────────────────────────────
    merged = _NodeMerger(cfg.node_merge_tolerance).merge(parser.raw_nodes)
    logger.info("步骤2 合并后: nodes=%d (原%d)", len(merged), len(parser.raw_nodes))

    # ── 3. 建立楼层和节点 ─────────────────────────────────────────
    ABBR = {NODE_ROOM:"R", NODE_CORRIDOR:"C", NODE_STAIR:"S", NODE_EXIT:"E"}
    seq: Dict[Tuple,int] = defaultdict(int)
    id_map: List[str] = []
    fl_groups: Dict[int,List[int]] = defaultdict(list)
    for idx,rn in enumerate(merged):
        k=(rn.ntype,rn.floor); seq[k]+=1
        nid=f"F{rn.floor}_{ABBR.get(rn.ntype,'N')}{seq[k]}"
        id_map.append(nid); fl_groups[rn.floor].append(idx)

    building = Building(cfg.building_name)
    for fnum in sorted(fl_groups):
        fl = Floor(fnum)
        for idx in fl_groups[fnum]:
            rn=merged[idx]
            fl.add_node(Node(id=id_map[idx], type=rn.ntype, floor=fnum,
                             x=rn.x, y=rn.y, label=rn.label or id_map[idx]))
        building.add_floor(fl)
    logger.info("步骤3 楼层节点: %s",
                {f:len(fl.nodes) for f,fl in building.floors.items()})

    # ── 4. 边映射 ─────────────────────────────────────────────────
    edge_set: Set[Tuple[str,str]] = set()
    tol = cfg.node_merge_tolerance * 3.0

    def _nearest(x,y,fl_num) -> Optional[int]:
        best=None; bd=tol
        for i,rn in enumerate(merged):
            if rn.floor!=fl_num: continue
            d=_d2(x,y,rn.x,rn.y)
            if d<bd: bd=d; best=i
        return best

    def _add(fnum,ia,ib) -> bool:
        if ia==ib: return False
        k=(min(id_map[ia],id_map[ib]),max(id_map[ia],id_map[ib]))
        if k in edge_set: return False
        edge_set.add(k)
        na=building.get_node(id_map[ia]); nb=building.get_node(id_map[ib])
        if na and nb:
            building.floors[fnum].edges.append(Edge(id_map[ia],id_map[ib],weight=na.dist_to(nb)))
            return True
        return False

    mapped=0
    for re_ in parser.raw_edges:
        ia=_nearest(re_.ax,re_.ay,re_.floor)
        ib=_nearest(re_.bx,re_.by,re_.floor)
        if ia is not None and ib is not None and _add(re_.floor,ia,ib):
            mapped+=1
    logger.info("步骤4 边映射: %d / %d", mapped, len(parser.raw_edges))

    # ── 5. 孤立节点补边 ───────────────────────────────────────────
    iso=0
    for fnum,fl in building.floors.items():
        nids=list(fl.nodes.keys())
        if len(nids)<2: continue
        touched={n for pair in edge_set for n in pair if n in fl.nodes}
        for iso_id in [n for n in nids if n not in touched]:
            nd=building.get_node(iso_id)
            best_id=None; bd=float("inf")
            for oid in nids:
                if oid==iso_id: continue
                d=_d2(nd.x,nd.y,building.get_node(oid).x,building.get_node(oid).y)
                if d<bd: bd=d; best_id=oid
            if best_id:
                k=(min(iso_id,best_id),max(iso_id,best_id))
                if k not in edge_set:
                    edge_set.add(k)
                    na=building.get_node(iso_id); nb=building.get_node(best_id)
                    fl.edges.append(Edge(iso_id,best_id,weight=na.dist_to(nb)))
                    iso+=1
    if iso: logger.info("步骤5 孤立补边: %d 条", iso)

    # ── 6. 注入手动出口节点 ───────────────────────────────────────
    _inject_exits(building, cfg, edge_set)

    # ── 7. 同步全局边 + 图构建 ────────────────────────────────────
    building._edges = []
    for fl in building.floors.values():
        building._edges.extend(fl.edges)
    building.connect_stairs()
    building.build_graph()

    exits  = sum(1 for n in building.all_nodes.values() if n.type==NODE_EXIT)
    stairs = sum(1 for n in building.all_nodes.values() if n.type==NODE_STAIR)
    logger.info("步骤7 完成: 节点=%d 边=%d 出口=%d 楼梯=%d",
                len(building.all_nodes), len(building.all_edges), exits, stairs)

    if exits==0:
        logger.warning(
            "⚠ 未找到出口节点(exit)。\n"
            "  → 在 cfg.exit_positions 中手动指定出口坐标（单位：米）。\n"
            "  → 用 --log-level DEBUG 运行，查看各节点坐标后确定出口位置。")
    if stairs==0 and len(building.floors)>1:
        logger.warning(
            "⚠ 多层建筑未找到楼梯节点(stair)。\n"
            "  → 将楼梯图层名加入 extra_stair_layers（当前: %s）。",
            cfg.extra_stair_layers)
    return building


def _inject_exits(building:Building, cfg:DXFImportConfig,
                  edge_set:Set[Tuple[str,str]]):
    """将 cfg.exit_positions 中的手动出口注入 Building。"""
    if not cfg.exit_positions: return
    tol=cfg.node_merge_tolerance*2
    for entry in cfg.exit_positions:
        fnum,ex,ey,label = entry[0],entry[1],entry[2],entry[3]
        fl=building.floors.get(fnum)
        if fl is None:
            logger.warning("exit_positions: F%d 不存在，跳过 '%s'", fnum, label); continue
        best_id=None; bd=float("inf")
        for nid,nd in fl.nodes.items():
            d=_d2(ex,ey,nd.x,nd.y)
            if d<bd: bd=d; best_id=nid
        if bd<=tol:
            nd=building.get_node(best_id)
            nd.type=NODE_EXIT; nd.label=label or nd.label
            logger.info("  exit: 重分类 %s → exit ('%s')", best_id, label)
        else:
            seq=sum(1 for n in fl.nodes.values() if n.type==NODE_EXIT)+1
            new_id=f"F{fnum}_E{seq}"
            nn=Node(id=new_id,type=NODE_EXIT,floor=fnum,x=ex,y=ey,label=label or new_id)
            fl.add_node(nn); building._nodes[new_id]=nn
            logger.info("  exit: 新建 %s @ (%.1f,%.1f) '%s'", new_id, ex, ey, label)
            if best_id:
                k=(min(new_id,best_id),max(new_id,best_id))
                if k not in edge_set:
                    edge_set.add(k)
                    nb=building.get_node(best_id)
                    fl.edges.append(Edge(new_id,best_id,weight=_d2(ex,ey,nb.x,nb.y)))


def _empty_hint(cfg:DXFImportConfig) -> str:
    return (
        "\n  可能原因：\n"
        "  1. DXF 单位为毫米但未设置 scale=0.001\n"
        "  2. structure_layers 白名单过滤了所有有效图层\n"
        "  3. floor_regions 坐标范围不覆盖任何实体\n"
        "  4. DXF 仅含 ACAD_PROXY_ENTITY 等不支持的实体\n"
        "  建议：先以 structure_layers=[] 运行，查看 DEBUG 日志中的图层统计"
    )
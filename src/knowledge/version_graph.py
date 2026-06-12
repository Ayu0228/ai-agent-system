"""Knowledge Graph — 知识版本图，追踪知识的演进和依赖关系。

ref: Lilian Weng — knowledge graph as external long-term memory
ref: LangChain Graph — knowledge relationships for better retrieval

图结构:
  - Node: 知识条目
  - Edge: 关系（derives_from / contradicts / supports / references / supersedes）
  - 支持版本追踪: 知识条目更新后创建新版本节点，旧版本保留可追溯
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class EdgeType(str, Enum):
    DERIVES_FROM = "derives_from"       # A 从 B 推导而来
    CONTRADICTS = "contradicts"         # A 与 B 矛盾
    SUPPORTS = "supports"               # A 支持 B
    REFERENCES = "references"           # A 引用 B
    SUPERSEDES = "supersedes"           # A 替代 B (新版本)
    RELATED = "related"                 # 一般相关


@dataclass
class KnowledgeNode:
    """知识图谱节点。"""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    content: str = ""
    source: str = ""
    version: int = 1
    created_at: float = field(default_factory=time.time)
    confidence: float = 1.0             # 0-1 置信度
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """知识图谱边。"""

    source_id: str
    target_id: str
    type: EdgeType = EdgeType.RELATED
    weight: float = 1.0                 # 关系强度
    created_at: float = field(default_factory=time.time)
    evidence: str = ""                  # 关系证据/说明


class KnowledgeGraph:
    """知识版本图。

    用法:
        kg = KnowledgeGraph()
        n1 = kg.add_node(KnowledgeNode(title="Python created 1991"))
        n2 = kg.add_node(KnowledgeNode(title="Python 3.0 released 2008"))
        kg.add_edge(KnowledgeEdge(n2.id, n1.id, EdgeType.DERIVES_FROM))
        kg.add_edge(KnowledgeEdge(n2.id, n1.id, EdgeType.SUPERSEDES, weight=0.8))

        # 查询
        related = kg.get_related(n1.id, depth=2)
        contradictions = kg.find_contradictions()
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KnowledgeNode] = {}
        # source_id → [edge, ...]
        self._out_edges: dict[str, list[KnowledgeEdge]] = defaultdict(list)
        # target_id → [edge, ...]
        self._in_edges: dict[str, list[KnowledgeEdge]] = defaultdict(list)

    # ── 节点 ───────────────────────────────────────

    def add_node(self, node: KnowledgeNode) -> KnowledgeNode:
        self._nodes[node.id] = node
        logger.debug("kg_node_added", id=node.id, title=node.title[:50])
        return node

    def get_node(self, node_id: str) -> KnowledgeNode | None:
        return self._nodes.get(node_id)

    def update_node(self, node_id: str, content: str,
                    create_new_version: bool = True) -> KnowledgeNode:
        """更新节点 — 默认创建新版本节点。"""
        old = self._nodes.get(node_id)
        if not old:
            raise KeyError(f"node not found: {node_id}")

        if create_new_version:
            new_node = KnowledgeNode(
                title=old.title,
                content=content,
                source=old.source,
                version=old.version + 1,
                confidence=old.confidence,
                metadata=dict(old.metadata),
            )
            self.add_node(new_node)

            # 添加替代边
            self.add_edge(KnowledgeEdge(
                source_id=new_node.id,
                target_id=old.id,
                type=EdgeType.SUPERSEDES,
                weight=1.0,
                evidence=f"Version {old.version} → {new_node.version}",
            ))
            return new_node
        else:
            old.content = content
            old.version += 1
            return old

    # ── 边 ─────────────────────────────────────────

    def add_edge(self, edge: KnowledgeEdge) -> None:
        self._out_edges[edge.source_id].append(edge)
        self._in_edges[edge.target_id].append(edge)

    def remove_edge(self, source_id: str, target_id: str,
                    edge_type: EdgeType | None = None) -> int:
        """移除边，返回移除数量。"""
        count = 0
        for lst in [self._out_edges.get(source_id, []),
                     self._in_edges.get(target_id, [])]:
            to_remove = [e for e in lst
                         if e.source_id == source_id and e.target_id == target_id
                         and (edge_type is None or e.type == edge_type)]
            for e in to_remove:
                lst.remove(e)
                count += 1
        return count

    # ── 查询 ───────────────────────────────────────

    def get_related(self, node_id: str, depth: int = 1,
                    edge_types: list[EdgeType] | None = None) -> list[tuple[KnowledgeNode, KnowledgeEdge, int]]:
        """BFS 查询相关节点。返回 [(node, edge, distance), ...]"""
        if node_id not in self._nodes:
            return []

        visited: set[str] = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        results: list[tuple[KnowledgeNode, KnowledgeEdge, int]] = []

        while queue:
            current, dist = queue.popleft()
            if dist >= depth:
                continue

            for edge in self._out_edges.get(current, []) + self._in_edges.get(current, []):
                # 确定邻居方向
                neighbor_id = edge.target_id if edge.source_id == current else edge.source_id
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                neighbor = self._nodes.get(neighbor_id)
                if neighbor and (edge_types is None or edge.type in edge_types):
                    results.append((neighbor, edge, dist + 1))
                    queue.append((neighbor_id, dist + 1))

        return results

    def find_contradictions(self) -> list[tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]]:
        """找出所有矛盾关系。"""
        results: list[tuple[KnowledgeNode, KnowledgeNode, KnowledgeEdge]] = []
        for edges in self._out_edges.values():
            for edge in edges:
                if edge.type == EdgeType.CONTRADICTS:
                    src = self._nodes.get(edge.source_id)
                    tgt = self._nodes.get(edge.target_id)
                    if src and tgt:
                        results.append((src, tgt, edge))
        return results

    def get_version_chain(self, node_id: str) -> list[KnowledgeNode]:
        """获取节点的版本链（从最早到最新）。"""
        node = self._nodes.get(node_id)
        if not node:
            return []

        chain = [node]

        # 向前追溯：找更早的版本（当前节点替代了谁）
        current_id = node_id
        while True:
            found = False
            for edge in self._out_edges.get(current_id, []):
                if edge.type == EdgeType.SUPERSEDES:
                    older = self._nodes.get(edge.target_id)
                    if older:
                        chain.insert(0, older)
                        current_id = older.id
                        found = True
                        break
            if not found:
                break

        # 向后追溯：找更新的版本（谁替代了当前节点）
        current_id = node_id
        while True:
            found = False
            for edge in self._in_edges.get(current_id, []):
                if edge.type == EdgeType.SUPERSEDES:
                    newer = self._nodes.get(edge.source_id)
                    if newer:
                        chain.append(newer)
                        current_id = newer.id
                        found = True
                        break
            if not found:
                break

        return chain

    # ── 规则冲突检测 ───────────────────────────────

    def detect_rule_conflicts(self, new_trigger: str, new_action: str,
                               existing_rules: list[tuple[str, str, float]]) -> list[dict]:
        """检测新规则与已有规则的冲突。

        Args:
            new_trigger: 新规则的触发条件
            new_action: 新规则的动作
            existing_rules: [(trigger, action, confidence), ...]

        Returns: [{"conflict_with": index, "overlap": float, "auto_resolve": "keep_new"/"keep_old"/"conflict"}]
        """
        new_words = set(new_trigger.lower().split())
        if not new_words:
            return []

        conflicts: list[dict] = []
        for i, (trigger, action, confidence) in enumerate(existing_rules):
            existing_words = set(trigger.lower().split())
            if not existing_words:
                continue

            intersection = new_words & existing_words
            overlap = len(intersection) / max(len(new_words | existing_words), 1)

            # 触发条件相似（>50% 交集）但动作不同 → 冲突
            if overlap > 0.5 and new_action.strip().lower() != action.strip().lower():
                conflict = {"conflict_with": i, "overlap": round(overlap, 2)}

                # 自动裁决：confidence 差距 >= 0.2
                conf_diff = abs(new_trigger and 0.5 - confidence)  # new_confidence assumed 0.5
                if conf_diff >= 0.2:
                    conflict["auto_resolve"] = "keep_new"
                else:
                    conflict["auto_resolve"] = "conflict"  # 需人工审核

                conflicts.append(conflict)

        return conflicts

    # ── 统计 ───────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        edge_types = defaultdict(int)
        for edges in self._out_edges.values():
            for e in edges:
                edge_types[e.type.value] += 1

        return {
            "nodes": len(self._nodes),
            "edges": sum(len(e) for e in self._out_edges.values()),
            "by_edge_type": dict(edge_types),
            "contradictions": len(self.find_contradictions()),
        }

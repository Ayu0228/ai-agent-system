"""RAG 检索器：向量召回 → 阈值过滤 → 去重 → 拼接 Prompt。

嵌入 ChromaDB 默认的 all-MiniLM-L6-v2 做语义检索。
"""

from __future__ import annotations

import hashlib

import structlog

from src.memory.long_term import LongTermMemory
from src.shared.models import MemoryQuery

logger = structlog.get_logger()


class RAGRetriever:
    """检索增强生成：从知识库检索相关上下文，拼接到 Prompt 前。

    用法::

        retriever = RAGRetriever(top_k=3, threshold=0.4)
        results = retriever.retrieve("Python async best practices")
        context = retriever.format_context(results)
    """

    def __init__(self, *, top_k: int = 3, threshold: float = 0.4) -> None:
        self._top_k = top_k
        self._threshold = threshold
        self._memory = LongTermMemory()

    # ── Public API ────────────────────────────────────────

    def retrieve(self, query: str, *, agent_id: str | None = None) -> list[dict]:
        """检索相关知识片段。

        Returns:
            list of dicts: [{"content": ..., "score": ..., "source": ...}, ...]
        """
        if not query.strip():
            return []

        try:
            raw = self._memory.search(
                MemoryQuery(
                    query_text=query,
                    agent_id=agent_id,
                    top_k=self._top_k * 3,  # 多取一些给 rerank 留余量
                )
            )
        except Exception as e:
            logger.warning("rag_search_failed", error=str(e))
            return []

        # ── Rerank: 阈值过滤 + 去重 + 排序 ──
        seen_hashes: set[str] = set()
        filtered: list[dict] = []
        for r in raw:
            if r.score < self._threshold:
                continue
            content_hash = hashlib.md5(r.entry.content.encode()).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            filtered.append({
                "content": r.entry.content,
                "score": round(r.score, 3),
                "source": r.entry.agent_id,
                "type": r.entry.memory_type.value,
            })

        filtered.sort(key=lambda r: r["score"], reverse=True)
        result = filtered[: self._top_k]

        logger.debug(
            "rag_retrieved",
            query=query[:80],
            candidates=len(raw),
            after_rerank=len(result),
        )
        return result

    @staticmethod
    def format_context(results: list[dict]) -> str:
        """把检索结果拼成 Prompt 前缀。"""
        if not results:
            return ""

        lines = [
            "---",
            "以下是从知识库检索到的相关信息，请基于这些信息回答：",
            "",
        ]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] 相关度={r['score']:.2f} | 来源={r['source']} | 类型={r['type']}"
            )
            lines.append(f"    {r['content']}")
            lines.append("")

        lines.append("如果以上信息不足以回答问题，请在 error 字段说明。")
        lines.append("---")
        return "\n".join(lines)

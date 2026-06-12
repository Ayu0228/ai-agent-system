"""Knowledge lifecycle management — freshness tracking, staleness detection, archival.

ref: Lilian Weng — "LLM Powered Autonomous Agents" memory architecture
ref: LangChain — knowledge lifecycle with expiration and refresh policies
ref: MemGPT — tiered memory with active management
"""

from src.knowledge.lifecycle import KnowledgeLifecycle, KnowledgeItem, FreshnessStatus
from src.knowledge.version_graph import KnowledgeGraph, KnowledgeNode, KnowledgeEdge

__all__ = [
    "KnowledgeLifecycle", "KnowledgeItem", "FreshnessStatus",
    "KnowledgeGraph", "KnowledgeNode", "KnowledgeEdge",
]

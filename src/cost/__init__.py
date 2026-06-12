"""Cost management — token budget, model routing, cost attribution.

ref: ParetoBandit (arXiv 2604.00136) — budget-paced adaptive routing
ref: TokenTrimmer — context optimization via token trimming
ref: OrcaRouter / Claude-Code-LLM-Router (MIT) — intelligent model selection
ref: Anthropic cost tracking — per-model pricing with real-time aggregation
"""

from src.cost.budget import BudgetManager, BudgetStatus, BudgetTier
from src.cost.router import ModelRouter, RoutingDecision, ModelTier
from src.cost.attribution import CostTracker, UsageRecord, CostReport

__all__ = [
    "BudgetManager", "BudgetStatus", "BudgetTier",
    "ModelRouter", "RoutingDecision", "ModelTier",
    "CostTracker", "UsageRecord", "CostReport",
]

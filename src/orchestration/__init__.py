"""Multi-agent orchestration engine.

Patterns (ref: Microsoft Multi-Agent RA, AWS Strands Agents 1.0):
  - Agent Registry — service discovery, health checks, capability registration
  - Message Bus — pub/sub event bus, handoff protocol, reliable delivery
  - Orchestrator — sequential, parallel, swarm, graph execution modes
"""

from src.orchestration.registry import AgentRegistry, AgentInfo, AgentStatus
from src.orchestration.bus import MessageBus, Message, HandoffRequest
from src.orchestration.engine import Orchestrator, OrchestrationMode

__all__ = [
    "AgentRegistry", "AgentInfo", "AgentStatus",
    "MessageBus", "Message", "HandoffRequest",
    "Orchestrator", "OrchestrationMode",
]

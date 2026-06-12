"""Real-time streaming engine — SSE/WebSocket for agent outputs.

ref: OpenAI streaming API — SSE-based token streaming
ref: Anthropic — streaming messages with content_block_start/delta/stop events
ref: LangChain — streaming callbacks for real-time agent monitoring
"""

from src.streaming.engine import StreamingEngine, StreamEvent, StreamEventType
from src.streaming.gateway import EdgeGateway, GatewayConfig

__all__ = ["StreamingEngine", "StreamEvent", "StreamEventType", "EdgeGateway", "GatewayConfig"]

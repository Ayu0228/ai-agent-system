"""统一配置管理。所有可变参数通过环境变量注入，零硬编码。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，全部从环境变量/.env 读取。"""

    # LLM
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_fallback_model: str = ""

    # Storage
    chromadb_path: str = "./data/chromadb"
    sqlite_path: str = "./data/agent_system.db"

    # Logging
    log_level: str = "INFO"
    log_file: str = "./data/agent_system.log"

    # Budget
    daily_token_budget_per_agent: int = 50000
    daily_cost_budget: float = 2.00

    # Timeouts (seconds)
    agent_task_timeout: int = 900
    tool_call_timeout: int = 30
    approval_timeout: int = 1800

    # Feishu
    feishu_webhook_url: str = ""

    # MCP
    mcp_server_port: int = 8765

    # Task memory (L2 Warm)
    task_retention_hours: int = 72
    task_max_concurrent: int = 10

    # Experience memory (L3 Cold)
    experience_decay_half_life_days: int = 30
    experience_recency_boost_days: int = 7
    experience_success_bonus: float = 1.3
    experience_shareable_threshold: float = 0.9
    experience_retrieval_limit: int = 3

    # User memory (L3 Warm priority)
    user_max_facts: int = 200
    user_confidence_threshold: float = 0.7
    user_write_rate_limit: int = 10

    # Knowledge memory (L3 Cold)
    knowledge_similarity_baseline: float = 0.85
    knowledge_similarity_min: float = 0.7
    knowledge_similarity_max: float = 0.9
    knowledge_chunk_size: int = 512
    knowledge_chunk_overlap: int = 50

    # Main agent heartbeat
    main_heartbeat_interval: int = 60
    main_heartbeat_max_misses: int = 2

    # Idempotency
    idempotency_archive_days: int = 7

    # Context (L1 Hot)
    context_max_tokens: int = 128_000
    context_recent_turns: int = 20
    context_system_prompt_reserved: int = 4_000
    context_compress_threshold: float = 0.8
    context_max_budget: float = 0.3

    # Orchestrator
    orchestrator_max_concurrent: int = 20

    # Project marker — 项目水印，附加在所有输出末尾，用于识别响应来源
    project_marker: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def resolve_path(self, relative: str) -> Path:
        """将相对路径解析为基于项目根目录的绝对路径。"""
        base = Path(__file__).resolve().parents[2]  # src/shared/ → project root
        return (base / relative).resolve()


_project_root: Path | None = None
_project_root_lock = threading.Lock()


def get_project_root() -> Path:
    """返回项目根目录。从 src/shared/config.py 向上 2 级。"""
    global _project_root
    if _project_root is None:
        with _project_root_lock:
            if _project_root is None:
                _project_root = Path(__file__).resolve().parents[2]
    return _project_root


# 全局单例（双重检查锁）
_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings

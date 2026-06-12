"""Pydantic 数据模型。定义系统中所有核心数据结构。"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════


class MemoryType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    EXPERIENCE = "experience"
    PREFERENCE = "preference"
    TASK_CONTEXT = "task_context"
    KNOWLEDGE = "knowledge"
    USER_FACT = "user_fact"
    TASK_STATE = "task_state"


class MemoryLayer(str, Enum):
    """记忆层级（L1/L2/L3 三层存储）。"""
    L1_HOT = "L1"
    L2_WARM = "L2"
    L3_COLD = "L3"


class TaskCognitiveState(str, Enum):
    """任务认知状态机（5 状态）。"""
    INTENT_RECOGNITION = "intent_recognition"
    PLANNING = "planning"
    EXECUTING_TOOLS = "executing_tools"
    OBSERVING_RESULTS = "observing_results"
    DONE = "done"
    FAILED = "failed"


class ConflictStatus(str, Enum):
    PENDING = "pending"
    AUTO_MERGED = "auto_merged"
    LLM_RESOLVED = "llm_resolved"
    MANUAL = "manual"


class SelfImprovingTrack(str, Enum):
    SELF_IMPROVING = "self-improving"
    AUTOSCALE = "autoscale"


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEGRADED = "degraded"  # 超时降级：直接写入+标记待审核


class StepType(str, Enum):
    TASK = "task"
    CONDITION = "condition"
    PARALLEL = "parallel"
    LOOP = "loop"
    HUMAN = "human"


class RetryStrategy(str, Enum):
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    SWITCH_MODEL = "switch_model"


class RiskLevel(int, Enum):
    L0 = 0  # 读操作 → 自动通过
    L1 = 1  # 安全写 → 自动通过 + 通知
    L2 = 2  # 风险写 → 飞书确认
    L3 = 3  # 危险操作 → 飞书 + 确认码


class HookDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    ESCALATE = "escalate"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"


class WriteDecision(str, Enum):
    WRITE = "write"
    SKIP = "skip"
    MERGE = "merge"


# ═══════════════════════════════════════════════════════════════
# 记忆系统
# ═══════════════════════════════════════════════════════════════


class MemoryEntry(BaseModel):
    """记忆条目。"""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    content: str = Field(max_length=2000)
    memory_type: MemoryType = MemoryType.FACT
    tags: list[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    source_trace_id: str | None = None
    created_at: str = Field(default_factory=_now)
    last_accessed_at: str = Field(default_factory=_now)
    access_count: int = 0
    expires_at: str | None = None
    chroma_id: str | None = None


class MemoryQuery(BaseModel):
    """记忆查询参数。"""

    query_text: str = ""
    agent_id: str | None = None
    memory_types: list[MemoryType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    min_importance: float = 0.0
    top_k: int = Field(default=5, ge=1, le=20)
    time_range_start: str | None = None
    time_range_end: str | None = None


class MemorySearchResult(BaseModel):
    """记忆搜索结果。"""

    entry: MemoryEntry
    score: float


class WriteEvaluation(BaseModel):
    """写入决策评估结果。"""

    decision: WriteDecision
    importance: float
    reason: str


# ═══════════════════════════════════════════════════════════════
# 工作流
# ═══════════════════════════════════════════════════════════════


class StepConfig(BaseModel):
    """工作流步骤配置。"""

    id: str
    name: str = ""
    agent: str | None = None
    type: StepType = StepType.TASK
    depends_on: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """步骤执行结果。"""

    step_id: str
    status: str = "success"  # success / failed / timeout / skipped
    output: Any = None
    error: str | None = None
    tokens_used: int = 0
    duration_ms: int = 0
    retry_count: int = 0
    trace_id: str = ""


class WorkflowDefinition(BaseModel):
    """工作流定义。"""

    name: str
    version: str = "1.0"
    description: str = ""
    trigger: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepConfig] = Field(default_factory=list)


class WorkflowResult(BaseModel):
    """工作流执行结果。"""

    workflow_name: str
    status: str = "pending"  # pending / completed / failed / cancelled
    steps: list[StepResult] = Field(default_factory=list)
    final_output: Any = None
    total_tokens: int = 0
    total_duration_ms: int = 0
    trace_id: str = ""


# ═══════════════════════════════════════════════════════════════
# 评估
# ═══════════════════════════════════════════════════════════════


class EvalDimension(str, Enum):
    GROUNDING = "grounding"
    UX_QUALITY = "ux_quality"
    SAFETY = "safety"


class JudgeResult(BaseModel):
    """LLM-as-Judge 评分结果。"""

    dimension: EvalDimension
    score: float = Field(ge=0.0, le=5.0)
    reasoning: str = ""
    uncertain: bool = False


class TrajectoryScore(BaseModel):
    """轨迹评估分数。"""

    tool_selection: float = 0.0
    param_accuracy: float = 0.0
    reasoning_chain: float = 0.0
    error_handling: float = 0.0
    efficiency: float = 0.0
    overall: float = 0.0
    hidden_failure: bool = False
    hidden_failure_markers: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """评估报告。"""

    agent_id: str
    test_set: str
    total_cases: int
    passed: int
    failed: int
    scores: dict[str, float] = Field(default_factory=dict)
    degraded_from_baseline: bool = False
    degradation_details: list[str] = Field(default_factory=list)
    trace_id: str = ""


# ═══════════════════════════════════════════════════════════════
# 审批
# ═══════════════════════════════════════════════════════════════


class ApprovalRequest(BaseModel):
    """审批请求。"""

    id: str = Field(default_factory=_new_id)
    agent_id: str
    action: str
    risk_level: RiskLevel
    summary: str
    params_hash: str = ""
    created_at: str = Field(default_factory=_now)
    status: str = "pending"  # pending / approved / rejected / timeout


class ApprovalResponse(BaseModel):
    """审批响应。"""

    request_id: str
    approved: bool
    reason: str = ""
    approver: str = ""
    resolved_at: str = Field(default_factory=_now)


# ═══════════════════════════════════════════════════════════════
# Agent 间消息格式（所有 Agent 必须使用此格式传递消息）
# ═══════════════════════════════════════════════════════════════


class AgentMessage(BaseModel):
    """Agent 间标准消息格式。所有 Agent 的输出都必须符合此格式。"""

    agent: str = ""                     # 发送方 Agent ID
    status: str = "success"            # success / error / blocked
    summary: str = ""                  # 一句话总结
    confidence: str = "medium"         # 置信度：high（有可靠依据）/ medium（有线索不确定）/ low（推测或无依据）
    data: dict[str, Any] = Field(default_factory=dict)   # 结构化输出
    error: str | None = None           # 错误描述（status=error 时必填）


# ═══════════════════════════════════════════════════════════════
# 审计与安全
# ═══════════════════════════════════════════════════════════════


class AuditEntry(BaseModel):
    """审计日志条目。"""

    ts: str = Field(default_factory=_now)
    agent_id: str
    event: str  # tool_call / approval_requested / approval_resolved / error
    tool: str | None = None
    params_hash: str | None = None
    result_code: str | None = None
    tokens: int = 0
    duration_ms: int = 0
    trace_id: str = ""
    risk_level: str = "L0"


class BudgetStatus(BaseModel):
    """预算状态。"""

    agent_id: str
    tokens_used_today: int
    tokens_remaining: int
    is_exceeded: bool
    budget_reset_at: str


# ═══════════════════════════════════════════════════════════════
# 经验
# ═══════════════════════════════════════════════════════════════


class Experience(BaseModel):
    """经验条目。（已弃用，由 ExperienceCard 替代。保留以兼容旧数据。）

    Phase 2 计划删除此冗余类。
    """

    id: str = Field(default_factory=_new_id)
    agent_id: str
    task_type: str
    trigger: str
    symptom: str
    root_cause: str
    solution: str
    outcome: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    validated: bool = False
    created_at: str = Field(default_factory=_now)
    last_applied_at: str | None = None
    apply_count: int = 0


# ═══════════════════════════════════════════════════════════════
# 规则记忆
# ═══════════════════════════════════════════════════════════════


class RuleStatus(str, Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    DEPRECATED = "deprecated"
    CONFLICT = "conflict"


class ScopeLevel(str, Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"
    GLOBAL = "global"


class Rule(BaseModel):
    """规则记忆条目 — WHEN trigger THEN action 的可执行规则。

    Phase 2 预留：文档方案中规则记忆的功能由 ExperienceMemory + SelfEvolution 覆盖。
    待 Phase 2 评估是否需要独立的规则推理引擎时再启用。
    """

    id: str = Field(default_factory=_new_id)
    name: str = ""
    description: str = ""
    trigger_condition: str = ""   # WHEN — 触发条件描述
    action: str = ""              # THEN — 执行动作
    scope: ScopeLevel = ScopeLevel.AGENT
    agent_id: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    fire_count: int = 0
    last_fired_at: str | None = None
    cooldown_seconds: int = 300
    status: RuleStatus = RuleStatus.ACTIVE
    source_experience_id: str | None = None
    created_at: str = Field(default_factory=_now)
    expires_at: str = ""
    tags: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════════════════════════════


class ToolParamType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class ToolParamSchema(BaseModel):
    """工具参数定义。"""

    name: str
    type: ToolParamType = ToolParamType.STRING
    description: str = ""
    required: bool = False
    default: Any = None
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
    """工具定义 — 替代裸字符串的工具描述。"""

    name: str
    description: str = ""
    parameters: list[ToolParamSchema] = Field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.L1
    requires_approval: bool = False
    max_retries: int = 1
    timeout_s: int = 30

    def validate(self, params: dict[str, Any]) -> "ToolValidationResult":
        """校验参数是否符合 schema。"""
        errors: list[str] = []
        warnings: list[str] = []
        missing_required: list[str] = []
        unknown_params: list[str] = []
        type_mismatches: list[str] = []

        schema_map = {p.name: p for p in self.parameters}

        # 检查必填
        for p in self.parameters:
            if p.required and p.name not in params:
                missing_required.append(p.name)
                errors.append(f"Missing required param: {p.name}")

        # 检查传入参数
        for key, value in params.items():
            schema = schema_map.get(key)
            if schema is None:
                unknown_params.append(key)
                warnings.append(f"Unknown param: {key}")
                continue

            # 类型检查
            expected = schema.type
            if expected == ToolParamType.STRING and not isinstance(value, str):
                type_mismatches.append(f"{key}: expected string, got {type(value).__name__}")
            elif expected == ToolParamType.NUMBER and not isinstance(value, (int, float)):
                type_mismatches.append(f"{key}: expected number, got {type(value).__name__}")
            elif expected == ToolParamType.BOOLEAN and not isinstance(value, bool):
                type_mismatches.append(f"{key}: expected boolean, got {type(value).__name__}")
            elif expected == ToolParamType.ARRAY and not isinstance(value, list):
                type_mismatches.append(f"{key}: expected array, got {type(value).__name__}")
            elif expected == ToolParamType.OBJECT and not isinstance(value, dict):
                type_mismatches.append(f"{key}: expected object, got {type(value).__name__}")

            # enum 约束检查
            if schema.enum and isinstance(value, str) and value not in schema.enum:
                errors.append(f"{key}: '{value}' not in allowed values: {schema.enum}")

        errors.extend(type_mismatches)
        valid = len(errors) == 0

        return ToolValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            missing_required=missing_required,
            unknown_params=unknown_params,
            type_mismatches=type_mismatches,
        )


class ToolValidationResult(BaseModel):
    """工具参数校验结果。"""

    valid: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    unknown_params: list[str] = Field(default_factory=list)
    type_mismatches: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 任务记忆（L2 Warm）
# ═══════════════════════════════════════════════════════════════


class TaskRecord(BaseModel):
    """任务记录 — SQLite tasks 表的模型。"""

    task_id: str = Field(default_factory=_new_id)
    session_step_id: str = ""
    owner_agent_id: str
    parent_task_id: str | None = None
    status: TaskCognitiveState = TaskCognitiveState.INTENT_RECOGNITION
    subtasks: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    acl: dict[str, str] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)


class TaskStateTransition(BaseModel):
    """认知状态机日志。"""

    task_id: str
    from_state: str
    to_state: str
    session_step_id: str
    timestamp: str = Field(default_factory=_now)


# ═══════════════════════════════════════════════════════════════
# 幂等记录
# ═══════════════════════════════════════════════════════════════


class IdempotencyRecord(BaseModel):
    """幂等记录 — agent_idempotency 表。"""

    session_step_id: str
    user_id: str = ""
    goal_hash: str = ""
    status: str = "processing"
    final_result: str | None = None
    created_at: str = Field(default_factory=_now)


# ═══════════════════════════════════════════════════════════════
# 经验记忆（L3 Cold）
# ═══════════════════════════════════════════════════════════════


class ExperienceCard(BaseModel):
    """经验卡片 — 任务完成后 LLM 自动生成。"""

    experience_id: str = Field(default_factory=_new_id)
    owner_agent_id: str
    scenario: str = ""
    approach: str = ""
    result: str = ""
    lesson: str = ""
    tags: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)
    shareable: bool = False
    usage_count: int = 0
    success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    self_improving_track: SelfImprovingTrack = SelfImprovingTrack.SELF_IMPROVING
    autoscale_eligible: bool = False
    created_at: str = Field(default_factory=_now)
    last_accessed_at: str = Field(default_factory=_now)
    task_id: str | None = None


class ExperienceQuery(BaseModel):
    """经验检索查询。"""

    query_text: str = ""
    agent_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=10)
    include_shared: bool = True
    is_success: bool | None = None


# ═══════════════════════════════════════════════════════════════
# 知识记忆（L3 Cold）
# ═══════════════════════════════════════════════════════════════


class KnowledgeTriple(BaseModel):
    """知识图谱三元组。"""

    id: str = Field(default_factory=_new_id)
    subject: str
    predicate: str
    object: str
    source: str = ""
    confidence_weight: float = Field(default=1.0, ge=0.0)
    created_at: str = Field(default_factory=_now)
    chroma_entity_id: str | None = None


class ConflictRecord(BaseModel):
    """知识冲突记录。"""

    id: str = Field(default_factory=_new_id)
    triple_a_id: str
    triple_b_id: str
    conflict_type: str  # contradiction / duplicate
    status: ConflictStatus = ConflictStatus.PENDING
    resolved_by: str | None = None
    resolved_at: str | None = None
    created_at: str = Field(default_factory=_now)


class KnowledgeQuery(BaseModel):
    """知识检索查询。"""

    query_text: str
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float | None = None  # None = 动态阈值
    include_triples: bool = True
    is_complex_query: bool = False  # 简单查询→高阈值, 复杂查询→低阈值


# ═══════════════════════════════════════════════════════════════
# 用户记忆（L3 Warm Priority）
# ═══════════════════════════════════════════════════════════════


class UserFact(BaseModel):
    """用户事实记录。"""

    id: str = Field(default_factory=_new_id)
    user_id: str
    entity: str
    predicate: str
    object: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    valid_from: str = Field(default_factory=_now)
    valid_to: str | None = None
    source_agent: str = ""
    write_decision: str = "pending"  # pending / approved / rejected / degraded
    created_at: str = Field(default_factory=_now)
    access_count: int = 0
    last_accessed_at: str = Field(default_factory=_now)


class MemoryProposal(BaseModel):
    """用户记忆写入提案 — 其他 agent 提交给 main 审核。"""

    id: str = Field(default_factory=_new_id)
    user_id: str
    entity: str
    predicate: str
    object: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_agent: str
    status: ProposalStatus = ProposalStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    created_at: str = Field(default_factory=_now)


class UserMemoryQuery(BaseModel):
    """用户记忆查询。"""

    user_id: str | None = None
    entity: str | None = None
    top_k: int = Field(default=10, ge=1, le=50)
    min_confidence: float = 0.0
    active_only: bool = True  # 只返回 valid_to 为空的当前事实


# ═══════════════════════════════════════════════════════════════
# 检索路由
# ═══════════════════════════════════════════════════════════════


class RetrievalPriority(BaseModel):
    """检索路由优先级配置。"""

    memory_type: MemoryType
    weight: float
    layer: MemoryLayer
    description: str = ""


class RetrievalBudget(BaseModel):
    """Token 预算分配。"""

    total_tokens: int = 128_000
    context_max: float = 0.3  # 30%
    system_prompt: float = 0.1  # 10%
    generation: float = 0.2  # 20%


# ═══════════════════════════════════════════════════════════════
# 自进化
# ═══════════════════════════════════════════════════════════════


class EvolutionRecord(BaseModel):
    """自进化记录。"""

    id: str = Field(default_factory=_new_id)
    track: SelfImprovingTrack
    source_experience_id: str | None = None
    content: str = ""  # Learnings.md 条目或 Skill.md 内容
    trigger_reason: str = ""  # 失败/纠正/反馈/高频模式
    promote_target: str | None = None  # AGENTS.md / skill 文件路径
    created_at: str = Field(default_factory=_now)
    last_updated_at: str = Field(default_factory=_now)


# ═══════════════════════════════════════════════════════════════
# 监控指标
# ═══════════════════════════════════════════════════════════════


class MemoryMetrics(BaseModel):
    """记忆系统监控指标。"""

    retrieval_latency_l1_ms: float = 0.0
    retrieval_latency_l2_ms: float = 0.0
    retrieval_latency_l3_ms: float = 0.0
    write_queue_length: int = 0
    idempotency_conflict_rate: float = 0.0
    heartbeat_success_rate: float = 1.0
    storage_usage_pct: float = 0.0
    embedding_rate_per_min: float = 0.0
    total_memories: int = 0
    total_tasks: int = 0
    total_experiences: int = 0
    total_user_facts: int = 0

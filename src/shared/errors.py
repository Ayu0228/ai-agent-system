"""自定义异常层级。所有异常继承自 AgentSystemError。"""


class AgentSystemError(Exception):
    """Agent 系统根异常。"""


class ConfigError(AgentSystemError):
    """配置相关错误。"""


class MemoryError(AgentSystemError):
    """记忆系统错误。"""


class MemoryUnavailableError(MemoryError):
    """记忆存储不可用。"""


class MemoryWriteError(MemoryError):
    """记忆写入失败。"""


class MemorySearchError(MemoryError):
    """记忆检索失败。"""


class WorkflowError(AgentSystemError):
    """工作流编排错误。"""


class WorkflowParseError(WorkflowError):
    """工作流 YAML 解析失败。"""


class WorkflowExecutionError(WorkflowError):
    """工作流执行失败。"""


class StepTimeoutError(WorkflowExecutionError):
    """步骤执行超时。"""


class CircularDependencyError(WorkflowError):
    """检测到循环依赖。"""


class EvaluationError(AgentSystemError):
    """评估系统错误。"""


class ApprovalError(AgentSystemError):
    """审批系统错误。"""


class ApprovalTimeoutError(ApprovalError):
    """审批超时。"""


class SafetyError(AgentSystemError):
    """安全系统错误。"""


class PromptInjectionDetected(SafetyError):
    """检测到 Prompt 注入攻击。"""


class UnauthorizedOperationError(SafetyError):
    """越权操作。"""


class BudgetExceededError(AgentSystemError):
    """Token / 成本预算超出。"""


class ExperienceError(AgentSystemError):
    """经验学习系统错误。"""


class LLMError(AgentSystemError):
    """LLM 调用错误。"""


class LLMTimeoutError(LLMError):
    """LLM 调用超时。"""


class LLMRateLimitError(LLMError):
    """LLM 速率限制。"""


class ToolError(AgentSystemError):
    """工具调用错误。"""


class ToolNotFoundError(ToolError):
    """工具未注册。"""


class ToolExecutionError(ToolError):
    """工具执行失败。"""


class BudgetBlockedPlanError(AgentSystemError):
    """预算不足导致规划被阻断。"""

    def __init__(self, tier: str = "", limit: int = 0, used: int = 0) -> None:
        self.tier = tier
        self.limit = limit
        self.used = used
        msg = f"Budget blocked: tier={tier}, limit={limit}, used={used}"
        super().__init__(msg)

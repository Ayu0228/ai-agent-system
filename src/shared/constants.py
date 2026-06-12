"""常量定义。"""

# Agent ID 白名单
AGENT_IDS = frozenset(
    {
        "main",
        "researcher",
        "tech-dev",
        "copywriter",
        "script-editor",
        "data-analyst",
        "visual-designer",
        "product-designer",
        "ops-monitor",
        "investment-analyst",
        "content-strategist",
    }
)

# 记忆类型
MEMORY_TYPES = frozenset(
    {"fact", "decision", "experience", "preference", "task_context"}
)

# 步骤类型
STEP_TYPES = frozenset(
    {"task", "condition", "parallel", "loop", "human"}
)

# 重试策略
RETRY_STRATEGIES = frozenset({"none", "fixed", "exponential", "switch_model"})

# 风险等级
RISK_LEVEL_L0 = 0  # 读操作
RISK_LEVEL_L1 = 1  # 安全写
RISK_LEVEL_L2 = 2  # 风险写
RISK_LEVEL_L3 = 3  # 危险操作

# Prompt 注入拦截模式
BLOCKED_PATTERNS = [
    r"忽略.*(?:指令|规则|系统提示|system prompt|以上)",
    r"ignore.*(?:instructions|rules|system|above)",
    r"你是.*(?:开发者|DAN|开发者模式|新角色)",
    r"you are.*(?:developer|DAN|dev mode|new role)",
    r"\[SYSTEM\]:.*",
    r"<\|im_start\|>system",
    r"\[INST\].*",
    r"<<SYS>>.*",
    r"从现在开始.*(?:你|你的身份)",
    r"forget.*(?:everything|all|previous)",
    r"disregard.*(?:instructions|rules)",
    r"扮演.*(?:角色|身份)",
    r"act as.*(?:developer|admin|root)",
]

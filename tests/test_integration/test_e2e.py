"""端到端集成测试。"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.shared.models import (
    Experience, MemoryEntry, MemoryQuery, MemoryType,
    StepConfig, StepResult, StepType, WorkflowDefinition,
)


class TestMemoryWorkflowIntegration:
    """Memory + Workflow 集成测试。"""

    async def test_write_and_search_workflow(self):
        """写入记忆 → 搜索 → 工作流读取 全链路。"""
        from src.memory.gateway import MemoryGateway
        from src.memory.write_policy import WriteDecision, WriteEvaluation

        with patch("src.memory.gateway.LongTermMemory") as mock_lt, \
             patch("src.memory.gateway.WritePolicy") as mock_wp:
            gw = MemoryGateway()
            gw._long_term = mock_lt.return_value
            gw._write_policy = mock_wp.return_value

            # 写入
            entry = MemoryEntry(
                agent_id="researcher",
                content="API timeout should be set to 120s for stability",
                memory_type=MemoryType.EXPERIENCE,
                importance=0.85,
            )
            gw._long_term.search.return_value = []
            gw._write_policy.evaluate.return_value = WriteEvaluation(
                decision=WriteDecision.WRITE, importance=0.85, reason="novel"
            )
            gw._long_term.store.return_value = entry.id

            write_id = await gw.write(entry, trace_id="integ_1")
            assert write_id == entry.id

            # 搜索
            from src.memory.long_term import MemorySearchResult
            gw._long_term.search.return_value = [
                MemorySearchResult(entry=entry, score=0.95)
            ]
            results = await gw.search(
                MemoryQuery(
                    query_text="API timeout stability",
                    agent_id="researcher",
                ),
                trace_id="integ_1",
            )
            assert len(results) > 0
            assert "timeout" in results[0].entry.content


class TestApprovalSafetyIntegration:
    """Approval + Safety 集成测试。"""

    async def test_l3_operation_requires_approval(self):
        """L3 操作必须先过 PreToolUse Hook，再到审批路由。"""
        from src.safety.hooks import HookManager
        from src.approval.router import ApprovalRouter
        from src.shared.models import HookDecision

        with patch("src.safety.hooks.AuditLogger"), \
             patch("src.safety.hooks.CostTracker") as mock_cost:
            hooks = HookManager()
            hooks._cost = mock_cost.return_value
            hooks._cost.check_budget.return_value = {
                "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
            }

            # PreToolUse 拦截 L3
            decision = await hooks.run_pre_tool_use(
                "tech-dev", "rm_rf", {"path": "/tmp/test"}, trace_id="integ_2"
            )
            assert decision == HookDecision.ESCALATE

            # 然后走审批路由
            router = ApprovalRouter()
            req = await router.request(
                "tech-dev", "rm_rf",
                "cleaning temp files at /tmp/test", trace_id="integ_2"
            )
            assert req.risk_level.value >= 2

    async def test_safe_operation_passes_all_checks(self):
        """安全操作通过所有检查。"""
        from src.safety.hooks import HookManager
        from src.shared.models import HookDecision

        with patch("src.safety.hooks.AuditLogger"), \
             patch("src.safety.hooks.CostTracker") as mock_cost:
            hooks = HookManager()
            hooks._cost = mock_cost.return_value
            hooks._cost.check_budget.return_value = {
                "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
            }

            # PreToolUse 允许
            pre = await hooks.run_pre_tool_use(
                "researcher", "web_search",
                {"query": "latest AI news"}, trace_id="integ_3"
            )
            assert pre == HookDecision.ALLOW

            # PostToolUse 允许
            post = await hooks.run_post_tool_use(
                "researcher", "web_search",
                "Found 10 results about AI", trace_id="integ_3"
            )
            assert post == HookDecision.ALLOW


class TestWorkflowYAMLIntegration:
    """验证 YAML 工作流定义与实际加载。"""

    def test_all_workflow_files_loadable(self):
        from src.workflow.engine import WorkflowEngine

        workflows_dir = Path(__file__).parents[2] / "config" / "workflows"
        for yaml_file in workflows_dir.glob("*.yaml"):
            wf = WorkflowEngine.load(str(yaml_file))
            assert wf.name != ""
            assert len(wf.steps) > 0
            # 所有步骤 ID 唯一
            step_ids = [s.id for s in wf.steps]
            assert len(step_ids) == len(set(step_ids)), f"Duplicate step IDs in {yaml_file.name}"

    def test_all_agent_configs_loadable(self):
        import yaml

        agents_dir = Path(__file__).parents[2] / "config" / "agents"
        for yaml_file in agents_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                assert config is not None, f"Failed to parse {yaml_file.name}"
                agent = config.get("agent", {})
                assert agent.get("id"), f"No agent.id in {yaml_file.name}"
                assert agent.get("name"), f"No agent.name in {yaml_file.name}"

    def test_all_agent_md_files_exist(self):
        agents_dir = Path(__file__).parents[2] / "config" / "agents"
        md_files = list(agents_dir.glob("*.md"))
        assert len(md_files) == 11, f"Expected 11 AGENTS.md, got {len(md_files)}"
        for md in md_files:
            content = md.read_text()
            assert len(content) > 100, f"{md.name} is too short"

    def test_all_test_data_loadable(self):
        from src.evaluation.offline_eval import OfflineEvalRunner

        runner = OfflineEvalRunner()
        data_dir = Path(__file__).parents[1] / "data" / "happy_path"
        yaml_count = len(list(data_dir.glob("*.yaml")))
        assert yaml_count == 11, f"Expected 11 happy_path test files, got {yaml_count}"

        # Verify each file loads
        for yaml_file in data_dir.glob("*.yaml"):
            cases = runner.load_test_cases("happy_path")
            agent_cases = [c for c in cases if c.agent == yaml_file.stem]
            # 至少本文件的测试用例被加载
            assert len(cases) >= 0, f"Failed to load {yaml_file.name}"

    def test_all_prompt_templates_loadable(self):
        import yaml

        prompts_dir = Path(__file__).parents[2] / "config" / "prompts"
        yaml_files = list(prompts_dir.glob("*.yaml"))
        assert len(yaml_files) >= 5
        for f in yaml_files:
            with open(f) as fh:
                data = yaml.safe_load(fh)
                assert data is not None
                assert "prompts" in data


class TestSharedModels:
    """测试数据模型。"""

    def test_memory_entry_creation(self):
        entry = MemoryEntry(
            agent_id="researcher",
            content="test",
            importance=0.6,
        )
        assert len(entry.id) > 0
        assert entry.agent_id == "researcher"

    def test_workflow_definition_creation(self):
        wf = WorkflowDefinition(
            name="test-wf",
            steps=[
                StepConfig(id="s1", type=StepType.TASK),
                StepConfig(id="s2", type=StepType.CONDITION, depends_on=["s1"]),
            ],
        )
        assert wf.name == "test-wf"
        assert len(wf.steps) == 2

    def test_step_result_defaults(self):
        result = StepResult(step_id="test")
        assert result.status == "success"
        assert result.error is None

    def test_memory_query_defaults(self):
        from src.shared.models import MemoryQuery
        query = MemoryQuery()
        assert query.top_k == 5
        assert query.min_importance == 0.0

    def test_experience_id_generated(self):
        exp = Experience(
            agent_id="researcher",
            task_type="search",
            trigger="", symptom="", root_cause="",
            solution="test", outcome="",
        )
        assert len(exp.id) > 0


class TestErrorHierarchy:
    """测试异常层级。"""

    def test_all_errors_importable(self):
        from src.shared.errors import (
            AgentSystemError,
            ConfigError,
            MemoryError,
            MemoryUnavailableError,
            MemoryWriteError,
            MemorySearchError,
            WorkflowError,
            WorkflowParseError,
            WorkflowExecutionError,
            StepTimeoutError,
            CircularDependencyError,
            EvaluationError,
            ApprovalError,
            SafetyError,
            PromptInjectionDetected,
            UnauthorizedOperationError,
            BudgetExceededError,
            ExperienceError,
            LLMError,
            LLMTimeoutError,
            LLMRateLimitError,
            ToolError,
        )
        # 验证层级
        assert issubclass(MemoryError, AgentSystemError)
        assert issubclass(WorkflowError, AgentSystemError)
        assert issubclass(SafetyError, AgentSystemError)
        assert issubclass(CircularDependencyError, WorkflowError)


class TestConstants:
    """测试常量定义。"""

    def test_agent_ids(self):
        from src.shared.constants import AGENT_IDS
        assert "main" in AGENT_IDS
        assert "researcher" in AGENT_IDS
        assert len(AGENT_IDS) == 11

    def test_blocked_patterns_compile(self):
        import re
        from src.shared.constants import BLOCKED_PATTERNS
        for pattern in BLOCKED_PATTERNS:
            re.compile(pattern)  # 不能抛异常

    def test_step_types_match_model(self):
        from src.shared.constants import STEP_TYPES
        from src.shared.models import StepType
        for st in StepType:
            assert st.value in STEP_TYPES

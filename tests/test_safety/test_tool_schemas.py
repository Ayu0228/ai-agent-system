"""Tool schema validation tests — ToolDefinition.validate(), AgentRegistry integration, hooks integration."""

import pytest

from src.shared.models import (
    HookDecision, ToolParamSchema, ToolParamType, ToolDefinition,
    ToolValidationResult, RiskLevel,
)


class TestToolParamSchema:
    """Test ToolParamSchema model."""

    def test_defaults(self):
        p = ToolParamSchema(name="query")
        assert p.name == "query"
        assert p.type == ToolParamType.STRING
        assert p.required is False
        assert p.default is None
        assert p.enum is None

    def test_required_param(self):
        p = ToolParamSchema(name="path", required=True, description="File path")
        assert p.required is True
        assert p.description == "File path"

    def test_enum_constraint(self):
        p = ToolParamSchema(name="method", type=ToolParamType.STRING, enum=["GET", "POST", "PUT"])
        assert p.enum == ["GET", "POST", "PUT"]

    def test_number_type(self):
        p = ToolParamSchema(name="timeout", type=ToolParamType.NUMBER, default=30)
        assert p.type == ToolParamType.NUMBER
        assert p.default == 30


class TestToolDefinition:
    """Test ToolDefinition model."""

    def test_defaults(self):
        t = ToolDefinition(name="search")
        assert t.name == "search"
        assert t.description == ""
        assert t.parameters == []
        assert t.risk_level == RiskLevel.L1
        assert t.requires_approval is False
        assert t.max_retries == 1
        assert t.timeout_s == 30

    def test_with_parameters(self):
        t = ToolDefinition(
            name="write_file",
            description="Write content to file",
            risk_level=RiskLevel.L1,
            parameters=[
                ToolParamSchema(name="path", required=True),
                ToolParamSchema(name="content", required=True),
                ToolParamSchema(name="mode", enum=["w", "a"]),
            ],
        )
        assert len(t.parameters) == 3

    def test_high_risk_tool(self):
        t = ToolDefinition(
            name="delete_file",
            risk_level=RiskLevel.L3,
            requires_approval=True,
        )
        assert t.risk_level == RiskLevel.L3
        assert t.requires_approval is True


class TestToolValidation:
    """Test ToolDefinition.validate()."""

    @pytest.fixture
    def search_tool(self):
        return ToolDefinition(
            name="web_search",
            parameters=[
                ToolParamSchema(name="query", required=True),
                ToolParamSchema(name="limit", type=ToolParamType.NUMBER, default=5),
                ToolParamSchema(name="source", enum=["web", "news", "scholar"]),
            ],
        )

    @pytest.fixture
    def write_tool(self):
        return ToolDefinition(
            name="write_file",
            parameters=[
                ToolParamSchema(name="path", required=True),
                ToolParamSchema(name="content", required=True),
                ToolParamSchema(name="overwrite", type=ToolParamType.BOOLEAN, default=False),
            ],
        )

    def test_valid_params_pass(self, search_tool):
        result = search_tool.validate({"query": "AI agents"})
        assert result.valid is True
        assert len(result.errors) == 0

    def test_valid_with_optional_params(self, search_tool):
        result = search_tool.validate({"query": "AI agents", "limit": 10})
        assert result.valid is True

    def test_missing_required_param_blocked(self, search_tool):
        result = search_tool.validate({})
        assert result.valid is False
        assert len(result.missing_required) == 1
        assert "query" in result.missing_required

    def test_wrong_type_blocked(self, write_tool):
        result = write_tool.validate({"path": "/tmp/f", "content": "data", "overwrite": "yes"})
        assert result.valid is False
        assert len(result.type_mismatches) >= 1
        assert any("overwrite" in e for e in result.type_mismatches)

    def test_unknown_param_warned(self, search_tool):
        result = search_tool.validate({"query": "test", "extra_param": "unexpected"})
        assert result.valid is True  # unknown params just warn, don't block
        assert len(result.warnings) >= 1
        assert "extra_param" in result.unknown_params

    def test_enum_invalid_value_blocked(self, search_tool):
        result = search_tool.validate({"query": "test", "source": "invalid_src"})
        assert result.valid is False
        assert any("invalid_src" in e for e in result.errors)

    def test_enum_valid_value_passes(self, search_tool):
        result = search_tool.validate({"query": "test", "source": "news"})
        assert result.valid is True

    def test_empty_parameters_always_valid(self):
        tool = ToolDefinition(name="no_params")
        result = tool.validate({"anything": "goes"})
        assert result.valid is True
        assert len(result.unknown_params) == 1  # warned but valid

    def test_number_type_rejects_string(self):
        tool = ToolDefinition(name="calc", parameters=[
            ToolParamSchema(name="value", type=ToolParamType.NUMBER, required=True),
        ])
        result = tool.validate({"value": "not_a_number"})
        assert result.valid is False

    def test_number_type_accepts_int(self):
        tool = ToolDefinition(name="calc", parameters=[
            ToolParamSchema(name="value", type=ToolParamType.NUMBER, required=True),
        ])
        result = tool.validate({"value": 42})
        assert result.valid is True

    def test_number_type_accepts_float(self):
        tool = ToolDefinition(name="calc", parameters=[
            ToolParamSchema(name="value", type=ToolParamType.NUMBER, required=True),
        ])
        result = tool.validate({"value": 3.14})
        assert result.valid is True

    def test_boolean_type_rejects_string(self):
        tool = ToolDefinition(name="toggle", parameters=[
            ToolParamSchema(name="flag", type=ToolParamType.BOOLEAN, required=True),
        ])
        result = tool.validate({"flag": "true"})
        assert result.valid is False

    def test_boolean_type_accepts_bool(self):
        tool = ToolDefinition(name="toggle", parameters=[
            ToolParamSchema(name="flag", type=ToolParamType.BOOLEAN, required=True),
        ])
        result = tool.validate({"flag": True})
        assert result.valid is True

    def test_array_type_check(self):
        tool = ToolDefinition(name="batch", parameters=[
            ToolParamSchema(name="items", type=ToolParamType.ARRAY, required=True),
        ])
        assert tool.validate({"items": [1, 2, 3]}).valid is True
        assert tool.validate({"items": "not_a_list"}).valid is False

    def test_object_type_check(self):
        tool = ToolDefinition(name="config", parameters=[
            ToolParamSchema(name="options", type=ToolParamType.OBJECT, required=True),
        ])
        assert tool.validate({"options": {"key": "val"}}).valid is True
        assert tool.validate({"options": "not_a_dict"}).valid is False


class TestToolValidationResult:
    """Test ToolValidationResult model."""

    def test_defaults(self):
        r = ToolValidationResult()
        assert r.valid is False
        assert r.errors == []
        assert r.warnings == []

    def test_valid_result(self):
        r = ToolValidationResult(valid=True)
        assert r.valid is True

    def test_with_errors(self):
        r = ToolValidationResult(
            errors=["Missing: query"],
            missing_required=["query"],
        )
        assert r.valid is False
        assert len(r.missing_required) == 1


class TestRegistryToolIntegration:
    """Test AgentRegistry tool index."""

    @pytest.fixture
    def registry(self):
        from src.orchestration.registry import AgentRegistry, AgentInfo
        return AgentRegistry()

    @pytest.fixture
    def search_tool_def(self):
        return ToolDefinition(
            name="web_search",
            parameters=[
                ToolParamSchema(name="query", required=True),
                ToolParamSchema(name="limit", type=ToolParamType.NUMBER, default=5),
            ],
        )

    def test_registry_indexes_tool_definitions(self, registry, search_tool_def):
        from src.orchestration.registry import AgentInfo

        info = AgentInfo(
            agent_id="researcher",
            tools=[search_tool_def],
        )
        registry.register(info)

        tool = registry.get_tool("researcher", "web_search")
        assert tool is not None
        assert tool.name == "web_search"
        assert len(tool.parameters) == 2

    def test_registry_handles_string_tools(self, registry):
        from src.orchestration.registry import AgentInfo

        info = AgentInfo(
            agent_id="researcher",
            tools=["web_search", "calculator"],
        )
        registry.register(info)

        tool = registry.get_tool("researcher", "web_search")
        assert tool is None  # string tools don't get indexed

    def test_validate_tool_params_registered(self, registry, search_tool_def):
        from src.orchestration.registry import AgentInfo

        info = AgentInfo(agent_id="researcher", tools=[search_tool_def])
        registry.register(info)

        result = registry.validate_tool_params("researcher", "web_search", {"query": "test"})
        assert result.valid is True

    def test_validate_tool_params_missing_required(self, registry, search_tool_def):
        from src.orchestration.registry import AgentInfo

        info = AgentInfo(agent_id="researcher", tools=[search_tool_def])
        registry.register(info)

        result = registry.validate_tool_params("researcher", "web_search", {})
        assert result.valid is False

    def test_validate_unregistered_tool_passes(self, registry):
        result = registry.validate_tool_params("agent", "unknown_tool", {"x": 1})
        assert result.valid is True  # loose mode
        assert len(result.warnings) >= 1


class TestHookManagerSchemaIntegration:
    """Test that HookManager integrates with registry for tool schema validation."""

    @pytest.fixture
    async def hooks_with_registry(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.safety.hooks import HookManager
        from src.orchestration.registry import AgentRegistry, AgentInfo
        from src.shared.models import ToolDefinition, ToolParamSchema

        registry = AgentRegistry()
        tool_def = ToolDefinition(
            name="write_file",
            parameters=[
                ToolParamSchema(name="path", required=True),
                ToolParamSchema(name="content", required=True),
            ],
        )
        info = AgentInfo(agent_id="researcher", tools=[tool_def])
        registry.register(info)

        with patch("src.safety.hooks.AuditLogger") as mock_audit, \
             patch("src.safety.hooks.CostTracker") as mock_cost:
            hooks = HookManager(registry=registry)
            hooks._audit = mock_audit.return_value
            hooks._cost = mock_cost.return_value
            yield hooks

    async def test_schema_validation_blocks_bad_params(self, hooks_with_registry):
        hooks_with_registry._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        # Missing required params "path" and "content"
        result = await hooks_with_registry.run_pre_tool_use(
            "researcher", "write_file", {}, trace_id="t1"
        )
        assert result == HookDecision.BLOCK

    async def test_schema_validation_allows_valid_params(self, hooks_with_registry):
        from src.shared.models import HookDecision
        hooks_with_registry._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        result = await hooks_with_registry.run_pre_tool_use(
            "researcher", "write_file",
            {"path": "/tmp/test", "content": "hello"}, trace_id="t1"
        )
        assert result == HookDecision.ALLOW

    async def test_unregistered_tool_skips_schema_check(self, hooks_with_registry):
        from src.shared.models import HookDecision
        hooks_with_registry._cost.check_budget.return_value = {
            "tokens_used": 100, "tokens_remaining": 49900, "is_exceeded": False,
        }
        result = await hooks_with_registry.run_pre_tool_use(
            "researcher", "unknown_tool", {"anything": 1}, trace_id="t1"
        )
        assert result == HookDecision.ALLOW

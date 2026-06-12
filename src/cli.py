"""CLI 入口：运行评估、工作流、系统管理。"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click

from src.shared.config import get_project_root, get_settings
from src.shared.marker import apply_marker


@click.group()
@click.version_option(version="0.1.0", prog_name="ai-agent-system")
def main():
    """AI Agent 系统 CLI — 11 Agent × 6 核心组件。"""


# ═══════════════════════════════════════════════════════════════
# eval
# ═══════════════════════════════════════════════════════════════


@main.group()
def eval_cmd():
    """离线评估命令组。"""


@eval_cmd.command("run")
@click.option("--agent", "-a", required=True, help="Agent ID")
@click.option("--test-set", "-t", default="happy_path",
              type=click.Choice(["happy_path", "edge_cases", "adversarial"]),
              help="测试集类型")
@click.option("--no-baseline", is_flag=True, help="不对比基线")
def eval_run(agent: str, test_set: str, no_baseline: bool):
    """运行离线评估。"""

    async def _run():
        from src.evaluation.offline_eval import OfflineEvalRunner

        runner = OfflineEvalRunner()
        report = await runner.run(
            agent, test_set, compare_baseline=not no_baseline,
        )

        click.echo(f"\n{'='*60}")
        click.echo(f"  Agent: {report.agent_id}")
        click.echo(f"  Test Set: {report.test_set}")
        click.echo(f"  Total: {report.total_cases} | Passed: {report.passed} | Failed: {report.failed}")
        if report.total_cases > 0:
            rate = report.passed / report.total_cases
            color = "green" if rate >= 0.9 else "yellow" if rate >= 0.7 else "red"
            click.echo(f"  Pass Rate: {click.style(f'{rate:.1%}', fg=color)}")
        if report.degraded_from_baseline:
            click.echo(f"  {click.style('DEGRADED from baseline!', fg='red', bold=True)}")
            for detail in report.degradation_details:
                click.echo(f"    - {detail}")
        click.echo(f"{'='*60}\n")

    asyncio.run(_run())


@eval_cmd.command("regression")
@click.option("--agent", "-a", default="", help="Agent ID（留空=全部）")
@click.option("--category", "-c", default="happy_path",
              type=click.Choice(["happy_path", "edge_cases", "adversarial"]),
              help="测试集类型")
@click.option("--use-relay", is_flag=True, help="使用 OpenClaw 真实 Agent")
@click.option("--save-baseline", is_flag=True, help="保存为基线")
def eval_regression(agent: str, category: str, use_relay: bool, save_baseline: bool):
    """运行回归测试（Golden Dataset）。"""

    async def _run():
        from src.evaluation.regression import RegressionRunner

        runner = RegressionRunner(use_relay=use_relay)
        if use_relay:
            from src.relay.client import get_relay_client
            get_relay_client(use_relay=True)

        if agent:
            click.echo(f"Running regression for: {agent} ({category})")
            report = await runner.run(agent, category=category)
            _print_regression_report(report)
            if save_baseline:
                RegressionRunner.save_baseline(agent, report)
                click.echo(f"Baseline saved for {agent}")
        else:
            click.echo(f"Running regression for all agents ({category})...")
            reports = await runner.run_all(category=category)
            total_pass = 0
            total_cases = 0
            for aid, report in reports.items():
                _print_regression_report(report, compact=True)
                total_pass += report.passed
                total_cases += report.total
                if save_baseline:
                    RegressionRunner.save_baseline(aid, report)
            click.echo(f"\n{'='*50}")
            click.echo(f"  Overall: {total_pass}/{total_cases} passed ({total_pass/max(total_cases,1):.1%})")
            if save_baseline:
                click.echo(f"  Baselines saved for all agents")

    asyncio.run(_run())


def _print_regression_report(report, compact: bool = False):
    from src.evaluation.regression import RegressionReport
    if report.total == 0:
        click.echo(f"  {report.dataset_name}: No cases found")
        return

    color = "green" if report.pass_rate >= 0.9 else "yellow" if report.pass_rate >= 0.7 else "red"
    prefix = f"  {report.dataset_name:20s}" if compact else f"  Agent: {report.dataset_name}"
    click.echo(f"{prefix}: {report.passed}/{report.total} passed "
               f"({click.style(f'{report.pass_rate:.1%}', fg=color)}) "
               f"score={report.avg_score:.2f} "
               f"[hard_fail={report.hard_failures}, soft_fail={report.soft_failures}]")

    if not compact and report.degraded_from_baseline:
        click.echo(f"    {click.style('DEGRADED from baseline!', fg='red', bold=True)}")
        for detail in report.degradation_details:
            click.echo(f"      - {detail}")


@eval_cmd.command("golden")
@click.option("--agent", "-a", default="", help="Agent ID")
@click.option("--category", "-c", default="happy_path",
              type=click.Choice(["happy_path", "edge_cases", "adversarial"]),
              help="数据集类型")
def eval_golden(agent: str, category: str):
    """查看 Golden Dataset 统计"""
    from src.evaluation.golden import GoldenStore

    store = GoldenStore()
    if agent:
        ds = store.load(agent, category)
        click.echo(f"Dataset: {ds.name}/{category} — {ds.total} cases")
    else:
        all_cases = store.load_all()
        by_agent: dict[str, int] = {}
        for c in all_cases:
            by_agent[c.agent_id] = by_agent.get(c.agent_id, 0) + 1
        click.echo(f"Total golden cases: {len(all_cases)}")
        for aid, count in sorted(by_agent.items()):
            click.echo(f"  {aid}: {count}")
@click.argument("agent_id")
def eval_baseline(agent_id: str):
    """查看基线数据。"""
    from src.evaluation.offline_eval import BaselineStore

    store = BaselineStore()
    data = store.load(agent_id)
    if data:
        click.echo(f"Baseline for {agent_id}:")
        for k, v in data.items():
            click.echo(f"  {k}: {v}")
    else:
        click.echo(f"No baseline found for {agent_id}")


# ═══════════════════════════════════════════════════════════════
# workflow
# ═══════════════════════════════════════════════════════════════


@main.group()
def workflow():
    """工作流命令组。"""


@workflow.command("run")
@click.argument("workflow_name")
@click.option("--param", "-p", multiple=True, help="参数 key=value")
@click.option("--verbose", "-v", is_flag=True, help="详细输出")
@click.option("--use-relay", is_flag=True, help="使用 OpenClaw 真实 Agent（默认 LLM 模拟）")
def workflow_run(workflow_name: str, param: tuple[str, ...], verbose: bool, use_relay: bool):
    """运行工作流。"""

    async def _run():
        from src.workflow.engine import WorkflowEngine
        from src.relay.client import get_relay_client
        from src.workflow.steps import StepExecutor

        # 解析参数
        params: dict = {}
        for p in param:
            if "=" in p:
                k, v = p.split("=", 1)
                params[k] = v

        engine = WorkflowEngine()

        # 查找工作流文件
        wf_path = get_project_root() / "config" / "workflows" / f"{workflow_name}.yaml"
        if not wf_path.exists():
            click.echo(f"Workflow '{workflow_name}' not found at {wf_path}", err=True)
            sys.exit(1)

        # 启用 relay 模式（必须在 engine.run 之前调用，seed 单例）
        if use_relay:
            from src.relay.client import get_relay_client, RelayClient
            get_relay_client(use_relay=True)  # seed singleton for StepExecutor
            available = RelayClient.is_openclaw_available()
            status = "已连接" if available else "未找到 CLI，将 fallback 到 LLM"
            click.echo(f"Relay mode: OpenClaw Agent ({status})")

        click.echo(f"Loading workflow: {workflow_name}")
        wf = engine.load(str(wf_path))

        click.echo(f"Steps: {len(wf.steps)}")
        for s in wf.steps:
            deps = f" (depends: {', '.join(s.depends_on)})" if s.depends_on else ""
            click.echo(f"  [{s.type.value}] {s.id}{deps}")

        click.echo(f"\nRunning...")
        result = await engine.run(wf, params)

        click.echo(f"\n{'='*60}")
        click.echo(f"  Status: {result.status}")
        click.echo(f"  Steps executed: {len(result.steps)}")
        click.echo(f"  Total tokens: {result.total_tokens}")
        click.echo(f"  Duration: {result.total_duration_ms}ms")
        click.echo(f"{'='*60}")
        click.echo(apply_marker(""))

        if verbose:
            for s in result.steps:
                status_color = "green" if s.status == "success" else "red"
                click.echo(f"  [{click.style(s.status, fg=status_color)}] {s.step_id}")
                if s.error:
                    click.echo(f"    Error: {s.error}")
                if s.tokens_used:
                    click.echo(f"    Tokens: {s.tokens_used}")

    asyncio.run(_run())


@workflow.command("list")
def workflow_list():
    """列出所有可用工作流。"""
    workflows_dir = get_project_root() / "config" / "workflows"
    if not workflows_dir.exists():
        click.echo("No workflows directory found")
        return

    for f in sorted(workflows_dir.glob("*.yaml")):
        import yaml
        with open(f) as fh:
            data = yaml.safe_load(fh)
            wf = data.get("workflow", data)
            click.echo(f"  {wf['name']:30s} — {wf.get('description', '')}")
            click.echo(f"    Steps: {len(wf.get('steps', []))} | Trigger: {wf.get('trigger', {}).get('type', 'manual')}")


# ═══════════════════════════════════════════════════════════════
# agent
# ═══════════════════════════════════════════════════════════════


@main.group()
def agent():
    """Agent 管理命令组。"""


@agent.command("list")
def agent_list():
    """列出所有 Agent 配置。"""
    import yaml

    agents_dir = get_project_root() / "config" / "agents"
    if not agents_dir.exists():
        click.echo("No agents configured")
        return

    click.echo(f"{'ID':25s} {'Name':15s} {'Model':20s}")
    click.echo("-" * 62)
    for f in sorted(agents_dir.glob("*.yaml")):
        with open(f) as fh:
            data = yaml.safe_load(fh)
            agent_data = data.get("agent", {})
            click.echo(
                f"{agent_data.get('id', ''):25s} "
                f"{agent_data.get('name', ''):15s} "
                f"{agent_data.get('model', ''):20s}"
            )


# ═══════════════════════════════════════════════════════════════
# memory
# ═══════════════════════════════════════════════════════════════


@main.group()
def memory():
    """记忆系统命令组。"""


@memory.command("stats")
def memory_stats():
    """显示记忆系统统计。"""
    from src.memory.gateway import MemoryGateway

    gw = MemoryGateway()
    try:
        # 快速统计
        results = gw._long_term.search(
            __import__("src.shared.models", fromlist=["MemoryQuery"]).MemoryQuery(top_k=1)
        )
        click.echo(f"Memory system: OK")
        click.echo(f"  ChromaDB: {get_settings().chromadb_path}")
        click.echo(f"  SQLite: {get_settings().sqlite_path}")
    except Exception as e:
        click.echo(f"Memory system error: {e}", err=True)
    finally:
        gw.close()


@memory.command("decay")
@click.option("--threshold", "-t", default=0.3, type=float, help="遗忘阈值")
def memory_decay(threshold: float):
    """运行记忆遗忘策略。"""
    from src.memory.gateway import MemoryGateway

    gw = MemoryGateway()
    try:
        removed = gw._long_term.decay(threshold=threshold)
        click.echo(f"Memory decay complete: {removed} entries removed (threshold={threshold})")
    finally:
        gw.close()


# ═══════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════


@main.group()
def relay():
    """Agent 中继命令组（OpenClaw CLI 桥接）。"""


@relay.command("test")
@click.option("--agent", "-a", default="researcher", help="Agent ID（默认 researcher）")
@click.option("--message", "-m", default="用一句话介绍你自己", help="测试消息")
@click.option("--timeout", "-t", default=60, type=int, help="超时秒数")
def relay_test(agent: str, message: str, timeout: int):
    """测试 OpenClaw Agent 连通性。"""
    import asyncio

    async def _test():
        from src.relay.client import RelayClient

        client = RelayClient(use_relay=True)
        click.echo(f"Testing OpenClaw Agent: {agent}")
        click.echo(f"Message: {message}")
        click.echo(f"Timeout: {timeout}s\n")

        try:
            result = await asyncio.wait_for(
                client._openclaw_send(agent, message, "cli-test"),
                timeout=timeout,
            )
            click.echo(f"{'='*60}")
            click.echo(f"Response ({len(result)} chars):")
            click.echo(f"{'='*60}")
            click.echo(apply_marker(result))
            click.echo(f"{'='*60}")
            click.echo(click.style("SUCCESS", fg="green", bold=True))
        except asyncio.TimeoutError:
            click.echo(click.style(f"Timeout after {timeout}s", fg="red"))
        except Exception as e:
            click.echo(click.style(f"Error: {e}", fg="red"))

    asyncio.run(_test())


@relay.command("send")
@click.option("--agent", "-a", required=True, help="Agent ID（必填）")
@click.option("--message", "-m", required=True, help="任务内容（必填）")
@click.option("--timeout", "-t", default=300, type=int, help="超时秒数")
def relay_send(agent: str, message: str, timeout: int):
    """派发任务给指定 Agent，返回执行结果。"""
    import asyncio

    async def _send():
        from src.relay.client import RelayClient

        client = RelayClient(use_relay=True)
        try:
            result = await asyncio.wait_for(
                client.call_agent(agent, message, trace_id=f"cli-send-{agent}"),
                timeout=timeout,
            )
            click.echo(result)
        except asyncio.TimeoutError:
            click.echo(f"Error: timeout after {timeout}s", err=True)
        except Exception as e:
            click.echo(f"Error: {e}", err=True)

    asyncio.run(_send())


@relay.command("list")
def relay_list():
    """列出所有可用的 OpenClaw Agent。"""
    from src.relay.client import RelayClient

    if RelayClient.is_openclaw_available():
        click.echo("OpenClaw Agent 可用（openclaw CLI 已安装）")
    else:
        click.echo("OpenClaw Agent 不可用（openclaw CLI 未找到）")
        click.echo("安装 OpenClaw: npm install -g openclaw")

    click.echo(f"\n已注册的 Agent ID:")
    from src.shared.constants import AGENT_IDS
    for aid in sorted(AGENT_IDS):
        click.echo(f"  - {aid}")


# ═══════════════════════════════════════════════════════════════
# observability
# ═══════════════════════════════════════════════════════════════


@main.group()
def observability():
    """可观测性命令组（追踪/指标/审计日志）。"""


@observability.command("traces")
@click.option("--limit", "-n", default=10, type=int, help="显示最近 N 条 Trace")
def obs_traces(limit: int):
    """查看最近的 Trace 记录。"""
    from src.observability.tracer import get_tracer

    tracer = get_tracer()
    traces = tracer.get_recent_traces(limit)

    if not traces:
        click.echo("No traces recorded yet.")
        return

    for t in traces:
        click.echo(f"\n{'='*60}")
        click.echo(f"Trace: {t.trace_id[:16]}...")
        click.echo(f"Session: {t.session_id}")
        click.echo(f"Spans: {len(t.spans)} | Tokens: {t.total_tokens} | "
                   f"Cost: ${t.total_cost:.4f} | Duration: {t.duration_ms:.0f}ms")
        for s in t.spans[:5]:  # 只显示前 5 个 span
            status_icon = click.style("✓", fg="green") if s.status == "ok" else click.style("✗", fg="red")
            click.echo(f"  {status_icon} [{s.kind.value}] {s.name} "
                       f"({s.duration_ms:.0f}ms, {s.total_tokens}toks)")


@observability.command("metrics")
def obs_metrics():
    """查看实时指标快照。"""
    from src.observability.metrics import get_metrics

    metrics = get_metrics()
    snap = metrics.snapshot()

    click.echo(f"\n{'='*50}")
    click.echo(f"  Token Usage")
    click.echo(f"  Input: {snap.total_input_tokens:,} | Output: {snap.total_output_tokens:,} | Total: {snap.total_tokens:,}")
    click.echo(f"\n  Calls")
    click.echo(f"  LLM: {snap.total_llm_calls} | Tool: {snap.total_tool_calls} | Errors: {snap.total_errors}")
    click.echo(f"\n  Latency (LLM)")
    click.echo(f"  P50: {snap.llm_latency_p50:.0f}ms | P95: {snap.llm_latency_p95:.0f}ms | P99: {snap.llm_latency_p99:.0f}ms")
    click.echo(f"\n  Cost")
    click.echo(f"  Total: ${snap.total_cost_usd:.4f}")
    click.echo(f"\n  By Agent:")
    for aid, am in snap.by_agent.items():
        click.echo(f"    {aid}: {am['llm_calls']} calls, {am['input_tokens'] + am['output_tokens']:,} tokens, ${am['total_cost']:.4f}")
    click.echo(f"{'='*50}")

    # SLO
    slo = metrics.get_slo_status()
    click.echo(f"\n  SLO Status:")
    p95_icon = click.style("✓", fg="green") if slo["llm_latency_slo"] == "ok" else click.style("⚠", fg="yellow")
    err_icon = click.style("✓", fg="green") if slo["error_rate_slo"] == "ok" else click.style("✗", fg="red")
    click.echo(f"  {p95_icon} LLM P95: {slo['llm_latency_p95_ms']:.0f}ms ({slo['llm_latency_slo']})")
    click.echo(f"  {err_icon} Error Rate: {slo['error_rate']:.1%} ({slo['error_rate_slo']})")


@observability.command("audit")
@click.option("--limit", "-n", default=20, type=int, help="显示最近 N 条审计记录")
@click.option("--type", "-t", "event_type", default="", help="按事件类型过滤")
@click.option("--agent", "-a", default="", help="按 Agent 过滤")
def obs_audit(limit: int, event_type: str, agent: str):
    """查看审计日志。"""
    from src.observability.audit import get_audit_logger

    audit = get_audit_logger()
    records = audit.search_logs(event_type=event_type, agent_id=agent, limit=limit)

    for r in records:
        ts = r.get("timestamp", "")[:19]
        rtype = r.get("type", "unknown")
        trace = r.get("trace_id", "")[:12]
        click.echo(f"[{ts}] {rtype:20s} trace={trace}...")
        if "span_id" in r:
            click.echo(f"  kind={r.get('kind','')} name={r.get('name','')} "
                       f"tokens={r.get('total_tokens',0)} cost=${r.get('cost_usd',0):.4f}")


# ═══════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════


@main.command("config")
def show_config():
    """显示当前配置。"""
    settings = get_settings()
    click.echo(f"LLM Model: {settings.llm_model}")
    click.echo(f"LLM Fallback: {settings.llm_fallback_model}")
    click.echo(f"LLM Base URL: {settings.llm_base_url}")
    click.echo(f"ChromaDB: {settings.chromadb_path}")
    click.echo(f"SQLite: {settings.sqlite_path}")
    click.echo(f"Daily Token Budget: {settings.daily_token_budget_per_agent}")
    click.echo(f"Approval Timeout: {settings.approval_timeout}s")


# ═══════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()

"""Memory CLI — MemoryGateway 的 JSON 命令行入口。

用法：
  python -m src.memory_cli search "用户的偏好" --agent main
  python -m src.memory_cli write '{"content":"...","memory_type":"fact"}' --agent main
  python -m src.memory_cli stats
  python -m src.memory_cli maintenance

所有命令输出 JSON 到 stdout，日志到 stderr。
"""

from __future__ import annotations

import argparse
import json as _json
import sys
from typing import Any

from src.memory.gateway import get_gateway
from src.shared.models import (
    ExperienceCard,
    KnowledgeTriple,
    MemoryEntry,
    MemoryQuery,
    MemoryType,
)


def _out(obj: Any) -> None:
    """输出 JSON 到 stdout。"""
    print(_json.dumps(obj, ensure_ascii=False, default=str))
    sys.stdout.flush()


def _die(msg: str, code: int = 1) -> None:
    """输出错误到 stdout（JSON 格式），然后退出。"""
    _out({"error": True, "message": msg})
    sys.exit(code)


# ═══════════════════════════════════════════════════════════════
# 子命令处理器
# ═══════════════════════════════════════════════════════════════


def cmd_write(args: argparse.Namespace) -> None:
    """写入一条记忆到长期记忆。"""
    gw = get_gateway()
    data = _json.loads(args.data) if args.data.startswith("{") else {"content": args.data}

    entry = MemoryEntry(
        agent_id=args.agent,
        content=data.get("content", ""),
        memory_type=MemoryType(data.get("memory_type", "fact")),
        tags=data.get("tags", []),
        importance=data.get("importance", 0.5),
        source_trace_id=data.get("trace_id"),
    )
    entry_id = gw._long_term.store(entry)
    _out({"ok": True, "id": entry_id, "agent": args.agent})


def cmd_search(args: argparse.Namespace) -> None:
    """语义搜索长期记忆。"""
    gw = get_gateway()
    results = gw._long_term.search(
        MemoryQuery(
            query_text=args.query,
            agent_id=args.agent or None,
            top_k=args.top_k,
            min_importance=args.min_importance,
        )
    )
    _out({
        "ok": True,
        "query": args.query,
        "count": len(results),
        "results": [
            {
                "id": r.entry.id,
                "content": r.entry.content,
                "type": r.entry.memory_type.value,
                "importance": r.entry.importance,
                "score": r.score,
                "agent": r.entry.agent_id,
                "tags": r.entry.tags,
                "created_at": r.entry.created_at,
            }
            for r in results
        ],
    })


def cmd_recall(args: argparse.Namespace) -> None:
    """按 ID 读取记忆（agent 级权限检查）。"""
    gw = get_gateway()
    entry = gw.read(args.id, agent_id=args.agent)
    if entry is None:
        _out({"ok": False, "found": False, "id": args.id})
    else:
        _out({
            "ok": True,
            "found": True,
            "id": entry.id,
            "content": entry.content,
            "type": entry.memory_type.value,
            "importance": entry.importance,
            "agent": entry.agent_id,
            "tags": entry.tags,
        })


def cmd_user_search(args: argparse.Namespace) -> None:
    """查询用户记忆。"""
    gw = get_gateway()
    from src.shared.models import UserMemoryQuery

    facts = gw.search_user_facts(
        UserMemoryQuery(
            user_id=args.user,
            top_k=args.top_k,
            min_confidence=args.min_confidence,
            active_only=not args.all,
        )
    )
    _out({
        "ok": True,
        "user": args.user,
        "count": len(facts),
        "facts": [
            {
                "id": f.id,
                "entity": f.entity,
                "predicate": f.predicate,
                "object": f.object,
                "confidence": f.confidence,
                "source": f.source_agent,
                "valid_from": f.valid_from,
                "valid_to": f.valid_to,
            }
            for f in facts
        ],
    })


def cmd_user_write(args: argparse.Namespace) -> None:
    """写入用户事实（main agent 专用）。"""
    gw = get_gateway()
    from src.shared.models import UserFact

    fact = UserFact(
        user_id=args.user,
        entity=args.entity,
        predicate=args.predicate,
        object=args.object,
        confidence=args.confidence,
        source_agent=args.agent or "main",
    )
    result = gw.write_user_fact(fact)
    if result is None:
        _out({"ok": False, "reason": "rate_limited_or_rejected"})
    else:
        _out({"ok": True, "id": result.id, "decision": result.write_decision})


def cmd_user_proposal(args: argparse.Namespace) -> None:
    """提交用户记忆提案（非 main agent 写入）。"""
    gw = get_gateway()
    from src.shared.models import MemoryProposal

    prop = MemoryProposal(
        user_id=args.user,
        entity=args.entity,
        predicate=args.predicate,
        object=args.object,
        confidence=args.confidence,
        source_agent=args.agent or "unknown",
    )
    prop_id = gw.submit_user_proposal(prop)
    _out({"ok": True, "proposal_id": prop_id, "status": "pending_review"})


def cmd_task_create(args: argparse.Namespace) -> None:
    """创建任务记录。"""
    gw = get_gateway()
    task = gw.create_task(owner_agent_id=args.agent)
    _out({
        "ok": True,
        "task_id": task.task_id,
        "status": task.status.value,
        "agent": task.owner_agent_id,
    })


def cmd_task_list(args: argparse.Namespace) -> None:
    """列出活跃任务。"""
    gw = get_gateway()
    tasks = gw.get_active_tasks(args.agent or None)
    _out({
        "ok": True,
        "count": len(tasks),
        "tasks": [
            {
                "task_id": t.task_id,
                "agent": t.owner_agent_id,
                "status": t.status.value if hasattr(t.status, "value") else str(t.status),
                "created_at": t.created_at,
                "updated_at": t.updated_at,
            }
            for t in tasks
        ],
    })


def cmd_task_update(args: argparse.Namespace) -> None:
    """更新任务状态。"""
    gw = get_gateway()
    task = gw.update_task_status(args.task_id, args.status)
    if task is None:
        _out({"ok": False, "reason": "not_found_or_race"})
    else:
        _out({
            "ok": True,
            "task_id": task.task_id,
            "status": task.status.value if hasattr(task.status, "value") else str(task.status),
        })


def cmd_experience_search(args: argparse.Namespace) -> None:
    """检索经验记忆。"""
    gw = get_gateway()
    from src.shared.models import ExperienceQuery

    cards = gw.search_experience(
        ExperienceQuery(query_text=args.query, agent_id=args.agent or None, top_k=args.top_k)
    )
    _out({
        "ok": True,
        "query": args.query,
        "count": len(cards),
        "experiences": [
            {
                "id": c.experience_id,
                "scenario": c.scenario,
                "approach": c.approach,
                "lesson": c.lesson,
                "weight": c.weight,
                "success_rate": c.success_rate,
                "usage_count": c.usage_count,
                "shareable": c.shareable,
            }
            for c in cards
        ],
    })


def cmd_knowledge_search(args: argparse.Namespace) -> None:
    """搜索知识图谱。"""
    gw = get_gateway()
    from src.shared.models import KnowledgeQuery

    results = gw.search_knowledge(KnowledgeQuery(query_text=args.query, top_k=args.top_k))
    _out({
        "ok": True,
        "query": args.query,
        "count": len(results),
        "results": [
            {
                "id": r.entry.id,
                "content": r.entry.content,
                "type": r.entry.memory_type.value,
                "score": r.score,
            }
            for r in results
        ],
    })


def cmd_knowledge_triples(args: argparse.Namespace) -> None:
    """查询知识三元组。"""
    gw = get_gateway()
    kwargs: dict[str, Any] = {}
    if args.subject:
        kwargs["subject"] = args.subject
    if args.predicate:
        kwargs["predicate"] = args.predicate
    if args.object:
        kwargs["object"] = args.object
    triples = gw.query_knowledge_triples(**kwargs)
    _out({
        "ok": True,
        "count": len(triples),
        "triples": [
            {"id": t.id, "subject": t.subject, "predicate": t.predicate,
             "object": t.object, "source": t.source, "confidence": t.confidence_weight}
            for t in triples
        ],
    })


def cmd_stats(args: argparse.Namespace) -> None:
    """获取记忆系统统计摘要。"""
    gw = get_gateway()
    summary = gw.get_stats_summary()
    _out({"ok": True, **summary})


def cmd_maintenance(args: argparse.Namespace) -> None:
    """运行记忆维护周期（需要事件循环）。"""
    import asyncio
    from src.memory.lifecycle import MemoryLifecycle

    gw = get_gateway()
    lifecycle = MemoryLifecycle(gw)
    report = asyncio.run(lifecycle.run_maintenance())
    _out({"ok": True, **report})


def cmd_health(args: argparse.Namespace) -> None:
    """健康检查。"""
    from src.memory.lifecycle import MemoryLifecycle

    gw = get_gateway()
    lifecycle = MemoryLifecycle(gw)
    result = lifecycle.health_check()
    _out({"ok": True, **result})


def cmd_route(args: argparse.Namespace) -> None:
    """查询路由分析（告诉你应该查哪些记忆层）。"""
    gw = get_gateway()
    plan = gw.route_query(args.query, agent_id=args.agent or "")
    _out({"ok": True, **plan})


def cmd_context_stats(args: argparse.Namespace) -> None:
    """上下文记忆统计。"""
    gw = get_gateway()
    _out({"ok": True, **gw.context_stats})


# ═══════════════════════════════════════════════════════════════
# CLI 主入口
# ═══════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory-cli",
        description="MemoryGateway JSON CLI — 五层记忆系统命令行接口",
    )
    p.add_argument("--agent", "-a", default="main", help="调用方 agent ID (默认 main)")
    subs = p.add_subparsers(dest="command", required=False)

    # search
    sp = subs.add_parser("search", help="语义搜索长期记忆", aliases=["find"])
    sp.add_argument("query", help="搜索查询文本")
    sp.add_argument("--top-k", "-k", type=int, default=5, help="返回条数 (默认 5)")
    sp.add_argument("--min-importance", "-i", type=float, default=0.0)
    sp.set_defaults(func=cmd_search)

    # write
    sp = subs.add_parser("write", help="写入长期记忆", aliases=["store"])
    sp.add_argument("data", help="JSON 格式 {'content':'...'} 或纯文本")
    sp.set_defaults(func=cmd_write)

    # recall
    sp = subs.add_parser("recall", help="按 ID 读取记忆", aliases=["get", "read"])
    sp.add_argument("id", help="记忆 ID")
    sp.set_defaults(func=cmd_recall)

    # user-search
    sp = subs.add_parser("user-search", help="查询用户记忆", aliases=["us"])
    sp.add_argument("--user", "-u", required=True, help="用户 ID")
    sp.add_argument("--top-k", "-k", type=int, default=10)
    sp.add_argument("--min-confidence", "-c", type=float, default=0.0)
    sp.add_argument("--all", action="store_true", help="包含已过期事实")
    sp.set_defaults(func=cmd_user_search)

    # user-write
    sp = subs.add_parser("user-write", help="写入用户事实 (main agent)", aliases=["uw"])
    sp.add_argument("--user", "-u", required=True, help="用户 ID")
    sp.add_argument("--entity", "-e", required=True, help="主语")
    sp.add_argument("--predicate", "-p", required=True, help="谓语")
    sp.add_argument("--object", "-o", required=True, dest="object_", help="宾语")
    sp.add_argument("--confidence", "-c", type=float, default=0.7)
    sp.set_defaults(func=lambda args: cmd_user_write(args))

    # user-proposal
    sp = subs.add_parser("user-proposal", help="提交用户记忆提案 (其他 agent)", aliases=["up"])
    sp.add_argument("--user", "-u", required=True)
    sp.add_argument("--entity", "-e", required=True)
    sp.add_argument("--predicate", "-p", required=True)
    sp.add_argument("--object", "-o", required=True, dest="object_")
    sp.add_argument("--confidence", "-c", type=float, default=0.5)
    sp.set_defaults(func=lambda args: cmd_user_proposal(args))

    # task-create
    sp = subs.add_parser("task-create", help="创建任务", aliases=["tc"])
    sp.set_defaults(func=cmd_task_create)

    # task-list
    sp = subs.add_parser("task-list", help="列出活跃任务", aliases=["tl"])
    sp.set_defaults(func=cmd_task_list)

    # task-update
    sp = subs.add_parser("task-update", help="更新任务状态", aliases=["tu"])
    sp.add_argument("task_id")
    sp.add_argument("status", help="新状态: planning/executing_tools/observing_results/done/failed")
    sp.set_defaults(func=cmd_task_update)

    # experience-search
    sp = subs.add_parser("experience-search", help="搜索经验记忆", aliases=["es", "exp"])
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--top-k", "-k", type=int, default=3)
    sp.set_defaults(func=cmd_experience_search)

    # knowledge-search
    sp = subs.add_parser("knowledge-search", help="搜索知识", aliases=["ks"])
    sp.add_argument("query")
    sp.add_argument("--top-k", "-k", type=int, default=5)
    sp.set_defaults(func=cmd_knowledge_search)

    # knowledge-triples
    sp = subs.add_parser("knowledge-triples", help="查询知识三元组", aliases=["kt"])
    sp.add_argument("--subject", "-s")
    sp.add_argument("--predicate", "-p")
    sp.add_argument("--object", "-o", dest="object_")
    sp.set_defaults(func=cmd_knowledge_triples)

    # stats
    sp = subs.add_parser("stats", help="记忆系统统计摘要", aliases=["status"])
    sp.set_defaults(func=cmd_stats)

    # maintenance
    sp = subs.add_parser("maintenance", help="运行记忆维护周期", aliases=["maint"])
    sp.set_defaults(func=cmd_maintenance)

    # health
    sp = subs.add_parser("health", help="健康检查")
    sp.set_defaults(func=cmd_health)

    # route
    sp = subs.add_parser("route", help="查询路由分析")
    sp.add_argument("query")
    sp.set_defaults(func=cmd_route)

    # context-stats
    sp = subs.add_parser("context-stats", help="上下文记忆统计", aliases=["cs"])
    sp.set_defaults(func=cmd_context_stats)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    # 修复 user-write/user-proposal 中的 object_ → object
    if hasattr(args, "object_"):
        args.object = args.object_
    try:
        args.func(args)
    except Exception as e:
        _die(str(e))


if __name__ == "__main__":
    main()

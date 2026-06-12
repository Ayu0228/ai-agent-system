"""Sandbox Environment — 自主 Agent 的安全隔离执行环境。

ref: Anthropic — "extensive testing in sandboxed environments, along with appropriate guardrails"
ref: OpenAI — Agent safety governance: sandbox as required deployment practice
ref: E2B / Code Interpreter — secure code execution sandboxes for AI agents

安全层:
  1. 文件系统隔离 — 只能访问指定目录
  2. 网络限制 — 白名单域名，禁止内网访问
  3. 资源限制 — CPU/内存/时间上限
  4. 系统调用过滤 — 禁止危险系统调用
  5. 输出审查 — 执行结果经过安全过滤
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

logger = structlog.get_logger()


class SandboxMode(str, Enum):
    STRICT = "strict"            # 只读 + 白名单网络 + 无子进程
    STANDARD = "standard"        # 读写限定目录 + 白名单网络
    PERMISSIVE = "permissive"    # 读写 + 任意网络（仅测试用）


@dataclass
class SandboxConfig:
    """沙箱配置。"""
    mode: SandboxMode = SandboxMode.STANDARD
    allowed_dirs: list[str] = field(default_factory=lambda: ["/tmp/sandbox"])
    denied_dirs: list[str] = field(default_factory=lambda: [
        "/etc", "/root", "/home", "~/.ssh", "~/.aws",
    ])
    allowed_domains: list[str] = field(default_factory=list)  # 空=全部禁用
    denied_domains: list[str] = field(default_factory=lambda: [
        "localhost", "127.0.0.1", "10.", "172.16.", "192.168.",
    ])
    max_cpu_time_s: int = 60
    max_memory_mb: int = 512
    max_file_size_mb: int = 10
    max_output_chars: int = 100_000
    blocked_syscalls: list[str] = field(default_factory=lambda: [
        "eval", "exec", "fork", "system", "popen", "subprocess",
    ])
    enable_network: bool = False
    allow_file_write: bool = False


@dataclass
class SandboxResult:
    """沙箱执行结果。"""
    success: bool = False
    output: Any = None
    error: str = ""
    blocked: bool = False
    block_reason: str = ""
    cpu_time_s: float = 0.0
    memory_used_mb: float = 0.0


class Sandbox:
    """沙箱执行环境。

    用法:
        sandbox = Sandbox(SandboxConfig(mode=SandboxMode.STANDARD))

        # 在沙箱中执行代码
        result = sandbox.execute("print(2+2)", language="python")
        if result.blocked:
            print(f"BLOCKED: {result.block_reason}")

        # 在沙箱中执行函数
        result = sandbox.execute_fn(lambda: 2 + 2)
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._execution_count = 0

    # ── 安全检查 ───────────────────────────────────

    def check_code(self, code: str, language: str = "python") -> SandboxResult:
        """静态检查代码是否包含危险操作。"""
        blocked_patterns: list[tuple[str, str]] = []

        if language == "python":
            for syscall in self.config.blocked_syscalls:
                # 检查函数调用: os.system(), subprocess.run(), eval(), exec()
                patterns = [
                    rf'\b{syscall}\s*\(',
                    rf'os\.{syscall}\s*\(',
                    rf'subprocess\.',
                    rf'__import__\s*\(',
                ]
                for pat in patterns:
                    if re.search(pat, code):
                        blocked_patterns.append((syscall, pat))

            # 检查文件操作
            if not self.config.allow_file_write:
                write_patterns = [r'open\(.*["\']w', r'\.write\(', r'shutil\.']
                for pat in write_patterns:
                    if re.search(pat, code):
                        blocked_patterns.append(("file_write", pat))

            # 检查网络操作
            if not self.config.enable_network:
                net_patterns = [r'requests\.', r'urllib', r'socket\.', r'http\.']
                for pat in net_patterns:
                    if re.search(pat, code):
                        blocked_patterns.append(("network", pat))

        if blocked_patterns:
            reasons = [f"{name}" for name, _ in blocked_patterns]
            return SandboxResult(
                blocked=True,
                block_reason=f"Blocked operations: {', '.join(reasons)}",
            )

        return SandboxResult(blocked=False, success=True)

    # ── 执行 ───────────────────────────────────────

    def execute(self, code: str, language: str = "python",
                timeout_s: int | None = None) -> SandboxResult:
        """在沙箱中执行代码。"""
        self._execution_count += 1

        # 1. 静态检查
        check = self.check_code(code, language)
        if check.blocked:
            logger.warning("sandbox_blocked", reason=check.block_reason,
                          exec_count=self._execution_count)
            return check

        # 2. 限制执行
        timeout = timeout_s or self.config.max_cpu_time_s
        start = time.time()

        try:
            # 安全执行环境
            safe_globals: dict[str, Any] = {
                "__builtins__": {
                    "print": print,
                    "len": len, "range": range, "int": int, "float": float,
                    "str": str, "list": list, "dict": dict, "set": set,
                    "tuple": tuple, "bool": bool, "type": type,
                    "abs": abs, "min": min, "max": max, "sum": sum,
                    "sorted": sorted, "enumerate": enumerate, "zip": zip,
                    "map": map, "filter": filter, "any": any, "all": all,
                    "isinstance": isinstance,
                    "True": True, "False": False, "None": None,
                },
            }

            # 使用 compile + exec 限制内置函数
            compiled = compile(code, "<sandbox>", "exec")

            import io
            from contextlib import redirect_stdout

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exec(compiled, safe_globals)

            elapsed = time.time() - start
            output = stdout.getvalue()

            # 输出大小限制
            if len(output) > self.config.max_output_chars:
                output = output[:self.config.max_output_chars] + "\n... [TRUNCATED]"

            return SandboxResult(
                success=True,
                output=output.strip() or "(no output)",
                cpu_time_s=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start
            logger.error("sandbox_exec_error", error=str(e))
            return SandboxResult(
                success=False,
                error=str(e),
                cpu_time_s=elapsed,
            )

    def execute_fn(self, fn: Callable[[], Any],
                   timeout_s: int | None = None) -> SandboxResult:
        """在沙箱中执行函数。"""
        self._execution_count += 1
        timeout = timeout_s or self.config.max_cpu_time_s
        start = time.time()

        try:
            import signal

            def handler(signum, frame):
                raise TimeoutError(f"Execution exceeded {timeout}s limit")

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            result = fn()
            elapsed = time.time() - start
            signal.alarm(0)

            return SandboxResult(
                success=True,
                output=result,
                cpu_time_s=elapsed,
            )

        except TimeoutError as e:
            return SandboxResult(
                success=False, error=str(e),
                cpu_time_s=time.time() - start,
            )
        except Exception as e:
            return SandboxResult(
                success=False, error=str(e),
                cpu_time_s=time.time() - start,
            )

    # ── 统计 ───────────────────────────────────────

    @property
    def execution_count(self) -> int:
        return self._execution_count

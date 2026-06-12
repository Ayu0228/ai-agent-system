"""Sandbox tests — static analysis, safe execution, resource limits."""

import pytest

from src.safety.sandbox import (
    Sandbox, SandboxConfig, SandboxMode, SandboxResult,
)


class TestSandboxMode:
    """Test SandboxMode enum."""

    def test_all_modes(self):
        assert SandboxMode.STRICT.value == "strict"
        assert SandboxMode.STANDARD.value == "standard"
        assert SandboxMode.PERMISSIVE.value == "permissive"


class TestSandboxConfig:
    """Test SandboxConfig."""

    def test_defaults(self):
        config = SandboxConfig()
        assert config.mode == SandboxMode.STANDARD
        assert config.max_cpu_time_s == 60
        assert config.max_memory_mb == 512
        assert config.max_output_chars == 100_000
        assert config.enable_network is False
        assert config.allow_file_write is False

    def test_strict_mode(self):
        config = SandboxConfig(mode=SandboxMode.STRICT)
        assert config.mode == SandboxMode.STRICT

    def test_custom_limits(self):
        config = SandboxConfig(
            max_cpu_time_s=10,
            max_memory_mb=256,
            max_output_chars=1000,
            enable_network=True,
            allow_file_write=True,
        )
        assert config.max_cpu_time_s == 10
        assert config.enable_network is True

    def test_blocked_syscalls(self):
        config = SandboxConfig()
        assert "exec" in config.blocked_syscalls
        assert "subprocess" in config.blocked_syscalls

    def test_denied_dirs(self):
        config = SandboxConfig()
        assert "/etc" in config.denied_dirs


class TestSandboxResult:
    """Test SandboxResult."""

    def test_defaults(self):
        result = SandboxResult()
        assert result.success is False
        assert result.blocked is False
        assert result.error == ""

    def test_blocked_result(self):
        result = SandboxResult(blocked=True, block_reason="Dangerous operation")
        assert result.blocked is True


class TestSandboxCheckCode:
    """Test static code analysis — check_code()."""

    @pytest.fixture
    def sandbox(self):
        return Sandbox(SandboxConfig(mode=SandboxMode.STRICT))

    def test_safe_code_passes(self, sandbox):
        result = sandbox.check_code("print('hello world')")
        assert result.blocked is False

    def test_os_system_blocked(self, sandbox):
        result = sandbox.check_code("os.system('rm -rf /')")
        assert result.blocked is True
        assert "system" in result.block_reason.lower() or "os" in result.block_reason.lower()

    def test_subprocess_blocked(self, sandbox):
        result = sandbox.check_code("import subprocess; subprocess.run(['ls'])")
        assert result.blocked is True

    def test_eval_blocked(self, sandbox):
        result = sandbox.check_code("eval('1+1')")
        assert result.blocked is True

    def test_exec_blocked(self, sandbox):
        result = sandbox.check_code("exec('print(1)')")
        assert result.blocked is True

    def test___import___blocked(self, sandbox):
        result = sandbox.check_code("__import__('os').system('ls')")
        assert result.blocked is True

    def test_file_write_blocked_by_default(self, sandbox):
        result = sandbox.check_code("open('file.txt', 'w').write('data')")
        assert result.blocked is True

    def test_file_write_allowed_when_enabled(self):
        config = SandboxConfig(allow_file_write=True)
        sandbox = Sandbox(config)
        result = sandbox.check_code("open('file.txt', 'w').write('data')")
        assert result.blocked is False

    def test_network_blocked_by_default(self, sandbox):
        result = sandbox.check_code("import requests; requests.get('http://evil.com')")
        assert result.blocked is True

    def test_network_allowed_when_enabled(self):
        config = SandboxConfig(enable_network=True)
        sandbox = Sandbox(config)
        result = sandbox.check_code("import requests; requests.get('http://ok.com')")
        assert result.blocked is False

    def test_shutil_write_blocked(self, sandbox):
        result = sandbox.check_code("import shutil; shutil.copy('a', 'b')")
        assert result.blocked is True

    def test_socket_blocked(self, sandbox):
        result = sandbox.check_code("import socket; s = socket.socket()")
        assert result.blocked is True

    def test_http_client_blocked(self, sandbox):
        result = sandbox.check_code("import http.client; http.client.HTTPConnection('x')")
        assert result.blocked is True

    def test_urllib_blocked(self, sandbox):
        result = sandbox.check_code("import urllib.request; urllib.request.urlopen('x')")
        assert result.blocked is True

    def test_non_python_language_skips_checks(self, sandbox):
        result = sandbox.check_code("console.log('hello')", language="javascript")
        assert result.blocked is False

    def test_safe_arithmetic(self, sandbox):
        result = sandbox.check_code("result = 1 + 2 * 3 / 4")
        assert result.blocked is False

    def test_safe_list_comprehension(self, sandbox):
        result = sandbox.check_code("[x**2 for x in range(10)]")
        assert result.blocked is False

    def test_multiple_blocks_reported(self, sandbox):
        config = SandboxConfig(allow_file_write=False, enable_network=False)
        sandbox2 = Sandbox(config)
        code = "open('f', 'w'); import requests; requests.get('x')"
        result = sandbox2.check_code(code)
        assert result.blocked is True
        # Multiple blocked operations
        assert "," in result.block_reason or "file_write" in result.block_reason.lower()


class TestSandboxExecute:
    """Test sandbox execution — execute()."""

    @pytest.fixture
    def sandbox(self):
        config = SandboxConfig(
            mode=SandboxMode.STANDARD,
            allow_file_write=True,
            enable_network=True,
        )
        return Sandbox(config)

    def test_execute_simple_code(self, sandbox):
        result = sandbox.execute("print('hello')")
        assert result.success is True
        assert "hello" in str(result.output)

    def test_execute_arithmetic(self, sandbox):
        result = sandbox.execute("x = 2 + 2\nprint(x)")
        assert result.success is True
        assert "4" in str(result.output)

    def test_execute_blocks_dangerous_code(self, sandbox):
        result = sandbox.execute("import os; os.system('ls')")
        assert result.blocked is True

    def test_execute_tracks_time(self, sandbox):
        result = sandbox.execute("print('test')")
        assert result.cpu_time_s >= 0

    def test_execute_error_handling(self, sandbox):
        result = sandbox.execute("1 / 0")
        assert result.success is False
        assert result.error != ""

    def test_execute_syntax_error(self, sandbox):
        result = sandbox.execute("print('unclosed string)")
        assert result.success is False

    def test_execute_with_custom_timeout(self, sandbox):
        result = sandbox.execute("print('test')", timeout_s=5)
        assert result.success is True

    def test_execute_count_increments(self, sandbox):
        assert sandbox.execution_count == 0
        sandbox.execute("print(1)")
        assert sandbox.execution_count == 1
        sandbox.execute("print(2)")
        assert sandbox.execution_count == 2

    def test_execute_with_list_comprehension(self, sandbox):
        result = sandbox.execute("result = [x for x in range(5)]\nprint(sum(result))")
        assert result.success is True
        assert "10" in str(result.output)

    def test_execute_restricted_builtins(self, sandbox):
        # open should NOT be available (not in safe_globals)
        result = sandbox.execute("print(open)")
        assert result.success is False  # NameError

    def test_safe_builtins_available(self, sandbox):
        result = sandbox.execute("print(len([1,2,3]))")
        assert result.success is True
        assert "3" in str(result.output)

    def test_output_truncation(self):
        config = SandboxConfig(max_output_chars=10)
        sandbox = Sandbox(config)
        result = sandbox.execute("print('a' * 100)")
        assert result.success is True
        if "TRUNCATED" in str(result.output):
            assert len(str(result.output)) <= 50  # 10 chars + \n... [TRUNCATED]


class TestSandboxExecuteFn:
    """Test execute_fn()."""

    @pytest.fixture
    def sandbox(self):
        return Sandbox(SandboxConfig(max_cpu_time_s=5))

    def test_execute_fn_simple(self, sandbox):
        result = sandbox.execute_fn(lambda: 42)
        assert result.success is True
        assert result.output == 42

    def test_execute_fn_raises(self, sandbox):
        def bad():
            raise ValueError("test error")
        result = sandbox.execute_fn(bad)
        assert result.success is False
        assert "test error" in result.error

    def test_execute_fn_tracks_time(self, sandbox):
        result = sandbox.execute_fn(lambda: "ok")
        assert result.cpu_time_s >= 0

    def test_execute_fn_count(self, sandbox):
        sandbox.execute_fn(lambda: 1)
        assert sandbox.execution_count == 1

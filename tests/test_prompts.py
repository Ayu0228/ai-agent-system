"""Prompt 版本管理测试。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from src.prompts.manager import PromptManager, PromptVersion, PromptFile


class TestPromptVersion:
    """PromptVersion 数据模型测试。"""

    def test_create_version(self):
        v = PromptVersion(
            name="researcher/web_search",
            version="1.2.0",
            content="搜索主题: {{topic}}",
            changelog="优化搜索词构造逻辑",
        )
        assert v.name == "researcher/web_search"
        assert v.version == "1.2.0"
        assert v.content == "搜索主题: {{topic}}"
        assert v.sha256 != ""

    def test_sha256_consistent(self):
        v1 = PromptVersion(name="t", version="1.0", content="hello")
        v2 = PromptVersion(name="t", version="1.0", content="hello")
        assert v1.sha256 == v2.sha256

    def test_sha256_different_for_different_content(self):
        v1 = PromptVersion(name="t", version="1.0", content="hello")
        v2 = PromptVersion(name="t", version="1.0", content="world")
        assert v1.sha256 != v2.sha256


class TestPromptFile:
    """PromptFile 版本查询测试。"""

    def test_latest_returns_last_version(self):
        v1 = PromptVersion(name="r/search", version="1.0", content="a")
        v2 = PromptVersion(name="r/search", version="1.1", content="b")
        pf = PromptFile(name="r", path=Path("/tmp/r.yaml"), versions=[v1, v2])
        assert pf.latest == v2

    def test_latest_returns_none_when_empty(self):
        pf = PromptFile(name="r", path=Path("/tmp/r.yaml"))
        assert pf.latest is None

    def test_get_version_exact_match(self):
        v1 = PromptVersion(name="r/search", version="1.0", content="a")
        v2 = PromptVersion(name="r/search", version="2.0", content="b")
        pf = PromptFile(name="r", path=Path("/tmp/r.yaml"), versions=[v1, v2])
        assert pf.get_version("1.0") == v1
        assert pf.get_version("2.0") == v2
        assert pf.get_version("3.0") is None

    def test_get_by_sha_partial(self):
        v = PromptVersion(name="r/search", version="1.0", content="unique content")
        pf = PromptFile(name="r", path=Path("/tmp/r.yaml"), versions=[v])
        found = pf.get_by_sha(v.sha256[:8])
        assert found == v
        assert pf.get_by_sha("deadbeef") is None


class TestPromptManager:
    """PromptManager 加载和查询测试。"""

    @pytest.fixture
    def prompts_dir(self, temp_dir):
        d = temp_dir / "prompts"
        d.mkdir()
        return d

    def test_load_single_template_yaml_flat_format(self, prompts_dir):
        """扁平格式: templates: { name: | content }"""
        file = prompts_dir / "test.yaml"
        file.write_text("""
templates:
  step1: |
    这是第一步
  step2: |
    这是第二步
version: "1.0.0"
""")

        mgr = PromptManager(str(prompts_dir))
        content = mgr.get("test", "step1")
        assert content is not None
        assert "第一步" in content

    def test_load_nested_prompts_format(self, prompts_dir):
        """嵌套格式: prompts: { name: { template: | ... } }"""
        file = prompts_dir / "researcher.yaml"
        file.write_text("""
prompts:
  web_search:
    template: |
      搜索主题: {{topic}}
      要求: 至少 3 个信息源
  fact_check:
    template: |
      请验证以下陈述: {{statement}}
      标注置信度
version: "2.0.0"
changlog: 初始版本
""")

        mgr = PromptManager(str(prompts_dir))
        content = mgr.get("researcher", "web_search")
        assert content is not None
        assert "搜索主题" in content
        assert "{{topic}}" in content

    def test_load_real_config_files(self, prompts_dir):
        """集成测试: 加载真实配置目录中的所有 YAML。"""
        import shutil
        real_dir = Path(__file__).parent.parent / "config" / "prompts"
        if real_dir.exists():
            for f in real_dir.glob("*.yaml"):
                shutil.copy(f, prompts_dir / f.name)

        mgr = PromptManager(str(prompts_dir))
        assert mgr.file_count > 0
        assert mgr.prompt_count > 0

    def test_reload_detects_new_files(self, prompts_dir):
        file = prompts_dir / "a.yaml"
        file.write_text("templates:\n  x: |\n    hello\nversion: \"1.0\"\n")
        mgr = PromptManager(str(prompts_dir))
        assert mgr.get("a", "x") == "hello\n"

        # 添加新文件但不 reload — 不应该自动找到
        file2 = prompts_dir / "b.yaml"
        file2.write_text("templates:\n  y: |\n    world\nversion: \"1.0\"\n")
        assert mgr.get("b", "y") is None

        # reload 后应该找到
        mgr.reload()
        assert mgr.get("b", "y") == "world\n"

    def test_get_nonexistent_returns_none(self, prompts_dir):
        mgr = PromptManager(str(prompts_dir))
        assert mgr.get("nonexistent") is None
        assert mgr.get("nonexistent", "template") is None

    def test_get_version_specific(self, prompts_dir):
        file = prompts_dir / "r.yaml"
        file.write_text("templates:\n  s: |\n    v1 content\nversion: \"1.0.0\"\n")
        mgr = PromptManager(str(prompts_dir))
        v = mgr.get_version("r", "1.0.0")
        assert v is not None
        assert "v1 content" in v

    def test_get_by_sha_immutable_ref(self, prompts_dir):
        file = prompts_dir / "r.yaml"
        file.write_text("templates:\n  s: |\n    immutable\nversion: \"1.0\"\n")
        mgr = PromptManager(str(prompts_dir))
        pf = mgr._cache["r"]
        sha = pf.latest.sha256
        content = mgr.get_by_sha("r", sha)
        assert content is not None
        assert "immutable" in content

    def test_list_prompts_returns_all(self, prompts_dir):
        file = prompts_dir / "r.yaml"
        file.write_text("""
templates:
  a: |\n    A
  b: |\n    B
version: "1.0"
""")
        mgr = PromptManager(str(prompts_dir))
        items = mgr.list_prompts()
        assert len(items) == 2
        names = {it["name"] for it in items}
        assert "r/a" in names
        assert "r/b" in names

    def test_diff_versions(self, prompts_dir):
        file = prompts_dir / "r.yaml"
        file.write_text("templates:\n  s: |\n    short\nversion: \"1.0\"\n")
        mgr = PromptManager(str(prompts_dir))
        result = mgr.diff("r", "1.0", "1.0")
        assert result["len_diff"] == 0

    def test_autoreload_after_cache_expiry(self, prompts_dir):
        file = prompts_dir / "r.yaml"
        file.write_text("templates:\n  s: |\n    v1\nversion: \"1.0\"\n")
        mgr = PromptManager(str(prompts_dir))
        # 手动设置 last_reload 到很久以前
        mgr._last_reload = 0
        # 修改文件
        file.write_text("templates:\n  s: |\n    v2 updated\nversion: \"1.1\"\n")
        content = mgr.get("r", "s")
        assert content is not None

    def test_standalone_version_without_templates(self, prompts_dir):
        """整个文件作为一个 prompt（没有 templates/prompts key）。"""
        file = prompts_dir / "standalone.yaml"
        file.write_text("name: standalone\nversion: \"3.0.0\"\ncontent: |\n  这是完整 prompt 内容\n")
        mgr = PromptManager(str(prompts_dir))
        assert mgr.get("standalone") is not None

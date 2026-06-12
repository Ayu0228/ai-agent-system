"""Prompt 版本管理 — GitOps 驱动的 Prompt 工程化。

- Git 仓库存储 Prompt 文件（Markdown + YAML frontmatter）
- SemVer 版本控制
- 热重载支持
- 不可变版本引用

ref: Spotify Claude Code + GitOps pattern (2025)
ref: LangSmith PromptLayer — artifact-level version histories
ref: "Prompts are code. Version them, test them, govern them, monitor them."
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.shared.config import get_project_root

logger = structlog.get_logger()


@dataclass
class PromptVersion:
    """单个 Prompt 版本。"""

    name: str
    version: str                       # SemVer MAJOR.MINOR.PATCH
    content: str                       # Prompt 正文
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str = ""                   # 内容哈希，用于不可变引用
    created_at: float = 0.0
    changelog: str = ""

    def __post_init__(self) -> None:
        if not self.sha256 and self.content:
            self.sha256 = hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class PromptFile:
    """Prompt 文件 = 一个 prompt 的所有版本历史。"""

    name: str
    path: Path
    versions: list[PromptVersion] = field(default_factory=list)

    @property
    def latest(self) -> PromptVersion | None:
        return self.versions[-1] if self.versions else None

    def get_version(self, version: str) -> PromptVersion | None:
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def get_by_sha(self, sha: str) -> PromptVersion | None:
        for v in self.versions:
            if v.sha256.startswith(sha):
                return v
        return None


class PromptManager:
    """Prompt 管理器 — 加载、版本管理、热重载。

    目录结构:
        config/prompts/
        ├── researcher.yaml     # YAML frontmatter 格式
        ├── copywriter.yaml
        └── templates/
            ├── article.md
            └── report.md

    YAML 格式:
        ```yaml
        name: researcher
        version: 1.2.0
        changelog: "优化了搜索词构造逻辑"
        templates:
          web_search: |
            ...
          fact_check: |
            ...
        ```
    """

    def __init__(self, prompts_dir: str = "") -> None:
        if prompts_dir:
            self._dir = Path(prompts_dir)
        else:
            self._dir = get_project_root() / "config" / "prompts"
        self._cache: dict[str, PromptFile] = {}
        self._last_reload: float = 0.0
        self.reload()

    # ── 加载 ───────────────────────────────────────

    def reload(self) -> None:
        """热重载所有 Prompt 文件。"""
        self._cache.clear()
        self._last_reload = time.time()

        if not self._dir.exists():
            logger.warning("prompts_dir_not_found", path=str(self._dir))
            return

        for fpath in sorted(self._dir.glob("*.yaml")):
            try:
                pf = self._load_file(fpath)
                self._cache[pf.name] = pf
            except Exception as e:
                logger.error("prompt_load_error", file=str(fpath), error=str(e))

        # 也加载 templates/ 子目录的 .md 文件
        templates_dir = self._dir / "templates"
        if templates_dir.exists():
            for fpath in sorted(templates_dir.glob("*.md")):
                try:
                    pf = self._load_markdown_template(fpath)
                    self._cache[pf.name] = pf
                except Exception as e:
                    logger.error("template_load_error", file=str(fpath), error=str(e))

        logger.info("prompts_reloaded", count=len(self._cache))

    def _load_file(self, fpath: Path) -> PromptFile:
        raw = fpath.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        name = fpath.stem

        versions: list[PromptVersion] = []
        # 多模板 YAML — 支持 "templates" 或 "prompts" key
        templates = data.get("templates") or data.get("prompts")
        if templates and isinstance(templates, dict):
            for tpl_name, tpl_entry in templates.items():
                # 支持两种格式：
                #   1. 扁平: `web_search: | ...` (字符串)
                #   2. 嵌套: `web_search: { template: | ... }` (dict)
                if isinstance(tpl_entry, dict):
                    tpl_content = str(tpl_entry.get("template", str(tpl_entry)))
                else:
                    tpl_content = str(tpl_entry)
                template_keys = {"templates", "prompts"}
                v = PromptVersion(
                    name=f"{name}/{tpl_name}",
                    version=str(data.get("version", "1.0.0")),
                    content=tpl_content,
                    metadata={k: v for k, v in data.items() if k not in template_keys},
                    sha256=hashlib.sha256(tpl_content.encode()).hexdigest()[:16],
                    created_at=fpath.stat().st_mtime,
                    changelog=str(data.get("changelog", "")),
                )
                versions.append(v)
        else:
            # 整个文件作为一个 prompt
            v = PromptVersion(
                name=name,
                version=data.get("version", "1.0.0"),
                content=raw,
                metadata=data,
                sha256=hashlib.sha256(raw.encode()).hexdigest()[:16],
                created_at=fpath.stat().st_mtime,
                changelog=data.get("changelog", ""),
            )
            versions.append(v)

        return PromptFile(name=name, path=fpath, versions=versions)

    def _load_markdown_template(self, fpath: Path) -> PromptFile:
        raw = fpath.read_text(encoding="utf-8")
        name = f"template/{fpath.stem}"

        v = PromptVersion(
            name=name,
            version="1.0.0",
            content=raw,
            sha256=hashlib.sha256(raw.encode()).hexdigest()[:16],
            created_at=fpath.stat().st_mtime,
        )
        return PromptFile(name=name, path=fpath, versions=[v])

    # ── 查询 ───────────────────────────────────────

    def get(self, name: str, template: str = "") -> str | None:
        """获取 Prompt 内容。如果指定 template，返回特定模板。

        用法:
            mgr.get("researcher", "web_search")
            mgr.get("researcher")
        """
        if name not in self._cache:
            # 自动 reload
            if time.time() - self._last_reload > 30:
                self.reload()
            if name not in self._cache:
                return None

        pf = self._cache[name]
        if template:
            full_name = f"{name}/{template}"
            for v in pf.versions:
                if v.name == full_name:
                    return v.content
            return None

        if pf.latest:
            return pf.latest.content
        return None

    def get_version(self, name: str, version: str) -> str | None:
        """获取特定版本的 Prompt。"""
        pf = self._cache.get(name)
        if not pf:
            return None
        pv = pf.get_version(version)
        return pv.content if pv else None

    def get_by_sha(self, name: str, sha: str) -> str | None:
        """通过内容哈希获取 Prompt（不可变引用）。"""
        pf = self._cache.get(name)
        if not pf:
            return None
        pv = pf.get_by_sha(sha)
        return pv.content if pv else None

    # ── 版本信息 ───────────────────────────────────

    def list_prompts(self) -> list[dict[str, Any]]:
        """列出所有 Prompt。"""
        result: list[dict[str, Any]] = []
        for name, pf in self._cache.items():
            for v in pf.versions:
                result.append({
                    "name": v.name,
                    "version": v.version,
                    "sha256": v.sha256,
                    "changelog": v.changelog,
                    "file": str(pf.path),
                })
        return result

    def diff(
        self, name: str, v1: str, v2: str
    ) -> dict[str, Any]:
        """比较两个版本的差异。"""
        pf = self._cache.get(name)
        if not pf:
            return {"error": f"prompt not found: {name}"}

        pv1 = pf.get_version(v1)
        pv2 = pf.get_version(v2)
        if not pv1 or not pv2:
            return {"error": "version not found"}

        return {
            "name": name,
            "v1": v1,
            "v2": v2,
            "len_diff": len(pv2.content) - len(pv1.content),
            "sha256_changed": pv1.sha256 != pv2.sha256,
        }

    # ── 统计 ───────────────────────────────────────

    @property
    def prompt_count(self) -> int:
        return sum(len(pf.versions) for pf in self._cache.values())

    @property
    def file_count(self) -> int:
        return len(self._cache)


# 模块级单例
_prompt_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_prompt(name: str, template: str = "") -> str:
    """快捷函数：获取 Prompt 内容。"""
    mgr = get_prompt_manager()
    result = mgr.get(name, template)
    if result is None:
        logger.warning("prompt_not_found", name=name, template=template)
        return ""
    return result

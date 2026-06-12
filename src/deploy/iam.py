"""IAM Manager — 基于角色的访问控制 (RBAC)。

ref: AWS IAM — resource-level permissions, role-based access
ref: Anthropic Workspaces — multi-tenant isolation with API-key scoping

权限模型:
  - Role: 角色（admin / operator / developer / viewer）
  - Permission: 操作权限（agent:invoke, workflow:run, config:write 等）
  - AccessPolicy: 资源级策略（哪些 agent/workflow/session 可访问）
  - API Key scoping: 每个 key 绑定一个 role + namespace
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class Permission(str, Enum):
    """操作权限枚举。"""
    # Agent
    AGENT_INVOKE = "agent:invoke"
    AGENT_CREATE = "agent:create"
    AGENT_DELETE = "agent:delete"
    AGENT_CONFIGURE = "agent:configure"

    # Workflow
    WORKFLOW_RUN = "workflow:run"
    WORKFLOW_CREATE = "workflow:create"
    WORKFLOW_DELETE = "workflow:delete"

    # Config
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"

    # Observability
    OBSERVE_READ = "observe:read"
    OBSERVE_MANAGE = "observe:manage"

    # Cost
    COST_READ = "cost:read"
    COST_MANAGE = "cost:manage"

    # Safety
    SAFETY_OVERRIDE = "safety:override"

    # Admin
    ADMIN_ALL = "admin:*"


# 预定义角色及权限
ROLE_PERMISSIONS: dict[str, list[Permission]] = {
    "admin": [
        Permission.ADMIN_ALL,
    ],
    "operator": [
        Permission.AGENT_INVOKE, Permission.AGENT_CONFIGURE,
        Permission.WORKFLOW_RUN, Permission.WORKFLOW_CREATE,
        Permission.CONFIG_READ, Permission.CONFIG_WRITE,
        Permission.OBSERVE_READ, Permission.OBSERVE_MANAGE,
        Permission.COST_READ, Permission.COST_MANAGE,
        Permission.SAFETY_OVERRIDE,
    ],
    "developer": [
        Permission.AGENT_INVOKE, Permission.AGENT_CONFIGURE,
        Permission.WORKFLOW_RUN, Permission.WORKFLOW_CREATE,
        Permission.CONFIG_READ,
        Permission.OBSERVE_READ,
        Permission.COST_READ,
    ],
    "viewer": [
        Permission.CONFIG_READ,
        Permission.OBSERVE_READ,
        Permission.COST_READ,
    ],
}


@dataclass
class Role:
    """IAM 角色。"""
    name: str
    permissions: list[Permission] = field(default_factory=list)

    def has_permission(self, permission: Permission) -> bool:
        if Permission.ADMIN_ALL in self.permissions:
            return True
        return permission in self.permissions


@dataclass
class AccessPolicy:
    """资源级访问策略。"""
    name: str
    allowed_agents: list[str] = field(default_factory=list)     # 空=全部
    denied_agents: list[str] = field(default_factory=list)
    allowed_workflows: list[str] = field(default_factory=list)
    denied_workflows: list[str] = field(default_factory=list)
    max_daily_tokens: int = 0                                   # 0=无限制
    max_concurrent_sessions: int = 5
    ip_whitelist: list[str] = field(default_factory=list)       # CIDR

    def can_access_agent(self, agent_id: str) -> bool:
        if agent_id in self.denied_agents:
            return False
        if not self.allowed_agents:
            return True
        return agent_id in self.allowed_agents

    def can_access_workflow(self, workflow_id: str) -> bool:
        if workflow_id in self.denied_workflows:
            return False
        if not self.allowed_workflows:
            return True
        return workflow_id in self.allowed_workflows


@dataclass
class ApiKey:
    """API 密钥 — 绑定到 role + policy。"""
    key_hash: str                           # SHA256(key)
    key_prefix: str                         # 前 8 字符，用于识别
    name: str = ""
    role: str = "viewer"
    policy_names: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0                 # 0=永不过期
    last_used_at: float = 0.0
    enabled: bool = True


@dataclass
class AccessDecision:
    """访问控制决策。"""
    allowed: bool
    reason: str = ""
    required_permission: str = ""
    role: str = ""


class IAMManager:
    """IAM 管理器 — RBAC + 资源策略 + API Key 管理。

    用法:
        iam = IAMManager()
        iam.create_role("developer", ROLE_PERMISSIONS["developer"])
        iam.create_policy("dev-policy", AccessPolicy(
            name="dev-policy",
            allowed_agents=["researcher", "copywriter"],
            max_daily_tokens=100_000,
        ))
        key = iam.create_api_key(name="dev-key-1", role="developer",
                                 policies=["dev-policy"])

        # 鉴权
        decision = iam.authorize(key, Permission.AGENT_INVOKE, agent_id="researcher")
        if not decision.allowed:
            raise PermissionDenied(decision.reason)
    """

    def __init__(self) -> None:
        self._roles: dict[str, Role] = {}
        self._policies: dict[str, AccessPolicy] = {}
        self._api_keys: dict[str, ApiKey] = {}  # hash → ApiKey

    # ── 角色管理 ───────────────────────────────────

    def create_role(self, name: str, permissions: list[Permission]) -> Role:
        role = Role(name=name, permissions=list(permissions))
        self._roles[name] = role
        logger.info("iam_role_created", role=name, perm_count=len(permissions))
        return role

    def get_role(self, name: str) -> Role | None:
        return self._roles.get(name)

    def create_default_roles(self) -> None:
        for name, perms in ROLE_PERMISSIONS.items():
            self.create_role(name, perms)

    # ── 策略管理 ───────────────────────────────────

    def create_policy(self, name: str, policy: AccessPolicy) -> None:
        policy.name = name
        self._policies[name] = policy

    def get_policy(self, name: str) -> AccessPolicy | None:
        return self._policies.get(name)

    # ── API Key 管理 ───────────────────────────────

    def create_api_key(self, name: str = "", role: str = "viewer",
                       policies: list[str] | None = None,
                       expires_in_days: int = 0) -> tuple[str, ApiKey]:
        """创建 API Key。返回 (原始 key, ApiKey 对象)。

        原始 key 只在此时可见，之后只存哈希。
        """
        raw_key = "ak-" + secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        api_key = ApiKey(
            key_hash=key_hash,
            key_prefix=raw_key[:11],  # "ak-xxxxxxxx"
            name=name,
            role=role,
            policy_names=policies or [],
            created_at=time.time(),
            expires_at=time.time() + expires_in_days * 86400 if expires_in_days else 0,
        )
        self._api_keys[key_hash] = api_key
        logger.info("iam_api_key_created", name=name, role=role,
                    prefix=api_key.key_prefix)
        return raw_key, api_key

    def revoke_key(self, key_hash: str) -> bool:
        if key_hash in self._api_keys:
            self._api_keys[key_hash].enabled = False
            logger.info("iam_api_key_revoked", hash=key_hash[:8])
            return True
        return False

    def rotate_key(self, old_key_hash: str, name: str = "") -> tuple[str, ApiKey] | None:
        """轮换密钥 — 撤销旧 key 并创建新 key。"""
        old = self._api_keys.get(old_key_hash)
        if not old:
            return None
        self.revoke_key(old_key_hash)
        return self.create_api_key(
            name=name or old.name,
            role=old.role,
            policies=list(old.policy_names),
        )

    # ── 鉴权 ───────────────────────────────────────

    def authorize(self, raw_key: str, permission: Permission,
                  agent_id: str = "", workflow_id: str = "",
                  session_id: str = "") -> AccessDecision:
        """鉴权: 检查 API Key → Role → Permission → Policy。"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self._api_keys.get(key_hash)

        if not api_key:
            return AccessDecision(allowed=False, reason="invalid api key")

        if not api_key.enabled:
            return AccessDecision(allowed=False, reason="api key revoked",
                                  role=api_key.role)

        if api_key.expires_at > 0 and time.time() > api_key.expires_at:
            return AccessDecision(allowed=False, reason="api key expired",
                                  role=api_key.role)

        # 检查角色权限
        role = self._roles.get(api_key.role)
        if not role:
            return AccessDecision(allowed=False, reason=f"unknown role: {api_key.role}")

        if not role.has_permission(permission):
            return AccessDecision(
                allowed=False,
                reason=f"role '{api_key.role}' lacks permission '{permission.value}'",
                required_permission=permission.value,
                role=api_key.role,
            )

        # 检查资源策略
        for pname in api_key.policy_names:
            policy = self._policies.get(pname)
            if not policy:
                continue
            if agent_id and not policy.can_access_agent(agent_id):
                return AccessDecision(
                    allowed=False,
                    reason=f"policy '{pname}' denies access to agent '{agent_id}'",
                    required_permission=permission.value,
                    role=api_key.role,
                )
            if workflow_id and not policy.can_access_workflow(workflow_id):
                return AccessDecision(
                    allowed=False,
                    reason=f"policy '{pname}' denies access to workflow '{workflow_id}'",
                    required_permission=permission.value,
                    role=api_key.role,
                )

        # 更新使用时间
        api_key.last_used_at = time.time()

        return AccessDecision(allowed=True, role=api_key.role)

    def authenticate(self, raw_key: str) -> ApiKey | None:
        """仅认证（不鉴权）。返回 ApiKey 或 None。"""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self._api_keys.get(key_hash)
        if not api_key or not api_key.enabled:
            return None
        if api_key.expires_at > 0 and time.time() > api_key.expires_at:
            return None
        return api_key

    # ── 统计 ───────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        keys = list(self._api_keys.values())
        return {
            "roles": len(self._roles),
            "policies": len(self._policies),
            "api_keys_total": len(keys),
            "api_keys_active": sum(1 for k in keys if k.enabled),
            "api_keys_expired": sum(
                1 for k in keys if k.expires_at > 0 and time.time() > k.expires_at
            ),
        }

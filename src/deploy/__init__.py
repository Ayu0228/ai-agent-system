"""Deployment & IAM platform.

ref: Argo Rollouts + Flagger — canary deployment with metric-gated promotion
ref: Grafana Cloud + OpenLIT — zero-code K8s operator pattern for AI observability
ref: AWS IAM — RBAC with resource-level permissions
"""

from src.deploy.canary import CanaryDeployment, CanaryConfig, CanaryStep, CanaryStatus
from src.deploy.rollback import RollbackManager, RollbackTrigger, RollbackPolicy
from src.deploy.iam import IAMManager, Role, Permission, AccessPolicy

__all__ = [
    "CanaryDeployment", "CanaryConfig", "CanaryStep", "CanaryStatus",
    "RollbackManager", "RollbackTrigger", "RollbackPolicy",
    "IAMManager", "Role", "Permission", "AccessPolicy",
]

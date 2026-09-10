"""Retry eligibility and attempt limits."""

from harbor_tasks.config import DEFAULT_MAX_ATTEMPTS
from harbor_tasks.policies import ToolPolicy


def is_retry_eligible(policy: ToolPolicy) -> bool:
    """Only replay-safe, side-effect-free tools are retryable."""

    return policy.idempotent and not policy.side_effect


def maximum_attempts(policy: ToolPolicy) -> int:
    """Return the total number of attempts available for this policy."""

    if not is_retry_eligible(policy):
        return 1

    return DEFAULT_MAX_ATTEMPTS

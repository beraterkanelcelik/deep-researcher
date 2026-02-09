"""Pass@k and pass^k metric calculations for agent evals.

Based on Anthropic's "Demystifying Evals for AI Agents" article.
"""

from math import comb

from .harness import Trial


def pass_at_k(trials: list[Trial], k: int) -> float:
    """P(at least 1 success in k trials) = 1 - C(n-c, k) / C(n, k).

    Where n = total trials, c = successful trials.
    This is the unbiased estimator from the Codex paper.
    """
    n = len(trials)
    c = sum(1 for t in trials if t.success)

    if n < k:
        k = n
    if k == 0:
        return 0.0
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0

    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_pow_k(trials: list[Trial], k: int) -> float:
    """P(all k succeed) = (c/n)^k.

    More conservative metric — measures consistency.
    """
    n = len(trials)
    if n == 0:
        return 0.0

    c = sum(1 for t in trials if t.success)
    rate = c / n
    return rate**k

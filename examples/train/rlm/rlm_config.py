"""Generator config extensions for the Recursive Language Model (RLM) environment.

These fields are RLM-specific and live outside the base ``GeneratorConfig`` so that
non-RLM training runs do not surface them. Wired into a ``SkyRLTrainConfig`` via
``make_config(generator_cls=RLMGeneratorConfig)`` in the RLM entry points.
"""

from dataclasses import dataclass
from typing import Optional

from skyrl.train.config import GeneratorConfig


@dataclass
class RLMGeneratorConfig(GeneratorConfig):
    train_child_trajectories: bool = False
    """Include child RLM agent trajectories in the training batch, with reward propagated from the parent."""
    enable_child_agents: bool = True
    """When False, skip subcall_fn injection for RLM envs so the top-level agent runs without
    child-spawning capability (single-paper mode)."""
    frozen_openrouter_model: Optional[str] = None
    """When set, in-REPL ``llm_query`` calls use this frozen model via OpenRouter
    instead of the policy engine."""
    hosted_openrouter_model: Optional[str] = None
    """When set, the eval entry point uses OpenRouter as the primary policy engine
    instead of launching / connecting to a local vLLM deployment."""
    hosted_openrouter_reasoning_effort: str = "none"
    """Reasoning effort for the hosted OpenRouter policy model."""
    judge_model: Optional[str] = None
    """Override the evidence judge model id (e.g. ``moonshotai/kimi-k3``)."""
    judge_base_url: Optional[str] = None
    """Override the evidence judge OpenAI-compatible base URL."""
    judge_reasoning_effort: str = "low"
    """Reasoning effort for the judge model when the endpoint supports it."""
    judge_max_concurrency: int = 1
    """Maximum concurrent evidence-judge requests per Ray worker."""
    judge_min_interval_seconds: float = 5.0
    """Minimum spacing between evidence-judge requests per Ray worker."""
    trace_output_dir: Optional[str] = None
    """When set, RLMGymGenerator writes visualization-compatible per-rollout traces here."""

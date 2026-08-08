"""Eval-only entry point for the Recursive Language Model (RLM) environment.

Mirrors ``skyrl.train.entrypoints.main_generate`` but uses ``RLMConfig`` and
constructs ``RLMGymGenerator`` via an overridden ``get_generator`` so the
RLM-specific hooks fire during eval rollouts too.

When ``generator.hosted_openrouter_model`` is set, the inference client is an
``OpenRouterInferenceClient`` pointed at that hosted model instead of local vLLM.
"""

import asyncio
import os
import sys

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg

from .openrouter_client import OpenRouterInferenceClient
from .rlm_config import RLMGeneratorConfig
from .rlm_generator import RLMGymGenerator


RLMConfig = make_config(generator_cls=RLMGeneratorConfig)


class RLMEvalEntrypoint(EvalOnlyEntrypoint):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return RLMGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_inference_client(self) -> InferenceEngineInterface:
        hosted_model = getattr(self.cfg.generator, "hosted_openrouter_model", None)
        if not hosted_model:
            return super().get_inference_client()

        logger.info(f"Using hosted OpenRouter policy model: {hosted_model}")
        # Tokenizer is still required for SkyRL bookkeeping / chat templates.
        # Prefer the configured policy path (often a local SFT checkpoint) so
        # apply_chat_template stays available even when generation is remote.
        return OpenRouterInferenceClient.from_model(
            model=hosted_model,
            tokenizer=self.tokenizer,
            reasoning_effort=getattr(self.cfg.generator, "hosted_openrouter_reasoning_effort", "none"),
        )


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg) -> dict:
    # Ensure OpenRouter credentials are visible inside the Ray worker.
    for src, dst in (("OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"), ("OPENROUTER_API_KEY", "OPEN_ROUTER_KEY")):
        if os.environ.get(src) and not os.environ.get(dst):
            os.environ[dst] = os.environ[src]

    exp = RLMEvalEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = RLMConfig.from_cli_overrides(sys.argv[1:])
    # Hosted OpenRouter eval does not launch local engines; relax placement checks.
    if getattr(cfg.generator, "hosted_openrouter_model", None):
        cfg.generator.inference_engine.run_engines_locally = False
        cfg.generator.inference_engine.external_proxy_url = (
            cfg.generator.inference_engine.external_proxy_url or "https://openrouter.ai/api/v1"
        )
        if cfg.trainer.placement.colocate_all:
            cfg.trainer.placement.colocate_all = False
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()

"""``OpenRouterInferenceClient``: a ``RemoteInferenceClient`` subclass that routes
generation to the OpenRouter chat-completions API.

Used as:
- the primary policy engine for hosted eval (``generator.hosted_openrouter_model``)
- the frozen in-REPL ``llm_query`` engine (``generator.frozen_openrouter_model``)
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
from transformers import PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)


def resolve_openrouter_api_key(explicit: Optional[str] = None) -> str:
    """Accept either local env naming convention used in this workspace."""
    if explicit:
        return explicit
    return os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_KEY") or ""


@dataclass
class OpenRouterInferenceClient(RemoteInferenceClient):
    """OpenRouter-backed inference client.

    Subclasses ``RemoteInferenceClient`` and overrides ``generate()`` to speak
    OpenAI chat-completions (with Bearer auth) instead of vLLM's token-id
    endpoint. Control-plane methods are no-ops since OpenRouter is a stateless
    external API.

    Set ``prefers_chat_prompts=True`` so ``SkyRLGymGenerator.agent_loop`` passes
    structured chat histories instead of collapsing them through token decode.
    """

    BASE_URL: str = field(default="https://openrouter.ai/api/v1", init=False, repr=False)

    api_key: str = field(default="", repr=False)
    """OpenRouter API key. Falls back to OPENROUTER_API_KEY / OPEN_ROUTER_KEY."""

    reasoning_effort: str = "none"
    """OpenRouter reasoning effort forwarded as ``reasoning: {effort}``."""

    prefers_chat_prompts: bool = True
    """Signal to agent_loop that this client wants role-preserving chat messages."""

    usage: Dict[str, Any] = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "cached_tokens": 0,
            "completion_tokens": 0,
            "requests": 0,
            "reasoning_tokens": 0,
            "cost": 0.0,
            "cost_reported": False,
        },
        repr=False,
    )
    """Per-call usage counters for observability."""

    def __post_init__(self):
        if not self.api_key:
            self.api_key = resolve_openrouter_api_key()
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY or OPEN_ROUTER_KEY must be set when using OpenRouterInferenceClient"
            )

    @classmethod
    def from_model(
        cls,
        model: str,
        tokenizer: PreTrainedTokenizerBase,
        api_key: Optional[str] = None,
        reasoning_effort: str = "none",
    ) -> "OpenRouterInferenceClient":
        """Convenience constructor matching the old OpenRouterInferenceEngine signature."""
        base_url = "https://openrouter.ai/api/v1"
        return cls(
            proxy_url=base_url,
            server_urls=[base_url],
            data_parallel_size=1,
            model_name=model,
            tokenizer=tokenizer,
            api_key=api_key or "",
            reasoning_effort=reasoning_effort,
            prefers_chat_prompts=True,
        )

    async def generate(self, input_batch: InferenceEngineInput, model: Optional[str] = None) -> InferenceEngineOutput:
        """Send batched chat-completions requests to OpenRouter.

        Prefers structured ``prompts`` (chat histories). Falls back to decoding
        ``prompt_token_ids`` into a single user message when needed.
        """
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        sampling_params: Dict[str, Any] = input_batch.get("sampling_params") or {}

        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided.")

        # Prefer role-preserving chat messages when both are present.
        if prompts is not None:
            message_lists: List[List[Dict[str, str]]] = list(prompts)
        else:
            assert prompt_token_ids is not None
            message_lists = [
                [{"role": "user", "content": self.tokenizer.decode(ids, skip_special_tokens=True)}]
                for ids in prompt_token_ids
            ]

        max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_generate_length", 1024)

        body_template: Dict[str, Any] = {
            "model": model or self.model_name,
            "temperature": sampling_params.get("temperature", 0.7),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_tokens": max_tokens,
            "reasoning": {"effort": sampling_params.get("reasoning_effort", self.reasoning_effort)},
        }
        if sampling_params.get("additional_kwargs"):
            body_template.update(sampling_params["additional_kwargs"])

        session = await self._get_session()

        async def _post_one(messages: List[Dict[str, str]]) -> Dict[str, Any]:
            body = dict(body_template)
            body["messages"] = messages
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            last_exc: Optional[Exception] = None
            for attempt in range(3):
                try:
                    async with session.post(
                        f"{self.BASE_URL}/chat/completions",
                        json=body,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=300),
                    ) as resp:
                        resp.raise_for_status()
                        return await resp.json()
                except Exception as e:
                    last_exc = e
                    if attempt < 2:
                        await asyncio.sleep(2**attempt)
            raise RuntimeError(f"OpenRouter API call failed after 3 attempts: {last_exc}") from last_exc

        responses_data = await asyncio.gather(*(_post_one(msgs) for msgs in message_lists))

        responses: List[str] = []
        response_ids: List[List[int]] = []
        stop_reasons: List[str] = []
        for data in responses_data:
            api_usage = data.get("usage", {}) or {}
            self.usage["prompt_tokens"] += api_usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += api_usage.get("completion_tokens", 0)
            self.usage["requests"] += 1
            self.usage["cached_tokens"] += (api_usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            reasoning_tokens = (api_usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            self.usage["reasoning_tokens"] += reasoning_tokens
            if "cost" in api_usage and api_usage["cost"] is not None:
                self.usage["cost"] += api_usage["cost"]
                self.usage["cost_reported"] = True

            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            text = message.get("content", "") or ""
            responses.append(text)
            response_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
            stop_reasons.append(choice.get("finish_reason") or "stop")

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
            rollout_expert_indices=None,
        )

    # ------------------------------------------------------------------
    # Control-plane no-ops (OpenRouter is a stateless external API)
    # ------------------------------------------------------------------

    async def finish_session(self, session_id: str) -> None:
        return None

    async def pause(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def resume(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def pause_generation(self) -> None:
        pass

    async def resume_generation(self) -> None:
        pass

    async def sleep(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def wake_up(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def init_weight_update_communicator(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def update_named_weights(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def reset_prefix_cache(self) -> None:
        pass

    async def teardown(self) -> None:
        # Defer to RemoteInferenceClient so the shared aiohttp session is closed.
        await super().teardown()

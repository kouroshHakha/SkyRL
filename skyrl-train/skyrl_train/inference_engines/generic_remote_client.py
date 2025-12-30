"""Generic remote inference client for external inference servers.

This module provides a simple client that communicates with inference servers
via HTTP, without being tightly coupled to a specific backend like vLLM or SGLang.
"""

import aiohttp
import asyncio
import json
from typing import Any, Dict

from loguru import logger
from omegaconf import DictConfig

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync import WeightLoader


class RemoteWeightLoader(WeightLoader):
    """Loads weights into remote inference engine via HTTP.

    This loader coordinates weight updates with remote inference servers
    via their HTTP APIs.
    """

    def __init__(self, url: str, model_name: str) -> None:
        """Initialize the loader.

        Args:
            url: Base URL of the remote inference server.
            model_name: Name of the model to update weights for.
        """
        self._url = url
        self._model_name = model_name

    async def init_communicator(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        override_existing: bool = False,
    ) -> Dict[str, Any]:
        """Initialize the distributed process group for syncing weights.

        Args:
            master_address: Master address for the process group.
            master_port: Master port for the process group.
            rank_offset: Rank offset for this process.
            world_size: Total world size.
            group_name: Name of the process group.
            backend: Backend to use (e.g., "nccl", "gloo").
            override_existing: Whether to override an existing group.

        Returns:
            Response from the remote server.
        """
        path = "/init_weight_update_communicator"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._url}{path}",
                json={
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                    "override_existing": override_existing,
                },
            ) as response:
                return await response.json()

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> Dict[str, Any]:
        """Load weights via HTTP to the remote inference server.

        Remote engines only support broadcast weight updates (no IPC).
        Each request should contain a single weight to update.

        Args:
            request: Weight update request containing names, dtypes, shapes.

        Returns:
            Response from the remote server.
        """
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self._url}/collective_rpc",
                json={
                    "model": self._model_name,
                    "method": "load_weights",
                    "kwargs": {
                        "request": request
                    },
                },
            )
            return await resp.json()

    async def destroy_group(self) -> Dict[str, Any]:
        """Destroy the weights update group.

        Returns:
            Response from the remote server.
        """
        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self._url}/collective_rpc",
                json={
                    "model": self._model_name,
                    "method": "destroy_weights_update_group",
                },
            )
            return await resp.json()


async def _get_world_size(url: str, model_name: str) -> int:
    """Get the world size from the remote server.

    Args:
        url: Base URL of the remote inference server.
        model_name: Name of the model.

    Returns:
        The world size of the inference server.
    """
    async with aiohttp.ClientSession() as session:
        resp = await session.get(f"{url}/server_info")
        server_info = await resp.json()
        return server_info["world_size"]


class GenericRemoteInferenceClient(InferenceEngineInterface):
    """Generic client for remote inference engines.

    This client communicates with inference servers via HTTP without
    being tightly coupled to a specific backend.
    """

    def __init__(self, cfg: DictConfig):
        """Initialize the client.

        Args:
            cfg: Configuration object containing generator settings.
        """
        assert (
            len(cfg.generator.remote_inference_engine_urls) == 1
        ), "Only one inference engine URL is supported"

        self._url = cfg.generator.remote_inference_engine_urls[0]
        self._model_name = cfg.generator.model_name
        self._weight_loader = RemoteWeightLoader(self._url, self._model_name)

        self._world_size = asyncio.run(_get_world_size(self._url, self._model_name))

    @property
    def world_size(self) -> int:
        """Return the world size of the inference engine."""
        return self._world_size

    async def sleep(self, *args: Any, **kwargs: Any):
        """Put the inference engine to sleep."""
        async with aiohttp.ClientSession() as session:
            logger.info(f"Calling {self._url}/sleep with params: {kwargs}")
            resp = await session.post(
                f"{self._url}/sleep",
                json={
                    "model": self._model_name,
                    "options": {"level": kwargs.get("level", 1)},
                },
            )
            return await resp.json()

    async def wake_up(self, *args: Any, **kwargs: Any):
        """Wake up the inference engine."""
        async with aiohttp.ClientSession() as session:
            logger.info(f"Calling {self._url}/wakeup with params: {kwargs}")
            resp = await session.post(
                f"{self._url}/wakeup",
                json={
                    "model": self._model_name,
                    "options": {"tags": kwargs.get("tags", 1)},
                },
            )
            return await resp.json()

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        """Initialize the distributed process group for syncing weights."""
        logger.info(
            f"Calling {self._url}/init_weight_update_communicator with params: "
            f"{master_addr}, {master_port}, {rank_offset}, {world_size}, "
            f"{group_name}, {backend}, {override_existing}"
        )
        return await self._weight_loader.init_communicator(
            master_addr,
            master_port,
            rank_offset,
            world_size,
            group_name,
            backend,
            override_existing,
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        """Update named weights on the inference engine."""
        logger.info(f"Calling {self._url}/update_named_weights with params: {request}")
        if "names" not in request:
            raise ValueError(
                f"Expected update weight request with 'names' entry, got keys: {request.keys()}"
            )

        assert len(request["names"]) == 1, (
            f"Remote inference engines support only requests with a single named weight "
            f"at a time, got request with {len(request['names'])} entries"
        )

        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. "
                "Only local engines support IPC."
            )

        return await self._weight_loader.load_weights(request)

    async def reset_prefix_cache(self):
        """Reset the prefix cache on the inference engine."""
        logger.info(f"Calling {self._url}/reset_prefix_cache")
        reset_prefix_cache_method = "reset_prefix_cache"

        async with aiohttp.ClientSession() as session:
            resp = await session.post(
                f"{self._url}/{reset_prefix_cache_method}",
                json={"model": self._model_name},
            )
            text = await resp.text()

        # First try to parse it as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If invalid JSON, return raw text plus status
            return {
                "status": resp.status,
                "body": text,
            }

    # Methods that are not implemented for generic remote inference engine

    def tp_size(self) -> int:
        raise NotImplementedError(
            "tp_size is not supported for generic remote inference engine"
        )

    def pp_size(self) -> int:
        raise NotImplementedError(
            "pp_size is not supported for generic remote inference engine"
        )

    def dp_size(self) -> int:
        raise NotImplementedError(
            "dp_size is not supported for generic remote inference engine"
        )

    def ep_size(self) -> int:
        raise NotImplementedError(
            "ep_size is not supported for generic remote inference engine"
        )

    async def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "generate is not supported for generic remote inference engine"
        )

    async def chat_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "chat_completion is not supported for generic remote inference engine"
        )

    async def completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "completion is not supported for generic remote inference engine"
        )

    async def abort_generation(self) -> None:
        raise NotImplementedError(
            "abort_generation is not supported for generic remote inference engine"
        )

    async def teardown(self) -> None:
        raise NotImplementedError(
            "teardown is not supported for generic remote inference engine"
        )


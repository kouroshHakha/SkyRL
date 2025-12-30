
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp
import hydra
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir, validate_cfg
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInterface
import aiohttp

from skyrl_train.weight_sync import WeightLoader

from loguru import logger
import asyncio

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.remote_inference_engine import RemoteInferenceEngine

import json

from typing import Any, Dict




class RemoteWeightLoader(WeightLoader):
    """Loads weights into remote inference engine via HTTP.

    This loader coordinates weight updates with remote inference servers
    (vLLM or SGLang) via their HTTP APIs.
    """

    def __init__(self, url: str, model_name: str) -> None:
        """Initialize the loader.

        Args:
            url: Base URL of the remote inference server.
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
        path = "/collective_rpc"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._url}{path}",
                json={
                    "model": self._model_name,
                    "method": "init_weight_update_communicator",
                    "kwargs": {
                        "master_address": master_address,
                        "master_port": master_port,
                        "rank_offset": rank_offset,
                        "world_size": world_size,
                        "group_name": group_name,
                        "backend": backend,
                        "override_existing": override_existing,
                    },
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
                        "names": request["names"],
                        "dtypes": request["dtypes"],
                        "shapes": request["shapes"],
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
            resp = await session.post(f"{self._url}/collective_rpc", json={
                "model": self._model_name,
                "method": "destroy_weights_update_group",
            })
            return await resp.json()


async def _get_world_size(url: str, model_name: str) -> int:
    async with aiohttp.ClientSession() as session:
        resp = await session.get(f"{url}/server_info")
        server_info = await resp.json()
        return server_info[model_name]["world_size"]
    
class GenericRemoteInferenceClient(RemoteInferenceEngine):
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        assert len(cfg.inference_engine.urls) == 1, "Only one inference engine URL is supported"
        
        self._url = cfg.inference_engine.urls[0]
        self._model_name = cfg.inference_engine.model_name
        self._weight_loader = RemoteWeightLoader(
            self._url, self._model_name)
        
        self._world_size = asyncio.run(_get_world_size(self._url, self._model_name))
    
    
    @property
    def world_size(self) -> int:
        return self._world_size
    
    
    async def sleep(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            # TODO(Charlie): this is vLLM's API, not SGLang (which uses tags). Fix when need to
            # support sleeping with remote engines.
            resp = await session.post(f"{self.url}/sleep", json={"model": self._model_name, "options": {"level": kwargs.get("level", 1)}})
            return await resp.json()
    
    async def wake_up(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/wake_up", json={"model": self._model_name, "options": {"tags": kwargs.get("tags", 1)}})
            return await resp.json()
        
    
    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize the distributed process group for syncing weights."""
        return await self._weight_loader.init_communicator(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )
        
    
    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        assert (
            len(request["names"]) == 1
        ), f"Remote inference engines support only requests with a single named weight at a time , got request with {len(request['names'])} entries"

        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. Only local engines support IPC."
            )

        return await self._weight_loader.load_weights(request)
    

    async def reset_prefix_cache(self):
        reset_prefix_cache_method = "reset_prefix_cache"

        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self._url}/{reset_prefix_cache_method}", json={"model": self._model_name})
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
        
    #### Things that are bypassed due to abstraction
    
    def tp_size(self) -> int:
        raise NotImplementedError("tp_size is not supported for generic remote inference engine")
    
    def pp_size(self) -> int:
        raise NotImplementedError("pp_size is not supported for generic remote inference engine")
    
    def dp_size(self) -> int:
        raise NotImplementedError("dp_size is not supported for generic remote inference engine")
    
    def ep_size(self) -> int:
        raise NotImplementedError("ep_size is not supported for generic remote inference engine")
    
    async def generate(self, *args, **kwargs) -> Any:
        raise NotImplementedError("generate is not supported for generic remote inference engine")
    
    
    async def chat_completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError("chat_completion is not supported for generic remote inference engine")
    
    async def completion(self, *args, **kwargs) -> Any:
        raise NotImplementedError("completion is not supported for generic remote inference engine")
    
    


class NewMainExp(BasePPOExp):
    def get_inference_engine_client(self) -> InferenceEngineClient:
        """Initializes the inference engine client.

        Returns:
            InferenceEngineClient: The inference engine client.
        """
        return GenericRemoteInferenceClient(self.cfg)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = NewMainExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()

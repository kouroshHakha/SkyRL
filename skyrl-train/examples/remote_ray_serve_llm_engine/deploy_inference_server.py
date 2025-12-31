from ray.serve.llm import LLMConfig, build_openai_app
from ray import serve
from ray.serve.config import HTTPOptions

from ray.llm._internal.serve.core.ingress.dev_ingress import DevIngress
import ray

from ray.llm._internal.serve.observability.logging import get_logger


import asyncio
from ray import serve
from ray.llm._internal.common.dict_utils import (
    maybe_apply_llm_deployment_config_defaults,
)
from ray.llm._internal.serve.core.server.llm_server import LLMServer
from ray.llm._internal.serve.core.ingress.builder import LLMServingArgs
from ray.llm._internal.serve.core.ingress.ingress import (
    DEFAULT_ENDPOINTS,
    OpenAiIngress,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.utils.broadcast import broadcast
from ray.llm._internal.serve.core.ingress.mixins import (
    CacheManagerIngressMixin,
    CollectiveRpcIngressMixin,
    PausableIngressMixin,
    SleepableIngressMixin,
    TokenizationIngressMixin,
)
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.deployment import Application
import pprint
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
from ray.serve._private.common import DeploymentID
from ray.serve.schema import ReplicaRank



logger = get_logger(__name__)

    
class DeploymentInfo(BaseModel):
    world_size: int
    
    
  
class InitWeightUpdateCommunicatorRequest(BaseModel):
    master_address: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str
    override_existing: bool

class LoadWeightsRequest(BaseModel):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    sizes: Optional[List[int]] = None
    extras: Optional[List[Dict[str, Any]]] = {}
    packed: Optional[bool] = False

    
class SkyRLLLMServer(LLMServer):
    
    
    async def init_weight_update_communicator(
        self, 
        request: InitWeightUpdateCommunicatorRequest
    ):
        """Initialize the distributed process group for syncing weights."""
        return await self.collective_rpc(
            "init_weight_update_communicator",
            kwargs=request.model_dump()
        )

      
class SkyRLIngress(DevIngress):
    ENDPOINTS = {
        "server_info": lambda app: app.get("/server_info"),
        "init_weight_update_communicator": lambda app: app.post("/init_weight_update_communicator"),
    }
    
    def __init__(self, llm_deployments):
        assert len(llm_deployments) == 1, "Only one llm deployment is supported"
        super().__init__(llm_deployments)
        
        self._handle = next(iter(llm_deployments))
        
    async def _get_num_devices_per_replica(self):
        llm_config = await self._handle.llm_config.remote()
        return llm_config.get_engine_config().num_devices
    
    async def _get_replica_ranks_mapping(self) -> Dict[str, ReplicaRank]:
        """Get mapping of replica_id (unique_id) -> ReplicaRank from controller."""
        controller = ray.get_actor(
            SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
        deployment_id = self._handle.deployment_id
        return await controller._get_replica_ranks_mapping.remote(deployment_id)

    async def init_weight_update_communicator(
        self, 
        request: InitWeightUpdateCommunicatorRequest
    ):
        """Initialize the distributed process group for syncing weights."""
        num_devices_per_replica = await self._get_num_devices_per_replica()
        
        # Fetch replica ranks from the controller
        replica_ranks = await self._get_replica_ranks_mapping()
        
        def get_kwargs(replica):
            # Get the unique_id from the replica
            replica_unique_id = replica.replica_id.unique_id
            
            # Look up the rank from the mapping
            replica_rank_info = replica_ranks.get(replica_unique_id)
            if replica_rank_info is None:
                raise RuntimeError(f"Rank not found for replica {replica_unique_id}")
            
            # Calculate rank_offset: global_rank * num_devices_per_replica + base_offset
            rank_offset = replica_rank_info.rank * num_devices_per_replica + request.rank_offset
            
            return {"request": InitWeightUpdateCommunicatorRequest(
                master_address=request.master_address,
                master_port=request.master_port,
                rank_offset=rank_offset,
                world_size=request.world_size,
                group_name=request.group_name,
                backend=request.backend,
                override_existing=request.override_existing,
            )}
        
        results = await asyncio.to_thread(
            broadcast, 
            self._handle, 
            "init_weight_update_communicator", 
            kwargs=get_kwargs
        )
        return results

    async def server_info(self) -> DeploymentInfo:
        num_devices_per_replica = await self._get_num_devices_per_replica()
        replica_ranks = await self._get_replica_ranks_mapping()
        
        num_replicas = len(replica_ranks)
        world_size = num_replicas * num_devices_per_replica
        
        return DeploymentInfo(
            world_size=world_size,
        )
        

# Endpoint map for DevIngress - includes all default endpoints plus control plane
ENDPOINTS = {
    **CacheManagerIngressMixin.ENDPOINTS,
    **CollectiveRpcIngressMixin.ENDPOINTS,
    **PausableIngressMixin.ENDPOINTS,
    **SleepableIngressMixin.ENDPOINTS,
    **TokenizationIngressMixin.ENDPOINTS,
    **DEFAULT_ENDPOINTS,
    **SkyRLIngress.ENDPOINTS,
}


def build_skyrl_ingress(builder_config: dict) -> Application:
    """Build an OpenAI compatible app with dev/control plane endpoints.

    This is similar to build_openai_app but uses DevIngress with
    additional control plane endpoints:
    - /sleep, /wakeup, /is_sleeping (sleep mode - offloads weights to CPU)
    - /pause, /resume, /is_paused (pause mode - keeps weights in GPU)
    - /reset_prefix_cache (cache management)
    - /collective_rpc (RLHF - execute RPC on all workers)
    - /tokenize, /detokenize (tokenization - convert text to/from token IDs)
    - /server_info (get deployment info like world_size)
    - /init_weight_update_communicator (initialize weight sync for RLHF)

    Args:
        builder_config: Configuration conforming to LLMServingArgs.
            See LLMServingArgs for details on the expected structure.

    Returns:
        The configured Ray Serve Application.

    Example:
        config = {
            "llm_configs": [llm_config],
            "ingress_deployment_config": {}
        }
        app = build_skyrl_ingress(config)
        serve.run(app)
    """
    config = LLMServingArgs.model_validate(builder_config)
    llm_configs = config.llm_configs

    llm_deployments = [build_llm_deployment(c, deployment_cls=SkyRLLLMServer) for c in llm_configs]

    ingress_cls_config = config.ingress_cls_config
    default_ingress_options = DevIngress.get_deployment_options(llm_configs)

    ingress_options = maybe_apply_llm_deployment_config_defaults(
        default_ingress_options, config.ingress_deployment_config
    )

    ingress_cls = make_fastapi_ingress(SkyRLIngress, endpoint_map=ENDPOINTS)

    logger.info("============== Ingress Options ==============")
    logger.info(pprint.pformat(ingress_options))

    return serve.deployment(ingress_cls, **ingress_options).bind(
        llm_deployments=llm_deployments, **ingress_cls_config.ingress_extra_kwargs
    )


def main():
    
    ray.init(
        runtime_env={
            "env_vars": {
                "PYTHONPATH": "/home/ray/anaconda3/lib/python3.12/site-packages"
            }
        }
    )
        
    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
        ),
        deployment_config=dict(
            num_replicas=2,
        ),
        engine_kwargs=dict(
            tensor_parallel_size=1,
            worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_num_seqs=1024,
            max_num_batched_tokens=8192,
            max_model_len=4096,
            enable_sleep_mode=True,
        ),
    )

    app = build_skyrl_ingress({"llm_configs": [llm_config]})
    http_options = HTTPOptions(host="0.0.0.0", port=8001)
    serve.start(http_options=http_options)
    serve.run(app, blocking=True)


if __name__ == "__main__":

    main()
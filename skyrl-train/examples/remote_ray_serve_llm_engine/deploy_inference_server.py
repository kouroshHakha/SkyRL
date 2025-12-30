from ray.serve.llm import LLMConfig, build_openai_app
from ray import serve
from ray.serve.config import HTTPOptions

from ray.llm._internal.serve.core.ingress.dev_ingress import DevIngress
import ray

from ray.llm._internal.serve.observability.logging import get_logger


from ray import serve
from ray.llm._internal.common.dict_utils import (
    maybe_apply_llm_deployment_config_defaults,
)
from ray.llm._internal.serve.core.ingress.builder import LLMServingArgs
from ray.llm._internal.serve.core.ingress.ingress import (
    DEFAULT_ENDPOINTS,
    OpenAiIngress,
    make_fastapi_ingress,
)
from ray.llm._internal.serve.core.ingress.mixins import (
    CacheManagerIngressMixin,
    CollectiveRpcIngressMixin,
    PausableIngressMixin,
    SleepableIngressMixin,
)
from ray.llm._internal.serve.core.server.builder import build_llm_deployment
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.deployment import Application
import pprint
from pydantic import BaseModel

logger = get_logger(__name__)

class ReplicaInfo(BaseModel):
    tp_size: int
    pp_size: int
    dp_size: int
    ep_size: int
    
class DeploymentInfo(BaseModel):
    num_replicas: int
    replica_info: ReplicaInfo
    
ServerInfo = Dict[str, DeploymentInfo]
    
    

class SkyRLIngress(DevIngress):
    ENDPOINTS = {
        "server_info": lambda app: app.get("/server_info"),
    }
    

    async def server_info(self) -> ServerInfo:
        
        for model_name in self.llm_configs.keys():
            deployment_info = DeploymentInfo(
                num_replicas=self.llm_configs[model_name].deployment_config.num_replicas,
                replica_info=ReplicaInfo(
                    tp_size=self.llm_configs[model_name].engine_kwargs.tensor_parallel_size,
                    pp_size=self.llm_configs[model_name].engine_kwargs.pipeline_parallel_size,
                    dp_size=self.llm_configs[model_name].engine_kwargs.data_parallel_size,
                    ep_size=self.llm_configs[model_name].engine_kwargs.expert_parallel_size,
                ),
            )
            
        return {model_name: deployment_info for model_name in self.llm_configs.keys()}
        
        

# Endpoint map for DevIngress - includes all default endpoints plus control plane
DEV_ENDPOINTS = {
    **CacheManagerIngressMixin.ENDPOINTS,
    **CollectiveRpcIngressMixin.ENDPOINTS,
    **PausableIngressMixin.ENDPOINTS,
    **SleepableIngressMixin.ENDPOINTS,
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
        app = build_dev_openai_app(config)
        serve.run(app)
    """
    config = LLMServingArgs.model_validate(builder_config)
    llm_configs = config.llm_configs

    llm_deployments = [build_llm_deployment(c) for c in llm_configs]

    ingress_cls_config = config.ingress_cls_config
    default_ingress_options = DevIngress.get_deployment_options(llm_configs)

    ingress_options = maybe_apply_llm_deployment_config_defaults(
        default_ingress_options, config.ingress_deployment_config
    )

    ingress_cls = make_fastapi_ingress(DevIngress, endpoint_map=DEV_ENDPOINTS)

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
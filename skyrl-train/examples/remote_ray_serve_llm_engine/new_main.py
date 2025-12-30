
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp
import hydra
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir, validate_cfg
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient

# Import the client from the proper module (not defined in __main__)
# This avoids pickle issues with cloudpickle trying to serialize __main__ module state
from skyrl_train.inference_engines.generic_remote_client import GenericRemoteInferenceClient


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

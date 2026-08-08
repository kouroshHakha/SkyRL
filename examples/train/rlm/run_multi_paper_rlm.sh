#!/usr/bin/env bash
set -euo pipefail
set -x

# Multi-paper RLM training with parent/child orchestration.
# The root agent (depth 0) coordinates by dispatching child agents to individual papers.

# Best local SFT checkpoint: alphaXiv/evidence-multi-rlm-sft-4b.
# The Kimi K3 reward needs OPEN_ROUTER_KEY or OPENROUTER_API_KEY in the
# environment. Source this workspace's .env so Ray workers inherit it.

: "${ROOT_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
if [[ -f "$ROOT_DIR/.env" ]]; then
  set +x
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
  set -x
fi

: "${DATA_DIR:=$ROOT_DIR/data/rlm-multi-paper-v1}"
: "${VAL_DATA:=$DATA_DIR/validation-first10.parquet}"
: "${MODEL_PATH:=alphaXiv/evidence-multi-rlm-sft-4b}"
: "${NUM_ENGINES:=1}"
: "${TP_SIZE:=4}"
: "${TRAIN_GPUS:=4}"
: "${LOGGER:=wandb}"
: "${INFERENCE_BACKEND:=vllm}"
: "${EPOCHS:=3}"
: "${N_SAMPLES:=4}"
: "${MAX_TURNS:=10}"
: "${EVAL_INTERVAL:=100}"
: "${CHECKPOINT_INTERVAL:=10}"
: "${MAX_TRAIN_SEQUENCE_LENGTH:=32768}"
: "${JUDGE_MAX_CONCURRENCY:=1}"
: "${JUDGE_MIN_INTERVAL_SECONDS:=5}"
: "${CKPT_PATH:=/mnt/cluster_storage/kourosh/rlm}"
: "${MAX_CKPTS_TO_KEEP:=5}"
: "${RUN_NAME:=rlm_multi_paper_grpo}"
: "${LOG_PATH:=/mnt/cluster_storage/kourosh/rlm/logs/$RUN_NAME}"
: "${EXPORT_PATH:=/mnt/cluster_storage/kourosh/rlm/exports/$RUN_NAME}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"

uv run --with "transformers==5.4.0" --extra fsdp --python 3.12 -m examples.train.rlm.main_rlm \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$VAL_DATA']" \
  environment.env_class=multipaper_evidence_rlm \
  generator.step_wise_trajectories=true \
  generator.enable_child_agents=true \
  generator.train_child_trajectories=true \
  generator.max_turns=$MAX_TURNS \
  generator.batched=false \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="$MODEL_PATH" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$TRAIN_GPUS \
  trainer.placement.ref_num_gpus_per_node=$TRAIN_GPUS \
  generator.inference_engine.num_engines=$NUM_ENGINES \
  generator.inference_engine.tensor_parallel_size=$TP_SIZE \
  trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
  trainer.epochs=$EPOCHS \
  trainer.eval_before_train=true \
  trainer.eval_interval=$EVAL_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.eval_batch_size=16 \
  trainer.train_batch_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=$CHECKPOINT_INTERVAL \
  trainer.max_ckpts_to_keep=$MAX_CKPTS_TO_KEEP \
  trainer.remove_microbatch_padding=false \
  trainer.max_prompt_length=32768 \
  trainer.max_train_sequence_length=$MAX_TRAIN_SEQUENCE_LENGTH \
  generator.sampling_params.max_generate_length=1024 \
  generator.eval_sampling_params.max_generate_length=1024 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.judge_max_concurrency=$JUDGE_MAX_CONCURRENCY \
  generator.judge_min_interval_seconds=$JUDGE_MIN_INTERVAL_SECONDS \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.01 \
  generator.inference_engine.backend=$INFERENCE_BACKEND \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.gpu_memory_utilization=0.6\
  generator.max_input_length=32768 \
  generator.inference_engine.engine_init_kwargs.language_model_only=true \
  generator.inference_engine.enforce_eager=false \
  generator.chat_template_kwargs.enable_thinking=false \
  generator.n_samples_per_prompt=$N_SAMPLES \
  trainer.logger="$LOGGER" \
  trainer.project_name="rlm" \
  trainer.run_name="$RUN_NAME" \
  trainer.log_path="$LOG_PATH" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.export_path="$EXPORT_PATH" \
  trainer.dump_eval_results=true \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  "$@"

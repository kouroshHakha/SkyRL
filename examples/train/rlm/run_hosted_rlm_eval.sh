set -x

# Hosted OpenRouter eval via main_rlm_eval.py.
# Uses OpenRouterInferenceClient as the primary policy engine and Kimi K3 as judge.
#
# Example:
#   export OPEN_ROUTER_KEY=...
#   DATA_DIR=/home/ray/default/work_rlm/data/rlm-multi-paper-v1 \
#     bash examples/train/rlm/run_hosted_rlm_eval.sh

: "${DATA_DIR:=$HOME/data/rlm-multi-paper-v1}"
: "${HOSTED_MODEL:=deepseek/deepseek-v4-flash-0731}"
: "${JUDGE_MODEL:=moonshotai/kimi-k3}"
: "${TOKENIZER_MODEL:=alphaXiv/evidence-multi-rlm-sft-4b}"
: "${EVAL_N_SAMPLES:=4}"

uv run --with "transformers==5.4.0" --extra fsdp --python 3.12 -m examples.train.rlm.main_rlm_eval \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  environment.env_class=multipaper_evidence_rlm \
  generator.step_wise_trajectories=true \
  generator.enable_child_agents=true \
  generator.train_child_trajectories=false \
  generator.max_turns=10 \
  generator.batched=false \
  generator.eval_n_samples_per_prompt=$EVAL_N_SAMPLES \
  generator.hosted_openrouter_model="$HOSTED_MODEL" \
  generator.hosted_openrouter_reasoning_effort=none \
  generator.judge_model="$JUDGE_MODEL" \
  generator.judge_base_url="https://openrouter.ai/api/v1" \
  generator.judge_reasoning_effort=low \
  trainer.policy.model.path="$TOKENIZER_MODEL" \
  trainer.placement.colocate_all=false \
  trainer.eval_batch_size=1 \
  trainer.max_prompt_length=32768 \
  generator.eval_sampling_params.max_generate_length=4096 \
  generator.eval_sampling_params.temperature=0.7 \
  generator.eval_sampling_params.top_p=1.0 \
  generator.max_input_length=32768 \
  generator.chat_template_kwargs.enable_thinking=false \
  trainer.logger="['console']" \
  trainer.project_name="rlm" \
  trainer.run_name="rlm_hosted_openrouter_eval" \
  trainer.log_path="$(pwd)/.neer/artifacts/skyrl-logs" \
  trainer.export_path="$(pwd)/.neer/artifacts/rlm_exports" \
  trainer.dump_eval_results=true \
  "$@"

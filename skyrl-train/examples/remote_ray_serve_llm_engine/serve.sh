
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="/home/ray/anaconda3/lib/python3.12/site-packages"

uv run --directory "$PROJECT_ROOT" --env-file "$SCRIPT_DIR/.env" --extra vllm0p12 python "$SCRIPT_DIR/deploy_inference_server.py"

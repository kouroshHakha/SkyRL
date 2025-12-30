
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

uv run --directory "$PROJECT_ROOT" --env-file "$SCRIPT_DIR/.env" --extra vllm0p12 python "$SCRIPT_DIR/deploy_inference_server.py"

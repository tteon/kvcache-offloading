#!/bin/bash
set -e

# Determine script directory for relative paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Default Configuration
IMAGE_NAME="lmcache/vllm-openai:latest"
MODEL_NAME="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
PORT=8000
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.75
QUANTIZATION="awq"
DTYPE="half"

# Parse Command Line Arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --max-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-util)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --quant)
      QUANTIZATION="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model <name>      Model name (default: $MODEL_NAME)"
      echo "  --port <number>     Port number (default: $PORT)"
      echo "  --max-len <int>     Max model len (default: $MAX_MODEL_LEN)"
      echo "  --gpu-util <float>  GPU memory utilization (default: $GPU_MEM_UTIL)"
      echo "  --quant <str>       Quantization (default: $QUANTIZATION)"
      echo "  --dtype <str>       Data type (default: $DTYPE)"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Logs go inside serving/serving_logs relative to this script, or we can keep them in root.
# Let's keep them in serving/serving_logs for self-containment.
LOG_DIR="$SCRIPT_DIR/serving_logs"
mkdir -p "$LOG_DIR"

# Cache and Configs are in Project Root
CACHE_DIR="$PROJECT_ROOT/cache_store"
CONFIG_DIR="$PROJECT_ROOT/configs"
mkdir -p "$CACHE_DIR" "$CONFIG_DIR"

CONTAINER_NAME="vllm_serving_container"

# Cleanup function
cleanup() {
    echo "Stopping background monitors..."
    kill $(jobs -p) 2>/dev/null || true
    echo "Stopping vLLM container..."
    docker stop "$CONTAINER_NAME" || true
}
trap cleanup EXIT

echo "Starting Profiling (Host Side)..."

# 1. GPU/PCIe Monitoring
echo "Starting nvidia-smi dmon..."
nvidia-smi dmon -s p -d 1 -c 1000 > "${LOG_DIR}/pcie_stats.csv" &

# 2. System Monitoring (Disk/CPU)
if command -v dstat &> /dev/null; then
    echo "Starting dstat..."
    dstat -d 1 > "${LOG_DIR}/disk_io_stats.csv" &
else
    echo "Warning: dstat not found, skipping system monitoring."
fi

# 3. vLLM Metrics Collector
echo "Starting metrics collector..."
(
    # Create headers for LMCache stats
    echo "timestamp,prefix_cache_hits,prefix_cache_queries,kv_cache_usage" > "${LOG_DIR}/lmcache_stats.csv"
    while true; do
        METRICS=$(curl -s "http://localhost:${PORT}/metrics")
        echo "--- $(date) ---" >> "${LOG_DIR}/vllm_metrics.log"
        echo "$METRICS" >> "${LOG_DIR}/vllm_metrics.log"
        
        # Extract LMCache/vLLM specific stats
        # Note: These regex patterns depend on exact metric names from LMCache integration
        HITS=$(echo "$METRICS" | grep 'vllm:prefix_cache_hits' | awk '{print $2}' || echo "0")
        QUERIES=$(echo "$METRICS" | grep 'vllm:prefix_cache_queries' | awk '{print $2}' || echo "0")
        USAGE=$(echo "$METRICS" | grep 'vllm:gpu_cache_usage_perc' | awk '{print $2}' || echo "0")
        
        TIMESTAMP=$(date +%s)
        echo "$TIMESTAMP,$HITS,$QUERIES,$USAGE" >> "${LOG_DIR}/lmcache_stats.csv"
        
        sleep 1
    done
) &

echo "Starting vLLM Server in Docker..."
# Ensure API Key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set in .env."
fi

# Run vLLM in Docker

# Command Construction
CMD_ARGS="--model $MODEL_NAME --quantization $QUANTIZATION --dtype $DTYPE --max-model-len $MAX_MODEL_LEN --gpu-memory-utilization $GPU_MEM_UTIL --enforce-eager --port 8000 --api-key $OPENAI_API_KEY"

if [ "$ENABLE_PROFILE" = true ]; then
    echo "Profiling ENABLED (Nsight Systems)"
    mkdir -p profiles
    
    # Nsight Arguments
    # Note: We use the same vLLM image. The image must have nsys installed or we mount it.
    # The reference 'lmcache/vllm-openai:latest' likely has it if it was used in utils.sh.
    # We mount a local profiles dir to capture output.
    
    PROFILE_CMD="nsys profile --delay=10 --duration=60 --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu --output=/profiles/vllm_profile --force-overwrite=true"
    
    # Run Docker with Profiling
    docker run \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --env OPENAI_API_KEY="$OPENAI_API_KEY" \
        --env HF_TOKEN="$HF_TOKEN" \
        -p "$PORT":8000 \
        -v "$CACHE_DIR":/data \
        -v "$CONFIG_DIR":/configs \
        -v "$(pwd)/profiles":/profiles \
        -v /dev/shm:/dev/shm \
        --entrypoint nsys \
        "$IMAGE_NAME" \
        $PROFILE_CMD \
        /opt/venv/bin/vllm serve $CMD_ARGS
else
    # Normal Run
    docker run \
        --name "$CONTAINER_NAME" \
        --gpus all \
        --env OPENAI_API_KEY="$OPENAI_API_KEY" \
        --env HF_TOKEN="$HF_TOKEN" \
        -p "$PORT":8000 \
        -v "$CACHE_DIR":/data \
        -v "$CONFIG_DIR":/configs \
        -v /dev/shm:/dev/shm \
        "$IMAGE_NAME" \
        $CMD_ARGS
fi

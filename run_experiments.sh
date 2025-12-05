#!/bin/bash
set -e

# Configuration
IMAGE_NAME="lmcache/vllm-openai:latest"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR="$(pwd)/cache_store"
CONFIG_DIR="$(pwd)/configs"
ENV_FILE="$(pwd)/.env"

# Ensure directories exist
mkdir -p "$CACHE_DIR"

# Function to start vLLM container
start_vllm() {
    local container_name=$1
    local port=$2
    local config_file=$3
    local extra_args=$4

    echo "Starting vLLM container: $container_name on port $port"
    
    docker run -d --rm --gpus all \
        --name "$container_name" \
        --env-file "$ENV_FILE" \
        -p "$port":8000 \
        -v "$CONFIG_DIR":/configs \
        -v "$CACHE_DIR":/data \
        -v /dev/shm:/dev/shm \
        -e LMCACHE_CONFIG_FILE="$config_file" \
        $extra_args \
        "$IMAGE_NAME" \
        --model "$MODEL_NAME" --gpu-memory-utilization 0.95 --max-model-len 512 --dtype half --enforce-eager --kv-cache-dtype "$CACHE_DTYPE"
        
    echo "Waiting for vLLM to be ready..."
    # Simple health check loop
    for i in {1..150}; do
        if curl -s "http://localhost:$port/health" | grep -q "ok"; then
            echo "vLLM is ready!"
            return 0
        fi
        sleep 2
    done
    echo "vLLM failed to start."
    docker logs "$container_name"
    return 1
}

stop_container() {
    local container_name=$1
    echo "Stopping container: $container_name"
    docker stop "$container_name" || true
}



start_monitoring() {
    local name=$1
    echo "Starting system monitoring for: $name"
    nvidia-smi dmon -s t -d 1 -o T > "pcie_stats_${name}.csv" &
    MONITOR_PID_1=$!
    dstat -cdngy 1 > "system_stats_${name}.csv" &
    MONITOR_PID_2=$!
    dstat -d 1 > "disk_io_stats_${name}.csv" &
    MONITOR_PID_3=$!
    # iotop requires root and interactive usually, running in batch mode
    sudo iotop -b -o -P -t -d 1 > "iotop_stats_${name}.log" &
    MONITOR_PID_4=$!
}

start_metrics_collection() {
    local name=$1
    local port=$2
    echo "Starting vLLM metrics collection for: $name on port $port"
    # Scrape metrics every second
    while true; do
        echo "--- $(date) ---" >> "vllm_metrics_${name}.log"
        curl -s "http://localhost:${port}/metrics" >> "vllm_metrics_${name}.log" || true
        sleep 1
    done &
    METRICS_PID=$!
}

stop_metrics_collection() {
    echo "Stopping metrics collection..."
    kill $METRICS_PID || true
}

stop_monitoring() {
    echo "Stopping monitoring..."
    kill $MONITOR_PID_1 $MONITOR_PID_2 $MONITOR_PID_3 $MONITOR_PID_4 || true
}

# Parse arguments
TIER="all"
PROMPT_LEN=1000
GEN_LEN=100
NUM_REQUESTS=10
CACHE_DTYPE="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    --tier)
      TIER="$2"
      shift 2
      ;;
    --model)
      MODEL_NAME="$2"
      shift 2
      ;;
    --prompt-len)
      PROMPT_LEN="$2"
      shift 2
      ;;
    --gen-len)
      GEN_LEN="$2"
      shift 2
      ;;
    --num-requests)
      NUM_REQUESTS="$2"
      shift 2
      ;;
    --cache-dtype)
      CACHE_DTYPE="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --profile)
      ENABLE_PROFILE=true
      shift 1
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Detect GPU Name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown GPU")

echo "Configuration:"
echo "  Tier: $TIER"
echo "  Device: $GPU_NAME"
echo "  Model: $MODEL_NAME"
echo "  KV Cache: $CACHE_DTYPE"
echo "  Prompt Len: $PROMPT_LEN"
echo "  Gen Len: $GEN_LEN"
echo "  Requests: $NUM_REQUESTS"
echo "  Label: ${LABEL:-auto}"
echo "  Profiling: ${ENABLE_PROFILE:-false}"

# Function to run benchmark with args
run_benchmark() {
    local name=$1
    # Use provided label or default to experiment name
    local experiment_label=${LABEL:-$name}
    local api_base=${2:-"http://localhost:8000/v1"}
    echo "Running benchmark for: $name (Label: $experiment_label)"
    python3 benchmark.py \
        --model "$MODEL_NAME" \
        --prompt-len "$PROMPT_LEN" \
        --gen-len "$GEN_LEN" \
        --num-requests "$NUM_REQUESTS" \
        --api-base "$api_base" \
        --device-name "$GPU_NAME" \
        --cache-dtype "$CACHE_DTYPE" \
        --label "$experiment_label" \
        > "results_${name}.txt"
}

# ... (monitoring functions remain same)

if [[ "$TIER" == "all" || "$TIER" == "baseline" ]]; then
    # --- Experiment 1: Baseline (No LMCache) ---
    echo "=== Running Baseline ==="
    start_monitoring "baseline"
    start_vllm "vllm_baseline" 8000 "" ""
    start_metrics_collection "baseline" 8000
    run_benchmark "baseline"
    stop_metrics_collection
    stop_container "vllm_baseline"
    stop_monitoring
fi

if [[ "$TIER" == "all" || "$TIER" == "cpu" ]]; then
    # --- Experiment 2: CPU Offload ---

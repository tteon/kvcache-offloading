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
    --gpu-memory-utilization)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Defaults
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.95}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-512}

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
echo "  GPU Mem Util: $GPU_MEM_UTIL"
echo "  Max Model Len: $MAX_MODEL_LEN"

# Create Results Directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR="results"
# Use label for folder name if provided, else tier + timestamp
if [ -n "$LABEL" ]; then # Check if LABEL is set and not empty
    RUN_DIR="${RESULTS_BASE_DIR}/${LABEL}_${TIMESTAMP}"
else
    RUN_DIR="${RESULTS_BASE_DIR}/${TIER}_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

echo "Results will be saved to: $RUN_DIR"

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
        --output-dir "$RUN_DIR" \
        > "${RUN_DIR}/benchmark_${name}.log"
}

start_container() {
    local name=$1
    local config_file=$2
    local port=$3
    local extra_args=${4:-""}
    
    echo "Starting vLLM container: $name on port $port"
    
    # Base Docker args
    local docker_args="-d --name $name --gpus all --env-file $ENV_FILE -p $port:8000 -v /dev/shm:/dev/shm -v $CONFIG_DIR:/configs -v $CACHE_DIR:/data"
    
    # Profiling args
    if [ "$ENABLE_PROFILE" = true ]; then
        echo "  -> Profiling enabled (Nsight Systems)"
        docker_args="$docker_args -v $(pwd)/profiles:/profiles --entrypoint nsys"
        # Wrap the command
        # Note: We need to reconstruct the vllm command because we overrode the entrypoint
        # Default entrypoint was ["/opt/venv/bin/vllm", "serve"]
        local cmd="profile --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu --output=/profiles/${name}_${LABEL:-auto} --force-overwrite=true /opt/venv/bin/vllm serve $MODEL_NAME --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN --dtype half --enforce-eager --kv-cache-dtype $CACHE_DTYPE"
        
        docker run $docker_args \
            -e LMCACHE_CONFIG_FILE="$config_file" \
            $extra_args \
            "$IMAGE_NAME" \
            $cmd
    else
        # Normal run
        docker run $docker_args \
            -e LMCACHE_CONFIG_FILE="$config_file" \
            $extra_args \
            "$IMAGE_NAME" \
            --model "$MODEL_NAME" --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --dtype half --enforce-eager --kv-cache-dtype "$CACHE_DTYPE"
    fi
        
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
    docker logs "$name"
    return 1
}

start_metrics_collection() {
    local name=$1
    local port=$2
    echo "Starting metrics collection for: $name"
    
    # PCIe Bandwidth
    nvidia-smi dmon -s p -d 1 -c 1000 > "${RUN_DIR}/pcie_stats_${name}.csv" &
    PCIE_PID=$!
    
    # Disk I/O (if dstat is available)
    if command -v dstat &> /dev/null; then
        dstat -d 1 > "${RUN_DIR}/disk_io_stats_${name}.csv" &
        DSTAT_PID=$!
    fi
    
    # vLLM Metrics (simple curl loop)
    # Collect every 1 second
    while true; do
        curl -s "http://localhost:$port/metrics" >> "${RUN_DIR}/system_stats_${name}.csv"
        sleep 1
    done &
    METRICS_PID=$!
}

stop_metrics_collection() {
    echo "Stopping metrics collection..."
    kill $PCIE_PID 2>/dev/null
    kill $DSTAT_PID 2>/dev/null
    kill $METRICS_PID 2>/dev/null
}

if [[ "$TIER" == "all" || "$TIER" == "baseline" ]]; then
    # --- Baseline ---
    echo "=== Running Baseline ==="
    start_container "vllm_baseline" "" 8000 ""
    start_metrics_collection "baseline" 8000
    run_benchmark "baseline"
    stop_metrics_collection
    docker stop vllm_baseline && docker rm vllm_baseline
fi

if [[ "$TIER" == "all" || "$TIER" == "cpu" ]]; then
    # --- Experiment 2: CPU Offload ---
    echo "=== Running CPU Offload ==="
    start_container "vllm_cpu" "/configs/cpu_offload.yaml" 8000 ""
    start_metrics_collection "cpu_offload" 8000
    run_benchmark "cpu_offload"
    stop_metrics_collection
    docker stop vllm_cpu && docker rm vllm_cpu
fi

if [[ "$TIER" == "all" || "$TIER" == "disk" ]]; then
    # --- Experiment 3: Disk Offload ---
    echo "=== Running Disk Offload ==="
    start_container "vllm_disk" "/configs/disk_offload.yaml" 8000 ""
    start_metrics_collection "disk_offload" 8000
    run_benchmark "disk_offload"
    stop_metrics_collection
    docker stop vllm_disk && docker rm vllm_disk
fi

if [[ "$TIER" == "all" || "$TIER" == "scalability" ]]; then
    # --- Experiment 4: Scalability (Redis) ---
    echo "=== Running Scalability (Redis) ==="
    echo "Starting Redis..."
    docker run -d --rm --name redis_cache -p 6379:6379 redis:alpine

    start_monitoring "scalability"

    # Instance A
    start_container "vllm_instance_a" "/configs/redis_offload.yaml" 8000 ""
    start_metrics_collection "scalability_A" 8000
    echo "Populating cache with Instance A..."
    run_benchmark "scalability_A"
    stop_metrics_collection

    # Instance B
    start_container "vllm_instance_b" "/configs/redis_offload.yaml" 8001 ""
    start_metrics_collection "scalability_B" 8001
    echo "Reading cache with Instance B..."
    run_benchmark "scalability_B" "http://localhost:8001/v1"
    stop_metrics_collection

    docker stop vllm_instance_a && docker rm vllm_instance_a
    docker stop vllm_instance_b && docker rm vllm_instance_b
    docker stop redis_cache
    stop_monitoring
fi

echo "Experiments completed."

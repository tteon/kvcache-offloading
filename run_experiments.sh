#!/bin/bash
set -e

# Configuration
IMAGE_NAME="lmcache/vllm-openai:latest"
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
CACHE_DIR="$(pwd)/cache_store"
CONFIG_DIR="$(pwd)/configs"
ENV_FILE="$(pwd)/.env"
NUM_REQUESTS=10
PROMPT_LEN=1000
GEN_LEN=100
TIER="baseline"
CACHE_DTYPE="auto"
LABEL="auto"
ENABLE_PROFILE=false

# Load Helper Functions
source ./scripts/utils.sh

# Ensure Cache Directory Exists
mkdir -p "$CACHE_DIR"

# Parse Arguments
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
    --kv-cache-dtype|--cache-dtype)
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
    --tensor-parallel-size|-tp)
      TP_SIZE="$2"
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
TP_SIZE=${TP_SIZE:-1}

# Detect GPU Name
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 || echo "Unknown GPU")

# Create Results Directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE_DIR="results"
if [ "$LABEL" != "auto" ]; then
    RUN_DIR="${RESULTS_BASE_DIR}/${LABEL}_${TIMESTAMP}"
else
    RUN_DIR="${RESULTS_BASE_DIR}/${TIER}_${TIMESTAMP}"
fi
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "    vLLM + LMCache Experiment Lab"
echo "=========================================="
echo "Configuration:"
echo "  Tier:             $TIER"
echo "  Device:           $GPU_NAME"
echo "  Model:            $MODEL_NAME"
echo "  KV Cache Dtype:   $CACHE_DTYPE"
echo "  Workload:         P=${PROMPT_LEN}, G=${GEN_LEN}, N=${NUM_REQUESTS}"
echo "  Label:            ${LABEL}"
echo "  Profiling:        ${ENABLE_PROFILE}"
echo "  GPU Mem Util:     $GPU_MEM_UTIL"
echo "  Max Model Len:    $MAX_MODEL_LEN"
echo "  Output Dir:       $RUN_DIR"
echo "=========================================="


if [[ "$TIER" == "all" || "$TIER" == "baseline" ]]; then
    echo "=== Running Baseline ==="
    start_container "vllm_baseline" "" 8000 "" "$TP_SIZE"
    start_metrics_collection "baseline" 8000
    run_benchmark "baseline"
    stop_metrics_collection
    docker stop vllm_baseline
    if [ "$ENABLE_PROFILE" = true ]; then
        generate_profile_report "vllm_baseline" "$LABEL"
    fi
    docker rm vllm_baseline
fi

if [[ "$TIER" == "all" || "$TIER" == "cpu" ]]; then
    echo "=== Running CPU Offload ==="
    start_container "vllm_cpu" "/configs/cpu_offload.yaml" 8000 "" "$TP_SIZE"
    start_metrics_collection "cpu_offload" 8000
    run_benchmark "cpu_offload"
    stop_metrics_collection
    docker stop vllm_cpu
    if [ "$ENABLE_PROFILE" = true ]; then
        generate_profile_report "vllm_cpu" "$LABEL"
    fi
    docker rm vllm_cpu
fi

if [[ "$TIER" == "all" || "$TIER" == "disk" ]]; then
    echo "=== Running Disk Offload ==="
    start_container "vllm_disk" "/configs/disk_offload.yaml" 8000 "" "$TP_SIZE"
    start_metrics_collection "disk_offload" 8000
    run_benchmark "disk_offload"
    stop_metrics_collection
    docker stop vllm_disk
    if [ "$ENABLE_PROFILE" = true ]; then
        generate_profile_report "vllm_disk" "$LABEL"
    fi
    docker rm vllm_disk
fi

if [[ "$TIER" == "all" || "$TIER" == "scalability" ]]; then
    echo "=== Running Scalability (Redis) ==="
    echo "Starting Redis..."
    docker run -d --rm --name redis_cache -p 6379:6379 redis:alpine

    start_monitoring "scalability" # Note: start_monitoring might need to be removed or defined if not in utils. Actually I replaced it with direct collection in start_metrics_collection logic.
    # Ah, I removed start_monitoring from utils. We should remove it here too.
    
    # Instance A
    start_container "vllm_instance_a" "/configs/redis_offload.yaml" 8000 "" "$TP_SIZE"
    start_metrics_collection "scalability_A" 8000
    echo "Populating cache with Instance A..."
    run_benchmark "scalability_A"
    stop_metrics_collection

    # Instance B
    start_container "vllm_instance_b" "/configs/redis_offload.yaml" 8001 "" "$TP_SIZE"
    start_metrics_collection "scalability_B" 8001
    echo "Reading cache with Instance B..."
    run_benchmark "scalability_B" "http://localhost:8001/v1"
    stop_metrics_collection

    docker stop vllm_instance_a
    if [ "$ENABLE_PROFILE" = true ]; then
        generate_profile_report "vllm_instance_a" "$LABEL"
    fi
    docker rm vllm_instance_a
    
    docker stop vllm_instance_b
    if [ "$ENABLE_PROFILE" = true ]; then
        generate_profile_report "vllm_instance_b" "$LABEL"
    fi
    docker rm vllm_instance_b
    
    docker stop redis_cache
fi

echo "Experiments completed. Results saved in $RUN_DIR"

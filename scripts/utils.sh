#!/bin/bash

# scripts/utils.sh

# Function to run benchmark with args
run_benchmark() {
    local name=$1
    # Use provided label or default to name
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
        mkdir -p profiles
        docker_args="$docker_args -v $(pwd)/profiles:/profiles --entrypoint nsys"
        
        # Reconstruct command for nsys
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

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
    local tp_size=${5:-1}
    
    echo "Starting vLLM container: $name on port $port"
    
    # Base Docker args
    local docker_args="-d --name $name --gpus all --env-file $ENV_FILE -p $port:8000 -v /dev/shm:/dev/shm -v $CONFIG_DIR:/configs -v $CACHE_DIR:/data"
    
    # Profiling args
    if [ "$ENABLE_PROFILE" = true ]; then
        echo "  -> Profiling enabled (Nsight Systems)"
        mkdir -p profiles
        docker_args="$docker_args -v $(pwd)/profiles:/profiles --entrypoint nsys"
        
        # Reconstruct command for nsys
        # Add duration to ensure clean exit even if docker stop is called later, 
        # or better yet, let it run for fixed time.
        local duration=${PROFILE_DURATION:-60}
        local cmd="profile --duration=$duration --trace=cuda,nvtx,osrt,cudnn,cublas --sample=cpu --output=/profiles/${name}_${LABEL:-auto} --force-overwrite=true /opt/venv/bin/vllm serve $MODEL_NAME --gpu-memory-utilization $GPU_MEM_UTIL --max-model-len $MAX_MODEL_LEN --dtype half --enforce-eager --kv-cache-dtype $CACHE_DTYPE --tensor-parallel-size $tp_size"
        
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
            --model "$MODEL_NAME" --gpu-memory-utilization "$GPU_MEM_UTIL" --max-model-len "$MAX_MODEL_LEN" --dtype half --enforce-eager --kv-cache-dtype "$CACHE_DTYPE" --tensor-parallel-size "$tp_size"
    fi
        
    echo "Waiting for vLLM to be ready..."
    # Simple health check loop
    for i in {1..150}; do
        if [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$port/health)" == "200" ]; then
            echo "vLLM is ready!"
            return 0
        fi
        sleep 2
    done
    echo "vLLM failed to start."
    docker logs "$name"
    return 1
}

generate_profile_report() {
    local name=$1
    local label=${2:-auto}
    # If label is auto, we might need to match wildcard or passed label. 
    # For simplicity, we assume the caller passes the same label used in start_container
    
    local report_path="profiles/${name}_${label}.nsys-rep"
    local output_path="profiles/${name}_${label}_stats.txt"
    
    if [ -f "$report_path" ]; then
        echo "Generating text stats for $report_path..."
        # Run nsys stats using the same container image to ensure compatibility
        # We disable set -e temporarily because nsys stats might fail on some vGPUs
        set +e
        docker run --rm -v "$(pwd)/profiles:/profiles" --entrypoint nsys "$IMAGE_NAME" stats --report gputrace,kernsum,nvtxsum "/profiles/${name}_${label}.nsys-rep" > "$output_path" 2>&1
        local exit_code=$?
        set -e
        
        if [ $exit_code -ne 0 ]; then
            echo "Warning: Failed to generate text stats (nsys stats returned $exit_code). This is common on vGPUs."
            # Don't delete the rep file, user might want to debug it
        else
            echo "Stats saved to $output_path"
        fi
    else
        echo "Warning: Profile report not found at $report_path"
    fi
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

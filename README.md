# vLLM + LMCache Experiment Lab

This repository contains a reproducible experiment lab designed to benchmark and analyze the performance of **vLLM** integrated with **LMCache**. It focuses on quantifying the effects of KV cache offloading across different storage tiers (GPU, CPU, Disk) and profiling system bottlenecks for various LLM workloads.

## 🚀 Features

*   **Multi-Tier Offloading**: Configurations for GPU-only (Baseline), CPU RAM offloading, and Local Disk offloading.
*   **Workload Simulation**: Flexible benchmark script to simulate **Long Context** (RAG) and **Agentic** workloads.
*   **Detailed Profiling**:
    *   **Latency Metrics**: Time-to-First-Token (TTFT), Inter-Token Latency (ITL), End-to-End (E2E) Latency.
    *   **System Metrics**: GPU PCIe bandwidth monitoring, CPU/Disk I/O tracking.
    *   **NVTX Annotation**: Automatic tagging of "Prefill" and "Decode" stages in Nsight timelines.
    *   **KV Cache Analysis**: Impact of offloading on memory efficiency and throughput.
*   **Visualization**: Automated plotting tools to compare performance across tiers.

## 📂 Repository Structure

```
.
├── benchmark.py            # Async OpenAI-compatible benchmark client
├── run_experiments.sh      # Orchestration script (Docker + Monitoring)
├── plot_results.py         # Analysis and plotting tool
├── configs/                # LMCache configurations
│   ├── cpu_offload.yaml    # CPU RAM offloading config
│   ├── disk_offload.yaml   # Local Disk offloading config
│   └── redis_offload.yaml  # Redis (Scalability) config
├── requirements.txt        # Python dependencies for client
└── README.md               # This file
```

## 🛠️ Prerequisites

*   **Docker** (with NVIDIA Container Toolkit support)
*   **NVIDIA GPU** (Tested on A100/H100, min 20GB VRAM for Llama-3-8B)
*   **Python 3.8+**

## ⚡ Quick Start

1.  **Setup Environment**:
    Create a `.env` file with your Hugging Face token:
    ```bash
    echo "HF_TOKEN=your_token_here" > .env
    ```

2.  **Install Client Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Experiments**:
    Use the `run_experiments.sh` script to launch the vLLM server and run benchmarks.

    *   **Baseline (GPU only)**:
        ```bash
        ./run_experiments.sh --tier baseline --model meta-llama/Meta-Llama-3-8B-Instruct
        ```
    *   **CPU Offloading**:
        ```bash
        ./run_experiments.sh --tier cpu
        ```
    *   **Disk Offloading**:
        ```bash
        ./run_experiments.sh --tier disk
        ```

## 📊 Workload Analysis

This lab allows you to simulate specific workloads to identify bottlenecks.

### 1. Long Context (RAG / Document Analysis)
*   **Characteristics**: Long input prompts, short to medium generation.
*   **Bottleneck**: Prefill Compute (TTFT) and VRAM Capacity (KV Cache).
*   **Simulation**:
    ```bash
    ./run_experiments.sh --tier disk --prompt-len 5000 --gen-len 100 --label long_context
    ```
*   **Analysis**: Check `ttft` in the results. LMCache Disk offload allows handling contexts larger than GPU RAM, though with a latency penalty during retrieval.

### 2. Agentic Workloads
*   **Characteristics**: Moderate context (System prompt + Tools), Multi-turn, Latency sensitive.
*   **Bottleneck**: Latency (TTFT for responsiveness) and Throughput (for parallel agents).
*   **Simulation**:
    ```bash
    ./run_experiments.sh --tier cpu --prompt-len 1000 --gen-len 200 --num-requests 20 --label agent_workload
    ```
*   **Analysis**: Check `avg_itl` and `e2e`. CPU offloading provides a middle ground, expanding capacity for many agent states while maintaining better latency than disk.

## 📈 Analyzing Results

The lab generates CSV files (`metrics_*.csv`) containing per-request performance data.

**Generate Comparative Plots**:
```bash
python3 plot_results.py --input "metrics_*.csv" --output-prefix "comparison"
```
This will output:
*   `comparison_ttft.png`: Time to First Token vs Sequence Length.
*   `comparison_e2e.png`: End-to-End Latency vs Sequence Length.

## 🔍 Profiling Details

*   **GPU Utilization**: The script automatically captures `nvidia-smi dmon` output to `pcie_stats_*.csv`. Use this to correlate PCIe bandwidth spikes with cache transfer events.
*   **Disk I/O**: For the `disk` tier, monitor `disk_io_stats_*.csv` to see the read/write throughput impact of LMCache.

## ⚙️ Configuration

Modify files in `configs/` to tune LMCache behavior:
*   `chunk_size`: Controls the granularity of cache transfer (default: 256).
*   `max_local_cache_size`: Limit for CPU/Disk usage.
*   `remote_url`: For Redis/Network offloading.

# vLLM + LMCache Experiment Lab

This repository contains a reproducible experiment lab designed to benchmark and analyze the performance of **vLLM** integrated with **LMCache**. It focuses on quantifying the effects of KV cache offloading across different storage tiers (GPU, CPU, Disk) and profiling system bottlenecks for various LLM workloads.

## 🚀 Features

*   **Multi-Tier Offloading**: Configurations for GPU-only (Baseline), CPU RAM offloading, and Local Disk offloading.
*   **Workload Simulation**: Flexible benchmark script to simulate **Long Context** (RAG) and **Agentic** workloads.
*   **Detailed Profiling**:
    *   **Latency Metrics**: Time-to-First-Token (TTFT), Inter-Token Latency (ITL), End-to-End (E2E) Latency.
    *   **System Metrics**: GPU PCIe bandwidth monitoring, CPU/Disk I/O tracking.
    *   **KV Cache Analysis**: Impact of offloading on memory efficiency and throughput.
*   **Visualization**: Automated plotting tools to compare performance across tiers.

## 📂 Repository Structure

```
.
├── benchmark.py            # Async OpenAI-compatible benchmark client
├── run_experiments.sh      # Orchestration script (Main entry point)
├── scripts/                # Helper scripts
│   └── utils.sh            # Common functions for container mgmt/monitoring
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

The lab generates results in the `results/` directory, organized by experiment label and timestamp (e.g., `results/baseline_20231027_120000/`).

Each folder contains:
*   `metrics_*.csv`: Per-request latency metrics.
*   `pcie_stats_*.csv`: PCIe bandwidth logs.
*   `system_stats_*.csv`: vLLM internal metrics.
*   `benchmark_*.log`: Benchmark console output.

**Generate Comparative Plots**:
You can point the plotting script to a specific results directory or use a wildcard:
```bash
python3 plot_results.py --input "results/archive/metrics_*.csv" --output-prefix "comparison"
```
This will output:
*   `comparison_ttft.png`: Time to First Token vs Sequence Length.
*   `comparison_e2e.png`: End-to-End Latency vs Sequence Length.
*   `comparison_pcie.png`: PCIe Bandwidth over Time (System Metric).
*   `comparison_disk_io.png`: Disk Read/Write Throughput over Time (System Metric).
*   `comparison_report.md`: Summary table of average latencies and peak resource usage.

## 🔍 Profiling Details

*   **Nsight Systems**:
    To capture a GPU profile (CUDA traces, NVTX, OS runtime), add the `--profile` flag:
    ```bash
    ./run_experiments.sh --tier baseline --profile
    ```
    **Outputs**:
    *   `profiles/*.nsys-rep`: Binary report. Open with Nsight Systems GUI.
    *   `profiles/*_stats.txt`: **[NEW]** Auto-generated text summary of top kernels and GPU events. Check this for quick insights without the GUI.

*   **GPU Utilization**: automatically captured in `pcie_stats_*.csv` via `nvidia-smi dmon`.
*   **Disk I/O**: captured in `disk_io_stats_*.csv` (requires `dstat`).

## ⚙️ Customization & Advanced Configuration

### 1. Hardware Adaptation
*   **Memory Tuning**:
    If you hit OOM (Out Of Memory) errors, reduce the GPU memory usage:
    ```bash
    ./run_experiments.sh --tier baseline --gpu-memory-utilization 0.85
    ```
*   **Multi-GPU (Tensor Parallelism)**:
    For systems with multiple GPUs (e.g., A100 x2, H100 x8), use `--tensor-parallel-size` (or `-tp`):
    ```bash
    ./run_experiments.sh --tier baseline --tensor-parallel-size 2
    ```

### 2. Software Tuning
*   **Model Swapping**:
    Change the target model (requires access token for gated models):
    ```bash
    ./run_experiments.sh --model "meta-llama/Llama-2-13b-hf" --tensor-parallel-size 2
    ```
*   **LMCache Configuration**:
    Edit YAML files in `configs/` to add new backends or tune chunk sizes:
    *   `chunk_size`: Larger chunks (e.g., 512) might improve disk I/O throughput but increase granularity penalty.
    *   `backend`: Switch between `cpu`, `file`, or `redis` in the YAML directly.

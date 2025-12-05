import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

def parse_system_metrics(run_dir):
    """Parses system stats CSVs in a directory."""
    pcie_files = glob.glob(os.path.join(run_dir, "pcie_stats_*.csv"))
    disk_files = glob.glob(os.path.join(run_dir, "disk_io_stats_*.csv"))
    
    metrics = {}
    
    # Parse PCIe (nvidia-smi dmon)
    # Format: # gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk    pviol    tviol    fb    bar1    sbecc    dbecc    pci    pci
    # Idx:      0      1     2     3     4     5     6     7     8     9       10       11    12      13       14       15     16     17
    # We want 16 (rx) and 17 (tx) usually, but dmon column widths vary.
    # Actually, dmon -s p output: # gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk    pviol    tviol    fb    bar1    sbecc    dbecc    pci    pci
    # Wait, -s p gives only pci metrics? No, -s p gives pci.
    # Let's assume standard dmon -s p (pci) output needs careful parsing or we assume headers.
    # The script uses `nvidia-smi dmon -s p -d 1 -c 1000`.
    # Headers: # gpu   pwr gtemp mtemp    rx    tx
    
    for f in pcie_files:
        try:
            # Skip header lines usually started with #
            # We'll try to read with pandas, skipping lines starting with # except the last one?
            # Standard dmon output is tricky. Let's try flexible whitespace sep.
            # Usually line 0 is units, line 1 is headers.
            df = pd.read_csv(f, sep=r'\s+', comment='#', names=['gpu', 'pwr', 'gtemp', 'mtemp', 'rx', 'tx'])
            # Add time relative to start
            df['time'] = df.index  # 1 second interval
            metrics[f] = {'type': 'pcie', 'data': df, 'label': os.path.basename(f).replace('pcie_stats_', '').replace('.csv', '')}
        except Exception as e:
            print(f"Failed to parse PCIe {f}: {e}")

    # Parse Disk I/O (dstat)
    for f in disk_files:
        try:
            # dstat csv usually has 5-6 header lines.
            df = pd.read_csv(f, skiprows=5)
            # Dstat columns often: "total/disk", "read", "writ"
            # We need to find read/write columns.
            read_col = [c for c in df.columns if 'read' in c.lower()][0]
            write_col = [c for c in df.columns if 'writ' in c.lower()][0]
            
            df = df[[read_col, write_col]].copy()
            df.columns = ['read_bytes', 'write_bytes']
            df['time'] = df.index # 1 second interval
            metrics[f] = {'type': 'disk', 'data': df, 'label': os.path.basename(f).replace('disk_io_stats_', '').replace('.csv', '')}
        except Exception as e:
            print(f"Failed to parse Disk {f}: {e}")
            
    return metrics

def plot_system_metrics(metrics, output_prefix):
    """Plots system metrics."""
    # Group by metric type
    pcie_data = [m for m in metrics.values() if m['type'] == 'pcie']
    disk_data = [m for m in metrics.values() if m['type'] == 'disk']
    
    if pcie_data:
        plt.figure(figsize=(12, 6))
        for m in pcie_data:
            plt.plot(m['data']['time'], m['data']['rx'], label=f"{m['label']} (RX)")
            plt.plot(m['data']['time'], m['data']['tx'], linestyle='--', label=f"{m['label']} (TX)")
        plt.xlabel("Time (s)")
        plt.ylabel("Bandwidth (MB/s)")
        plt.title("PCIe Bandwidth Usage")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(f"{output_prefix}_pcie.png")
        print(f"Saved {output_prefix}_pcie.png")

    if disk_data:
        plt.figure(figsize=(12, 6))
        for m in disk_data:
            # Convert to MB
            plt.plot(m['data']['time'], m['data']['read_bytes'] / 1024 / 1024, label=f"{m['label']} (Read)")
            plt.plot(m['data']['time'], m['data']['write_bytes'] / 1024 / 1024, linestyle='--', label=f"{m['label']} (Write)")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (MB/s)")
        plt.title("Disk I/O Usage")
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.savefig(f"{output_prefix}_disk_io.png")
        print(f"Saved {output_prefix}_disk_io.png")

def generate_summary_report(latency_df, sys_metrics, output_file):
    """Generates a markdown summary report."""
    with open(output_file, 'w') as f:
        f.write("# Experiment Summary Report\n\n")
        
        f.write("## 1. Latency Metrics\n")
        if not latency_df.empty:
            summary = latency_df.groupby('label')[['ttft', 'avg_itl', 'e2e']].mean().reset_index()
            f.write(summary.to_markdown(index=False))
        else:
            f.write("No latency data found.\n")
        f.write("\n\n")
        
        f.write("## 2. System Resource Peaks\n")
        f.write("| Label | Metric | Peak Value |\n")
        f.write("|---|---|---|\n")
        
        for m in sys_metrics.values():
            if m['type'] == 'pcie':
                peak_rx = m['data']['rx'].max()
                peak_tx = m['data']['tx'].max()
                f.write(f"| {m['label']} | PCIe RX Peak | {peak_rx:.2f} MB/s |\n")
                f.write(f"| {m['label']} | PCIe TX Peak | {peak_tx:.2f} MB/s |\n")
            elif m['type'] == 'disk':
                peak_read = m['data']['read_bytes'].max() / 1024 / 1024
                peak_write = m['data']['write_bytes'].max() / 1024 / 1024
                f.write(f"| {m['label']} | Disk Read Peak | {peak_read:.2f} MB/s |\n")
                f.write(f"| {m['label']} | Disk Write Peak | {peak_write:.2f} MB/s |\n")

def parse_and_plot(input_pattern, output_prefix):
    # Latency Plots
    files = glob.glob(input_pattern, recursive=True)
    all_data = []
    
    # 1. Latency Data
    if files:
        for f in files:
            try:
                df = pd.read_csv(f)
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
        
    combined_df = pd.DataFrame()
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        if 'label' not in combined_df.columns:
            combined_df['label'] = 'unknown'

        # Plot Latencies (Grouped by total_len)
        metrics = {
            'ttft': 'Time to First Token (s)',
            'avg_itl': 'Inter-Token Latency (s)',
            'e2e': 'End-to-End Latency (s)'
        }
        
        grouped = combined_df.groupby(['label', 'total_len'])[list(metrics.keys())].mean().reset_index()
        unique_labels = combined_df['label'].unique()
        
        for metric, ylabel in metrics.items():
            plt.figure(figsize=(10, 6))
            for label in unique_labels:
                subset = grouped[grouped['label'] == label].sort_values('total_len')
                if not subset.empty:
                    plt.plot(subset['total_len'], subset[metric], marker='o', label=label)
            plt.xlabel('Sequence Length')
            plt.ylabel(ylabel)
            plt.title(f'{ylabel} vs Sequence Length')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.savefig(f"{output_prefix}_{metric}.png")
            print(f"Saved {output_prefix}_{metric}.png")

    # 2. System Metrics
    # Assumes input_pattern is like "results/archive/metrics_*.csv"
    # We want to find the parent directories of these files to look for stats.
    # Or just search recursively in the *base* info from input pattern?
    # Simple approach: If input pattern contains a directory, scan that directory for stats.
    
    root_dir = os.path.dirname(input_pattern.split('*')[0])
    if not root_dir: root_dir = "."
    print(f"Scanning for system metrics in: {root_dir}")
    sys_metrics = parse_system_metrics(root_dir)
    plot_system_metrics(sys_metrics, output_prefix)
    
    # 3. Report
    generate_summary_report(combined_df, sys_metrics, f"{output_prefix}_report.md")
    print(f"Report saved to {output_prefix}_report.md")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Comparative Results")
    parser.add_argument("--input", type=str, default="results/archive/metrics_*.csv", help="Input CSV pattern")
    parser.add_argument("--output-prefix", type=str, default="comparison", help="Output prefix")
    args = parser.parse_args()

    parse_and_plot(args.input, args.output_prefix)

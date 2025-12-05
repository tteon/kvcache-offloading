import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_and_plot(input_pattern, output_prefix):
    files = glob.glob(input_pattern)
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            all_data.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No data loaded.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Ensure we have a 'label' column. If not (legacy files), default to 'unknown'
    if 'label' not in combined_df.columns:
        combined_df['label'] = 'unknown'

    # Metrics to plot
    metrics = {
        'ttft': 'Time to First Token (s)',
        'avg_itl': 'Inter-Token Latency (s)',
        'e2e': 'End-to-End Latency (s)'
    }

    # Group by Label and Total Length to get averages for plotting lines
    # We want to see how each metric scales with sequence length for each tier (label)
    grouped = combined_df.groupby(['label', 'total_len'])[list(metrics.keys())].mean().reset_index()
    
    unique_labels = combined_df['label'].unique()
    print(f"Found experiments: {unique_labels}")

    for metric, ylabel in metrics.items():
        plt.figure(figsize=(10, 6))
        
        for label in unique_labels:
            subset = grouped[grouped['label'] == label].sort_values('total_len')
            if not subset.empty:
                plt.plot(subset['total_len'], subset[metric], marker='o', label=label)
        
        plt.xlabel('Sequence Length (Tokens)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Sequence Length')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        out_file = f"{output_prefix}_{metric}.png"
        plt.savefig(out_file)
        print(f"Saved plot: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Comparative Results")
    parser.add_argument("--input", type=str, default="metrics_*.csv", help="Input CSV pattern")
    parser.add_argument("--output-prefix", type=str, default="comparison_plot", help="Output image prefix")
    args = parser.parse_args()

    parse_and_plot(args.input, args.output_prefix)

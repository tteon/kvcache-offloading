import math
import argparse
import pandas as pd
import numpy as np
import os
import glob
from typing import Optional, Tuple, Dict, Union

class PrefillSpeedupCalculator:
    """
    Implements the Prefill Speed Up Analysis equations.
    """

    def __init__(self, N: int, R: float, T: float, alpha: float, W: Optional[float] = None):
        self.N = N
        self.R = R
        self.T = T
        self.alpha = alpha
        self.W = W if W is not None else 0.0
        self.E = 0.0
        self.alpha_star = 0.0
        self.ttft_vanilla = 0.0
        self.ttft_offloaded = 0.0

    def calculate_metrics(self) -> Dict[str, float]:
        if self.R > 0:
            self.E = self.T / self.R
        else:
            self.E = float('inf') 

        if self.E != float('inf'):
            self.alpha_star = self.E / (1 + self.E)
        else:
            self.alpha_star = 1.0

        self.ttft_vanilla = self.N * self.T

        retrieval_cost = self.alpha * self.N * self.R
        compute_cost = (1 - self.alpha) * self.N * self.T
        write_cost = (1 - self.alpha) * self.N * self.W
        
        self.ttft_offloaded = max(retrieval_cost, compute_cost, write_cost)
        
        return {
            "E": self.E,
            "alpha_star": self.alpha_star,
            "ttft_vanilla": self.ttft_vanilla,
            "ttft_offloaded": self.ttft_offloaded
        }

    def analyze_speedup(self) -> Dict[str, str]:
        metrics = self.calculate_metrics()
        
        if self.ttft_offloaded > 0:
            speedup_actual = self.ttft_vanilla / self.ttft_offloaded
        else:
            speedup_actual = float('inf')

        if self.alpha <= self.alpha_star:
            regime = "Compute-Bound"
            insight = "Speedup is limited by GPU compute time (T). I/O is fast enough."
            optimal_strategy = "Current setup is optimal."
            speedup_theoretical = 1 / (1 - self.alpha) if (1 - self.alpha) > 0 else float('inf')
        else:
            regime = "I/O-Bound"
            insight = "Speedup is limited by Storage Speed (R). Retrieval is too slow."
            max_potential_gain = metrics['E'] + 1
            optimal_strategy = (f"OPTIMIZATION REQUIRED: Only retrieve {metrics['alpha_star']:.1%} "
                                f"of data instead of {self.alpha:.1%}. "
                                f"Potential Speedup: {max_potential_gain:.2f}x")
            
            speedup_theoretical = metrics['E'] / self.alpha if self.alpha > 0 else 0

        return {
            "Phase": "PREFILL",
            "Regime": regime,
            "Metric": "Speedup",
            "Actual": round(speedup_actual, 2),
            "Theoretical": round(speedup_theoretical, 2),
            "Alpha (Current)": f"{self.alpha:.1%}",
            "Alpha_Star (Critical)": f"{self.alpha_star:.1%}",
            "Insight": insight,
            "Strategy": optimal_strategy
        }

class DecodeSpeedupCalculator:
    """
    Calculates the Latency per Token (TPOT) for the Decode phase.
    Goal: Minimize Slowdown.
    """

    def __init__(self, L: int, R: float, T: float, alpha: float):
        self.L = L
        self.R = R
        self.T = T
        self.alpha = alpha

    def analyze(self) -> Dict[str, Union[float, str]]:
        # 1. Vanilla Latency (Baseline)
        tpot_vanilla = self.T

        # 2. Offloaded Latency
        # Retrieval time = (alpha * L) * R
        # Compute time = T
        io_time = (self.alpha * self.L) * self.R
        compute_time = self.T
        
        # With overlap, we take the maximum
        tpot_offloaded = max(compute_time, io_time)

        # 3. Calculate Slowdown
        if tpot_vanilla > 0:
            slowdown = tpot_offloaded / tpot_vanilla
        else:
            slowdown = float('inf')

        # 4. Calculate Alpha Star for Decode
        # Threshold: T = alpha_star * L * R  => alpha_star = T / (L * R)
        if (self.L * self.R) > 0:
            alpha_star = self.T / (self.L * self.R)
        else:
            alpha_star = float('inf')

        # 5. Determine Regime
        if self.alpha <= alpha_star:
            regime = "Compute-Bound (Safe)"
            insight = "Storage is fast enough to feed history. No slowdown."
            actual_impact = "No Latency Impact"
            strategy = "Current setup maintains baseline latency."
        else:
            regime = "I/O-Bound (Laggy)"
            insight = f"Context is too long for Storage speed. GPU is starving."
            actual_impact = f"{slowdown:.2f}x Slower"
            strategy = (f"CRITICAL: Reduce offload fraction to below {alpha_star:.1%} "
                        f"or switch to faster storage.")

        return {
            "Phase": "DECODE",
            "L (Context)": self.L,
            "Regime": regime,
            "Metric": "Slowdown",
            "Actual": actual_impact,
            "TPOT_Vanilla (us)": round(tpot_vanilla, 2),
            "TPOT_Offloaded (us)": round(tpot_offloaded, 2),
            "Alpha (Current)": f"{self.alpha:.1%}",
            "Alpha_Star (Max Allowed)": f"{alpha_star:.1%}",
            "Insight": insight,
            "Strategy": strategy
        }

def find_latest_file(pattern):
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def derive_constants_from_results(baseline_dir, offload_dir):
    """
    Attempts to derive T and R from actual experiment results.
    Returns a dictionary with derived constants.
    """
    constants = {}
    
    # 1. Get N, T_prefill, T_decode from Baseline
    baseline_metrics = find_latest_file(os.path.join(baseline_dir, "metrics_*.csv"))
    if not baseline_metrics:
        # If no baseline stats found, return partial or empty
        print(f"Warning: No metrics CSV found in {baseline_dir}")
        return constants
    
    df_base = pd.read_csv(baseline_metrics)
    
    # Prefill Constants
    N = df_base['prompt_len'].mean()
    ttft_base = df_base['ttft'].mean() * 1e6 # s to us
    T_prefill = ttft_base / N if N > 0 else 0
    
    # Decode Constants
    # T_decode is roughly avg_itl
    T_decode = df_base['avg_itl'].mean() * 1e6 # s to us
    
    constants['N'] = int(N)
    constants['T_prefill'] = T_prefill
    constants['T_decode'] = T_decode
    
    # 2. Estimate R from Offload PCIe Stats (if available)
    # This is trickier. We need to look at disk_offload or cpu_offload stats.
    pcie_stats = find_latest_file(os.path.join(offload_dir, "pcie_stats_*.csv"))
    R_estimate = 50.0 # Default fallback (us)
    
    constants['R'] = R_estimate
    
    return constants

def run_scenario(name, N_or_L, R, T, alpha, mode="prefill"):
    print(f"--- {name} [{mode.upper()}] ---")
    
    if mode == "prefill":
        calc = PrefillSpeedupCalculator(N=N_or_L, R=R, T=T, alpha=alpha)
        result = calc.analyze_speedup()
        print(f"Inputs: N={N_or_L}, R={R:.1f}us, T={T:.1f}us, Alpha={alpha}")
        print(f"Regime: {result['Regime']}")
        print(f"Critical Hit Rate (α*): {result['Alpha_Star (Critical)']}")
        print(f"Speedup: {result['Actual']}x (Theoretical: {result['Theoretical']}x)")
        print(f"Strategy: {result['Strategy']}")
    else:
        calc = DecodeSpeedupCalculator(L=N_or_L, R=R, T=T, alpha=alpha)
        result = calc.analyze()
        print(f"Inputs: L={N_or_L}, R={R:.1f}us, T={T:.1f}us, Alpha={alpha}")
        print(f"Regime: {result['Regime']}")
        print(f"Max Offload (α*): {result['Alpha_Star (Max Allowed)']}")
        print(f"Impact: {result['Actual']} (TPOT: {result['TPOT_Offloaded (us)']}us vs {result['TPOT_Vanilla (us)']}us)")
        print(f"Strategy: {result['Strategy']}")
    
    print("")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottleneck Calculator for vLLM LMCache (Prefill & Decode)")
    parser.add_argument("--mode", type=str, choices=["prefill", "decode", "both"], default="both", help="Analysis mode")
    parser.add_argument("--N", type=int, help="Number of prompt tokens (Prefill) / Context Length (Decode)")
    parser.add_argument("--L", type=int, help="Context Length (Alias for N in Decode mode)")
    parser.add_argument("--R", type=float, help="Retrieval time per token (us)")
    parser.add_argument("--T", type=float, help="Compute time per token (us)")
    parser.add_argument("--alpha", type=float, help="Cache hit ratio (0.0 - 1.0)")
    
    parser.add_argument("--baseline-dir", type=str, help="Path to baseline experiment results dir (to auto-derive T)")
    parser.add_argument("--offload-dir", type=str, help="Path to offload experiment results dir (to auto-derive R)")
    
    args = parser.parse_args()
    
    # Handle alias
    val_N = args.N if args.N is not None else args.L

    # Default examples if no args provided and no derived source
    if not val_N and not args.R and not args.baseline_dir:
        print(">>> RUNNING DEFAULT SCENARIOS <<<\n")
        
        # Prefill Examples
        run_scenario("Scenario 1: A100 GPU (Compute Bound)", 2000, 41, 110, 0.60, "prefill")
        run_scenario("Scenario 2: H100 GPU (I/O Bound)", 2000, 50, 100, 0.90, "prefill")

        # Decode Examples
        run_scenario("Decode 1: Short Ctx, Fast SSD", 1000, 0.1, 200, 0.8, "decode")
        run_scenario("Decode 2: Long Ctx (I/O Bound)", 8000, 0.1, 200, 0.8, "decode")
        run_scenario("Decode 3: NVMe Reality Check", 4000, 0.5, 200, 0.5, "decode")

    else:
        # Use provided or derived args
        T = args.T
        R = args.R
        alpha = args.alpha if args.alpha is not None else 1.0 
        
        if args.baseline_dir:
            try:
                # Auto-derive defaults
                constants = derive_constants_from_results(args.baseline_dir, args.offload_dir if args.offload_dir else ".")
                
                if val_N is None: val_N = constants.get('N')
                if R is None: R = constants.get('R')
                
                # Pick T based on mode
                if T is None:
                    if args.mode == 'decode':
                        T = constants.get('T_decode')
                    else:
                        T = constants.get('T_prefill')

                print(f"[Auto-Derived] N={val_N}, T={T:.2f}us, R={R:.2f}us")

            except Exception as e:
                print(f"Error deriving constants: {e}")
                # Don't exit, fall through in case user provided args
        
        # Check inputs
        if None in [val_N, R, T]:
            print("Error: Must provide --N (or --L), --R, --T arguments or valid paths to results directories.")
            exit(1)
            
        if args.mode in ["prefill", "both"]:
            run_scenario("Custom Prefill Analysis", val_N, R, T, alpha, "prefill")
        
        if args.mode in ["decode", "both"]:
            run_scenario("Custom Decode Analysis", val_N, R, T, alpha, "decode")

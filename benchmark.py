import time
import asyncio
import argparse
import numpy as np
from openai import AsyncOpenAI

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
API_KEY = "EMPTY"
API_BASE = "http://localhost:8000/v1"

# Workload parameters
PROMPT_LEN = 1000  # Approx tokens
GEN_LEN = 100
NUM_REQUESTS = 10

# Synthetic prompt generator
def generate_prompt(length):
    # Create a long prompt with a shared prefix if needed
    # For this experiment, we want to test cache reuse, so we use a fixed prefix
    # and vary the suffix slightly or keep it identical to test exact match.
    # To test KV offloading benefit, we want repeated access to the same context.
    base_text = "The quick brown fox jumps over the lazy dog. " * (length // 10)
    return base_text

async def run_request(client, prompt, request_id, model, gen_len):
    start_time = time.time()
    ttft = 0
    token_times = []
    
    try:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=gen_len,
            stream=True,
            temperature=0, # Deterministic for cache testing
        )
        
        first_token = True
        async for chunk in stream:
            now = time.time()
            if first_token:
                ttft = now - start_time
                first_token = False
                token_times.append(now)
            else:
                token_times.append(now)
                
    except Exception as e:
        print(f"Request {request_id} failed: {e}")
        return None

    end_time = time.time()
    e2e_latency = end_time - start_time
    
    # Calculate ITL
    itls = []
    for i in range(1, len(token_times)):
        itls.append(token_times[i] - token_times[i-1])
    
    avg_itl = np.mean(itls) if itls else 0
    
    return {
        "request_id": request_id,
        "ttft": ttft,
        "e2e": e2e_latency,
        "avg_itl": avg_itl,
        "output_len": len(token_times)
    }

async def main(args):
    client = AsyncOpenAI(api_key=API_KEY, base_url=args.api_base)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    prompt = generate_prompt(args.prompt_len)
    total_tokens_per_req = args.prompt_len + args.gen_len
    total_workload_tokens = total_tokens_per_req * args.num_requests

    print(f"--- Experiment Configuration ---")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Label: {args.label}")
    print(f"Device: {args.device_name}")
    print(f"Model: {args.model}")
    print(f"KV Cache Dtype: {args.cache_dtype}")
    print(f"Workload: {args.num_requests} requests x {args.prompt_len + args.gen_len} tokens (Prompt: {args.prompt_len}, Gen: {args.gen_len})")
    print(f"Total Expected Tokens: {args.num_requests * (args.prompt_len + args.gen_len)}")
    print(f"Warmup: {args.warmup}")
    print(f"--------------------------------")

    tasks = []
    for i in range(args.num_requests):
        tasks.append(run_request(client, prompt, i, args.model, args.gen_len))
        await asyncio.sleep(0.5) 

    results = await asyncio.gather(*tasks)
    results = [r for r in results if r is not None]

    if not results:
        print("No successful requests.")
        return

    # Save detailed results to CSV
    safe_model = args.model.replace('/', '_')
    csv_filename = os.path.join(args.output_dir, f"metrics_{args.label}_{safe_model}_{args.prompt_len}_{args.gen_len}.csv")
    with open(csv_filename, "w") as f:
        f.write("request_id,label,prompt_len,gen_len,total_len,ttft,e2e,avg_itl\n")
        for r in results:
            total_len = args.prompt_len + r['output_len']
            f.write(f"{r['request_id']},{args.label},{args.prompt_len},{r['output_len']},{total_len},{r['ttft']:.6f},{r['e2e']:.6f},{r['avg_itl']:.6f}\n")
    print(f"Detailed metrics saved to {csv_filename}")

    avg_ttft = np.mean([r['ttft'] for r in results])
    avg_e2e = np.mean([r['e2e'] for r in results])
    avg_itl = np.mean([r['avg_itl'] for r in results])

    print("\n--- Results ---")
    print(f"Avg TTFT: {avg_ttft:.4f} s")
    print(f"Avg E2E:  {avg_e2e:.4f} s")
    print(f"Avg ITL:  {avg_itl:.4f} s")
    print("---------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", action="store_true", help="Run warmup requests")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN, help="Prompt length")
    parser.add_argument("--gen-len", type=int, default=GEN_LEN, help="Generation length")
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS, help="Number of requests")
    parser.add_argument("--api-base", type=str, default=API_BASE, help="API Base URL")
    parser.add_argument("--device-name", type=str, default="Unknown", help="Name of the GPU/Device")
    parser.add_argument("--cache-dtype", type=str, default="auto", help="KV Cache Data Type")
    parser.add_argument("--label", type=str, default="experiment", help="Label for the experiment")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save results")
    args = parser.parse_args()

    asyncio.run(main(args))

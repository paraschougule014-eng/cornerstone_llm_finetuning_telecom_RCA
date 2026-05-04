"""
Inference script for phase_1_test.csv
Model: Paras014/qwen35_2b_finetune_16bit_final

Converted from HuggingFace batched generate to vLLM for maximum H100 throughput.
vLLM uses PagedAttention + continuous batching internally — no manual batch loop needed.

Output CSV columns: ID | question | predicted | full_response

Usage:
    python run_inference_vllm.py
        --test_csv  phase_1_test.csv
        --truth_csv phase_1_test_truth.csv
        --output    predictions.csv
        --enable_thinking

Install:
    pip install vllm pandas tqdm transformers accelerate huggingface_hub
"""

import argparse
import re
import time

import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction  (unchanged from original)
# ──────────────────────────────────────────────────────────────────────────────
_BOXED_RE = re.compile(
    r'(?:\\boxed|\\oxed|oxed)\s*\{([Cc][1-8])\}',
    re.IGNORECASE,
)
_BRACE_RE  = re.compile(r'\{([Cc][1-8])\}', re.IGNORECASE)
_SIGNAL_RE = re.compile(
    r'(?:answer|therefore|conclusion|most likely|root cause'
    r'|is\s+(?:option|cause)?|select(?:ed)?|result is)'
    r'[^\n]{0,80}?\b([Cc][1-8])\b',
    re.IGNORECASE,
)

def extract_class(text: str) -> str:
    """Return the LAST predicted class (C1-C8) found in text, or 'UNKNOWN'."""
    hits = _BOXED_RE.findall(text)
    if hits:
        return hits[-1].upper()
    hits = _BRACE_RE.findall(text)
    if hits:
        return hits[-1].upper()
    hits = _SIGNAL_RE.findall(text)
    if hits:
        return hits[-1].upper()
    return "UNKNOWN"


# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Everything that runs at startup MUST be inside this guard.
# vLLM uses multiprocessing with the 'spawn' method. Without this guard,
# Python tries to re-run the entire script in each worker process, which
# causes the "bootstrapping phase" RuntimeError you saw.
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id",        default="Paras014/qwen35_2b_finetune_16bit_final")
    parser.add_argument("--test_csv",        default="phase_1_test.csv")
    parser.add_argument("--truth_csv",       default="phase_1_test_truth.csv",
                        help="Optional ground-truth CSV for accuracy evaluation")
    parser.add_argument("--output",          default="predictions.csv")
    parser.add_argument("--max_new_tokens",  type=int, default=4096,
                        help="Keep small for classification. "
                             "Use 2000+ only with --enable_thinking.")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable Qwen3 chain-of-thought. "
                             "Slower but may improve accuracy. "
                             "Requires --max_new_tokens 2000+.")
    parser.add_argument("--dtype", default="bfloat16",
                        help="Model dtype: float16 (T4/V100), bfloat16 (A100/H100 only). "
                             "T4 requires float16. A100/H100 can use bfloat16.")
    parser.add_argument("--tensor_parallel", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    args = parser.parse_args()

    # ── Sanity warning ────────────────────────────────────────────────────────
    if args.enable_thinking and args.max_new_tokens < 1000:
        print(
            f"[WARN] --enable_thinking is ON but --max_new_tokens={args.max_new_tokens}. "
            "Thinking traces are 500-2000 tokens. Model may be cut off before its final answer. "
            "Consider --max_new_tokens 2000 or drop --enable_thinking for pure classification."
        )

    print(f"[INFO] Thinking mode  : {'ON' if args.enable_thinking else 'OFF (fast classification mode)'}")
    print(f"[INFO] Max new tokens : {args.max_new_tokens}")
    print(f"[INFO] Tensor parallel: {args.tensor_parallel} GPU(s)\n")

    # ── Load test data ────────────────────────────────────────────────────────
    test_df = pd.read_csv(args.test_csv)
    print(f"[INFO] Loaded {len(test_df)} samples from {args.test_csv}")

    # ── Load model via vLLM ───────────────────────────────────────────────────
    print(f"\n[INFO] Loading model via vLLM: {args.model_id}")

    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=6500,
        gpu_memory_utilization=0.90,
        enforce_eager=False,              # False = use CUDA graphs (faster)
    )

    print("[INFO] Model loaded.\n")

    # ── Build prompts using chat template ─────────────────────────────────────
    print("[INFO] Applying chat template to all prompts...")

    tokenizer = llm.get_tokenizer()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    rows    = test_df.to_dict("records")
    prompts = []

    for row in tqdm(rows, desc="Templating"):
        messages = [{"role": "user", "content": row["question"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        prompts.append(prompt_text)

    print(f"[INFO] {len(prompts)} prompts ready.\n")

    # ── Sampling parameters ───────────────────────────────────────────────────
    sampling_params = SamplingParams(
        temperature=0.7,                    # greedy decoding
        max_tokens=args.max_new_tokens,
        skip_special_tokens=False,        # keep <think> tags visible if needed
        repetition_penalty=1.1,      # ✅ extra safety net against loops
        stop=["<|im_end|>", "<|endoftext|>"],  # ✅ hard stop strings for Qwen
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    print("[INFO] Running inference (vLLM continuous batching)...")
    total_start = time.time()

    outputs = llm.generate(prompts, sampling_params)

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] Inference done in {total_elapsed:.1f}s  "
          f"({total_elapsed / len(rows):.3f}s/sample avg)  "
          f"({len(rows) / total_elapsed:.1f} samples/s)")

    # ── Parse outputs ─────────────────────────────────────────────────────────
    results       = []
    unknown_count = 0

    for row, output in zip(rows, outputs):
        full_response = output.outputs[0].text
        predicted     = extract_class(full_response)

        if predicted == "UNKNOWN":
            unknown_count += 1

        results.append({
            "ID":            row["ID"],
            "question":      row["question"],
            "predicted":     predicted,
            "full_response": full_response,
        })

    print(f"[INFO] UNKNOWN predictions: {unknown_count}/{len(results)}")

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame(results)
    pred_df.to_csv(args.output, index=False)
    print(f"[INFO] Predictions saved → {args.output}")

    # ── Optional accuracy evaluation ──────────────────────────────────────────
    if args.truth_csv:
        try:
            truth_df = pd.read_csv(args.truth_csv)

            truth_df["base_ID"] = truth_df["ID"].str.rsplit("_", n=1).str[0]
            truth_dedup = (
                truth_df
                .sort_values("ID")
                .drop_duplicates(subset="base_ID", keep="first")
                [["base_ID", "Qwen3-32B"]]
                .rename(columns={"base_ID": "ID", "Qwen3-32B": "ground_truth"})
            )

            eval_df = pred_df[["ID", "predicted"]].merge(truth_dedup, on="ID", how="inner")
            correct = (eval_df["predicted"] == eval_df["ground_truth"]).sum()
            total   = len(eval_df)
            print(f"\n[EVAL] Accuracy : {correct}/{total} = {correct / total * 100:.2f}%")
            print(f"[EVAL] UNKNOWN  : {unknown_count}")

            print("\n[EVAL] Per-class accuracy:")
            for cls in sorted(eval_df["ground_truth"].unique()):
                sub = eval_df[eval_df["ground_truth"] == cls]
                acc = (sub["predicted"] == sub["ground_truth"]).mean() * 100
                print(f"  {cls}: {acc:.1f}%  (n={len(sub)})")

        except FileNotFoundError:
            print(f"[WARN] Truth CSV not found: {args.truth_csv} — skipping eval.")
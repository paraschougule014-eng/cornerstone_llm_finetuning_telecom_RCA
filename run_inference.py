"""
Inference script for phase_1_test.csv
Model: Paras014/qwen35_2b_finetune_16bit_final

Optimized for A100 GPU with batched inference for maximum throughput.
Output CSV columns: ID | question | predicted | full_response

Usage:
    python run_inference.py \
        --test_csv  phase_1_test.csv \
        --truth_csv phase_1_test_truth.csv \   # optional, for accuracy eval
        --output    predictions.csv \
        --batch_size 16                         # tune to your VRAM
"""

import argparse
import re
import time

import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# CLI args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model_id",       default="Paras014/qwen35_2b_finetune_16bit_final")
parser.add_argument("--test_csv",       default="phase_1_test.csv")
parser.add_argument("--truth_csv",      default="phase_1_test_truth.csv",
                    help="Optional ground-truth CSV for accuracy evaluation")
parser.add_argument("--output",         default="predictions.csv")
parser.add_argument("--max_new_tokens", type=int, default=5000)
parser.add_argument("--batch_size",     type=int, default=16,
                    help="Number of samples per batch. Tune based on VRAM. "
                         "A100 80 GB can usually handle 16–32 for a 2B model.")
args = parser.parse_args()

assert torch.cuda.is_available(), "CUDA GPU not found. This script requires a GPU."
DEVICE = "cuda"
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n[INFO] Loading model: {args.model_id}")

processor = AutoProcessor.from_pretrained(args.model_id)

# Use bfloat16 on A100 — numerically more stable than float16, same speed.
# sdpa (Scaled Dot-Product Attention) uses torch's fused kernels — fast and universally supported.
model = AutoModelForImageTextToText.from_pretrained(
    args.model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)
model.eval()

# Compile the model for extra throughput (PyTorch 2.x only; safe to remove if
# you're on an older PyTorch or if the model doesn't support torch.compile).
try:
    # model = torch.compile(model)
    print("[INFO] torch.compile() applied.")
except Exception as e:
    print(f"[WARN] torch.compile() skipped: {e}")

# Make sure pad token is defined — required for batched generation.
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model.config.pad_token_id = processor.tokenizer.eos_token_id

print("[INFO] Model loaded.\n")

# ──────────────────────────────────────────────────────────────────────────────
# Answer extraction
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
    """Return the LAST predicted class (C1–C8) found in text, or 'UNKNOWN'."""
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
# Load test data
# ──────────────────────────────────────────────────────────────────────────────
test_df = pd.read_csv(args.test_csv)
print(f"[INFO] Loaded {len(test_df)} samples from {args.test_csv}")
print(f"[INFO] Batch size: {args.batch_size}")
print(f"[INFO] Total batches: {(len(test_df) + args.batch_size - 1) // args.batch_size}\n")

# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a padded batch of token tensors
# ──────────────────────────────────────────────────────────────────────────────
def build_batch(rows: list[dict]) -> dict:
    """
    Apply chat template to each row individually, then left-pad the batch so
    that all sequences share the same length (required for batched generate).
    Returns a dict of tensors already on DEVICE.
    """
    encoded_list = []
    for row in rows:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": row["question"]}],
            }
        ]
        enc = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=True,
        )
        encoded_list.append(enc)

    # Left-pad to the longest sequence in this batch.
    max_len = max(e["input_ids"].shape[1] for e in encoded_list)
    pad_id  = processor.tokenizer.pad_token_id

    input_ids_padded      = []
    attention_mask_padded = []

    for enc in encoded_list:
        seq_len = enc["input_ids"].shape[1]
        pad_len = max_len - seq_len

        # Left-pad input_ids with pad_id
        padded_ids  = torch.cat(
            [torch.full((1, pad_len), pad_id, dtype=torch.long), enc["input_ids"]], dim=1
        )
        # Left-pad attention_mask with 0
        padded_mask = torch.cat(
            [torch.zeros(1, pad_len, dtype=torch.long), enc["attention_mask"]], dim=1
        )

        input_ids_padded.append(padded_ids)
        attention_mask_padded.append(padded_mask)

    return {
        "input_ids":      torch.cat(input_ids_padded,      dim=0).to(DEVICE),
        "attention_mask": torch.cat(attention_mask_padded, dim=0).to(DEVICE),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Batched inference loop
# ──────────────────────────────────────────────────────────────────────────────
results     = []
total_start = time.time()
rows        = test_df.to_dict("records")

n_batches   = (len(rows) + args.batch_size - 1) // args.batch_size

for batch_idx in tqdm(range(n_batches), desc="Batches"):
    batch_rows = rows[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]

    inputs     = build_batch(batch_rows)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            # Pad token needed for batched generation
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens for each item in the batch.
    new_ids = output_ids[:, prompt_len:]

    for i, row in enumerate(batch_rows):
        full_output = processor.tokenizer.decode(
            new_ids[i], skip_special_tokens=False
        )
        predicted = extract_class(full_output)

        results.append({
            "ID":            row["ID"],
            "question":      row["question"],
            "predicted":     predicted,
            "full_response": full_output,
        })

    # Progress log per batch
    done    = min((batch_idx + 1) * args.batch_size, len(rows))
    elapsed = time.time() - total_start
    rate    = done / elapsed
    eta     = (len(rows) - done) / rate if rate > 0 else 0
    tqdm.write(
        f"  Batch {batch_idx+1}/{n_batches} | "
        f"{done}/{len(rows)} samples | "
        f"{rate:.1f} samples/s | "
        f"ETA {eta:.0f}s"
    )

total_elapsed = time.time() - total_start
print(f"\n[INFO] Inference done in {total_elapsed:.1f}s  "
      f"({total_elapsed / len(rows):.2f}s/sample avg)  "
      f"({len(rows) / total_elapsed:.2f} samples/s)")

# ──────────────────────────────────────────────────────────────────────────────
# Save predictions
# ──────────────────────────────────────────────────────────────────────────────
pred_df = pd.DataFrame(results)
pred_df.to_csv(args.output, index=False)
print(f"[INFO] Predictions saved → {args.output}")

# ──────────────────────────────────────────────────────────────────────────────
# Optional accuracy evaluation
# ──────────────────────────────────────────────────────────────────────────────
if args.truth_csv:
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

    unknown = (eval_df["predicted"] == "UNKNOWN").sum()
    print(f"[EVAL] UNKNOWN  : {unknown}")

    print("\n[EVAL] Per-class accuracy:")
    for cls in sorted(eval_df["ground_truth"].unique()):
        sub = eval_df[eval_df["ground_truth"] == cls]
        acc = (sub["predicted"] == sub["ground_truth"]).mean() * 100
        print(f"  {cls}: {acc:.1f}%  (n={len(sub)})")

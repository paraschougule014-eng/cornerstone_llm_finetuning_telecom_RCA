import os
import re
import time
import argparse
import pandas as pd
from openai import OpenAI

LIGHTNING_API_KEY = ""


client = OpenAI(
    base_url="https://lightning.ai/api/v1",
    api_key=LIGHTNING_API_KEY,
)

# -----------------------
# Model Config
# -----------------------
MODEL_NAME = "lightning-ai/minimax-m2.5"

TEMPERATURE = 0.2
TOP_P = 0.9
MAX_TOKENS = 8192
RAW_RESPONSE_DIR = "api_responses"

# -----------------------
# Pricing (per million tokens)
# -----------------------
INPUT_COST_PER_M  = 0.25   # $0.25 per 1M input tokens
OUTPUT_COST_PER_M = 1.20   # $1.20 per 1M output tokens

def calc_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens / 1_000_000) * INPUT_COST_PER_M + \
           (completion_tokens / 1_000_000) * OUTPUT_COST_PER_M

# -----------------------
# Rate Limit Config
# -----------------------
SAFE_RPM = 15
DELAY = 60 / SAFE_RPM
DELAY = 0
# -----------------------
# FULL META PROMPT
# -----------------------
SYSTEM_PROMPT = """ou are an Expert Telecom RF Optimization Engineer. Your task is to generate a step-by-step, analytical reasoning trace for a 5G network Root Cause Analysis (RCA) dataset.

You will be provided with:
1. The problem description and rules for the drive-test scenario.
2. User plane drive test data (tabular format).
3. Engineering parameters data for the cells (tabular format).
4. The final correct ground-truth answer.

Your goal is to produce a highly detailed, logically rigorous reasoning process that:
- Proves exactly why the given answer is correct.
- Systematically eliminates all other plausible root causes.
- Uses explicit calculations and data references wherever applicable.

This reasoning will be used to train a smaller language model, so it must be:
- Step-by-step
- Explicit (no hidden assumptions)
- Numerically and logically precise
- Easy to follow and reproducible

------------------------------------------------------------

Structure your response EXACTLY as follows:

### 1. Problem Identification
- Identify the exact timestamps where the throughput drops below 600 Mbps.
- List the corresponding throughput values.
- Identify the Serving Cell PCI during those timestamps.

### 2. Hypothesis Testing

Evaluate each relevant root cause using the data. Show all calculations and reasoning explicitly.

C7: Speed Check (Speed > 40 km/h)
- Extract GPS Speed values during the degradation period.
- Determine whether any values exceed 40 km/h.
- State conclusion clearly.

C8: Resource Block Check (RBs < 160)
- Extract "Layer1 DL RB Num" values during degradation.
- Compute the average RB allocation.
- Compare against threshold (160).
- State conclusion.

C5: Handover Check (Frequent Handovers)
- Track Serving PCI changes across timestamps.
- Determine whether ping-pong or frequent handovers occur.
- Justify conclusion.

C6: Interference Check (PCI mod 30 Collision)
- Compute: Serving PCI mod 30.
- Compute: Neighbor PCI mod 30 for strongest neighbors.
- Check for equality indicating collision.
- State conclusion.

C2: Distance / Overshooting Check (>1 km)
- Extract UE latitude and longitude during degradation.
- Extract serving cell coordinates from engineering data.
- Compute:
  - Latitude difference → convert using 1° ≈ 111 km
  - Longitude difference → convert using 1° ≈ 94 km (at ~32° latitude)
- Calculate approximate distance:
  distance ≈ sqrt((Δlat_km)^2 + (Δlon_km)^2)
- Compare against 1 km threshold.
- State conclusion.

C3: RSRP/SINR Coverage Check
- Extract RSRP (dBm) and SINR (dB) values during degradation.
- Check if RSRP < -110 dBm (weak signal) or SINR < 0 dB (poor quality).
- State conclusion on coverage quality.

C4: Neighbor Cell Overshooting Check
- Compare neighbor cell RSRP values against serving cell RSRP.
- Check if any neighbor is within 6 dB of serving cell RSRP (strong interferer).
- Check if neighbor PCI mod 30 equals serving PCI mod 30 (collision).
- State conclusion on overshooting interference.

C1: Coverage / Downtilt Check
- Extract Mechanical Downtilt and Digital Tilt for serving cell.
- Compute total downtilt = Mechanical + Digital
- Evaluate: total downtilt > 12° is considered excessive for coverage
- Consider beam scenario if relevant.
- State conclusion.

### 3. Step-by-Step Deduction
- Synthesize all findings into a clear logical narrative.
- Explicitly eliminate each incorrect root cause with justification.
- Clearly explain why only one root cause remains valid.
- Reference RF conditions (RSRP, SINR) and spatial reasoning where applicable.

### 4. Final Answer
- Conclude with EXACTLY the following format (include the backslashes and braces literally):

The most likely root cause is \\boxed{CX}.

Where CX is one of: C1 (Downtilt), C2 (Distance/Overshooting), C3 (RSRP/SINR Coverage), C4 (Neighbor Overshooting), C5 (Handover), C6 (Interference), C7 (Speed), or C8 (Resource Blocks).

Example:
The most likely root cause is \\boxed{C2}.

IMPORTANT: The \\boxed{} LaTeX syntax is REQUIRED. Do NOT omit it.

------------------------------------------------------------

Important Rules:
- Do NOT skip any step.
- Do NOT assume conclusions without calculation.
- Always reference actual values from the dataset.
- Ensure the final answer matches the provided ground-truth label.
- ALWAYS use \\boxed{CX} format for the final answer.
"""

# -----------------------
# Prompt Builder
# -----------------------
def build_messages(question, answer):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem:\n{question}\n\nGround Truth Answer:\n{answer}\n\nGenerate the full reasoning trace:"},
    ]


def save_raw_response(record_num, response):
    os.makedirs(RAW_RESPONSE_DIR, exist_ok=True)
    response_path = os.path.join(RAW_RESPONSE_DIR, f"row_{record_num:06d}.json")
    with open(response_path, "w", encoding="utf-8") as f:
        f.write(response.model_dump_json(indent=2, exclude_none=True))

# -----------------------
# API Call with Retry
# -----------------------
def call_model(messages, record_num, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning_split": True},
            )
            save_raw_response(record_num, response)

            message = response.choices[0].message
            content = message.content or ""
            usage = response.usage

            # Extract thinking from reasoning_details
            reasoning_details = getattr(message, "reasoning_details", None)
            if reasoning_details:
                thinking = " ".join(
                    rd.get("text", "") for rd in reasoning_details if isinstance(rd, dict)
                )
            else:
                think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
                thinking = think_match.group(1).strip() if think_match else ""
                if think_match:
                    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            return content, thinking, usage.prompt_tokens, usage.completion_tokens

        except Exception as e:
            wait_time = (2 ** attempt) * 10
            print(f"  [retry {attempt+1}] {e} — sleeping {wait_time}s...")
            time.sleep(wait_time)

    # All retries failed (API/network error) — return empty to preserve row count
    return "", "", 0, 0

# -----------------------
# Main Processing
# -----------------------
def process_csv(input_csv, output_csv, start_idx, end_idx):
    df = pd.read_csv(input_csv)

    # -----------------------
    # AUTO-RESUME LOGIC
    # -----------------------
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_count = len(existing_df)
        print(f"Resuming: {processed_count} rows already processed")
        start_idx += processed_count
    else:
        processed_count = 0

    if end_idx is not None:
        df = df.iloc[start_idx:end_idx]
    else:
        df = df.iloc[start_idx:]

    total_rows = len(df)
    print(f"Processing {total_rows} rows starting from index {start_idx}")
    print("-" * 60)

    session_cost = 0.0
    total_cost = 0.0

    for i, (_, row) in enumerate(df.iterrows()):
        record_num = start_idx + i + 1

        question = row["question"]
        answer = row["answer"]

        messages = build_messages(question, answer)
        content, thinking, prompt_tokens, completion_tokens = call_model(messages, record_num)

        if content == "":
            print(f"Record {record_num:>4} | FAILED after all retries — saving empty row and continuing.")

        cost = calc_cost(prompt_tokens, completion_tokens)
        session_cost += cost
        total_cost += cost

        print(
            f"Record {record_num:>4} | answer={answer} | "
            f"in={prompt_tokens} out={completion_tokens} | "
            f"cost=${cost} | session total=${session_cost}"
        )

        pd.DataFrame([{
            "question": question,
            "answer": answer,
            "thinking": thinking,
            "reasoning": content,
        }]).to_csv(
            output_csv,
            mode="a",
            header=not os.path.exists(output_csv),
            index=False,
        )


    print("-" * 60)
    print(f"Done. Records this session: {i + 1} | Session cost: ${session_cost}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",  required=True)
    parser.add_argument("--output_csv", default="lightning_output.csv")
    parser.add_argument("--start_idx",  type=int, default=0)
    parser.add_argument("--end_idx",    type=int, default=None)
    args = parser.parse_args()

    process_csv(args.input_csv, args.output_csv, args.start_idx, args.end_idx)

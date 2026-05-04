"""
Qwen2.5-1.5B-Instruct fine-tuning for 5G telco troubleshooting classification.

Key improvement: pre-computes quantitative features for each of the 8 root-cause
categories from the raw drive-test data and engineering parameters, then appends
them to the prompt. This gives the model explicit signals instead of relying on
it to parse raw tabular numbers.

Usage (on a machine with an NVIDIA GPU):
    1. Create a virtual environment:
         python -m venv venv && source venv/bin/activate
    2. Install dependencies:
         pip install -r requirements_local.txt
    3. Place train.csv, phase_1_test.csv, phase_1_test_truth.csv in the same directory.
    4. Run:
         python qwen_train.py
    5. When done, deactivate and delete the venv to free space:
         deactivate && rm -rf venv
"""

import os
import re
import math
import argparse
from collections import Counter

import torch
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq


# ══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════

def _safe_float(val, default=None):
    if val is None or str(val).strip() in ("-", ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(min(1.0, math.sqrt(a)))


def _parse_pipe_table(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return [], []
    headers = [h.strip() for h in lines[0].split("|")]
    rows = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split("|")]
        if len(fields) == len(headers):
            rows.append(dict(zip(headers, fields)))
    return headers, rows


def _get_vertical_beamwidth(beam_scenario, bw_rules=None):
    if bw_rules is None:
        bw_rules = {}
    low_max = bw_rules.get("low_max", 5)
    mid_max = bw_rules.get("mid_max", 11)
    low_bw  = bw_rules.get("low_bw", 6)
    mid_bw  = bw_rules.get("mid_bw", 12)
    high_bw = bw_rules.get("high_bw", 25)

    if not beam_scenario or beam_scenario.upper() == "DEFAULT":
        return low_bw
    m = re.search(r"(\d+)", beam_scenario)
    if not m:
        return low_bw
    num = int(m.group(1))
    if num <= low_max:
        return low_bw
    elif num <= mid_max:
        return mid_bw
    else:
        return high_bw


def extract_base_stats(question_text):
    stats = {
        "digital_tilt_default_code": 255,
        "digital_tilt_default_angle": 6,
        "bw_rules": {
            "low_max": 5, "mid_max": 11, "low_bw": 6, "mid_bw": 12, "high_bw": 25,
        },
    }
    m = re.search(r"default\s+electronic\s+downtilt\s+value\s+is\s+(\d+).*?downtilt\s+angle\s+of\s+(\d+)\s+degrees", question_text, re.IGNORECASE | re.DOTALL)
    if m:
        stats["digital_tilt_default_code"] = int(m.group(1))
        stats["digital_tilt_default_angle"] = int(m.group(2))

    bw_patterns = re.findall(r"SCENARIO_(\d+)\s*(?:to|[-–])\s*SCENARIO_(\d+).*?vertical\s+beamwidth\s+is\s+(\d+)\s+degrees", question_text, re.IGNORECASE)
    if len(bw_patterns) >= 2:
        stats["bw_rules"]["low_max"] = int(bw_patterns[0][1])
        stats["bw_rules"]["low_bw"] = int(bw_patterns[0][2])
        stats["bw_rules"]["mid_max"] = int(bw_patterns[1][1])
        stats["bw_rules"]["mid_bw"] = int(bw_patterns[1][2])

    m_high = re.search(r"SCENARIO_(\d+)\s+or\s+above.*?vertical\s+beamwidth\s+is\s+(\d+)\s+degrees", question_text, re.IGNORECASE)
    if m_high:
        stats["bw_rules"]["high_bw"] = int(m_high.group(2))

    m_default_low = re.search(r"Default\s+or\s+SCENARIO_\d+\s*(?:to|[-–])\s*SCENARIO_(\d+).*?vertical\s+beamwidth\s+is\s+(\d+)\s+degrees", question_text, re.IGNORECASE)
    if m_default_low:
        stats["bw_rules"]["low_max"] = int(m_default_low.group(1))
        stats["bw_rules"]["low_bw"] = int(m_default_low.group(2))

    return stats


def compute_features(question_text):
    """
    Parse drive-test data and engineering params from the question text,
    compute quantitative checks for each of the 8 root-cause categories,
    and return a concise analysis summary string.
    """
    dt_match = re.search(r"User plane drive test data as follows[：:]?\s*\n(.*?)(?:\n\s*\n\s*Eng|\nEng)", question_text, re.DOTALL)
    eng_match = re.search(r"Eng[ei]neering parameters data as follows[：:]?\s*\n(.*)$", question_text, re.DOTALL)
    if not dt_match or not eng_match:
        return ""

    _, dt_rows = _parse_pipe_table(dt_match.group(1))
    _, eng_rows = _parse_pipe_table(eng_match.group(1))
    if not dt_rows or not eng_rows:
        return ""

    base = extract_base_stats(question_text)
    dt_default_code = base["digital_tilt_default_code"]
    dt_default_angle = base["digital_tilt_default_angle"]
    bw_rules = base["bw_rules"]

    pci_to_eng = {}
    pci_to_gnb = {}
    for e in eng_rows:
        pci = _safe_float(e.get("PCI"))
        if pci is not None:
            pci_to_eng[int(pci)] = e
            pci_to_gnb[int(pci)] = e.get("gNodeB ID", "")

    speeds, rsrps, sinrs, tputs, rbs = [], [], [], [], []
    serving_pcis = []
    ue_coords = []
    for r in dt_rows:
        speeds.append(_safe_float(r.get("GPS Speed (km/h)")))
        rsrps.append(_safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]")))
        sinrs.append(_safe_float(r.get("5G KPI PCell RF Serving SS-SINR [dB]")))
        tputs.append(_safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]")))
        rbs.append(_safe_float(r.get("5G KPI PCell Layer1 DL RB Num (Including 0)")))
        serving_pcis.append(_safe_float(r.get("5G KPI PCell RF Serving PCI")))
        lat = _safe_float(r.get("Latitude"))
        lon = _safe_float(r.get("Longitude"))
        ue_coords.append((lat, lon))

    speeds  = [v for v in speeds  if v is not None]
    rsrps   = [v for v in rsrps   if v is not None]
    sinrs   = [v for v in sinrs   if v is not None]
    tputs   = [v for v in tputs   if v is not None]
    rbs     = [v for v in rbs     if v is not None]
    srv_pcis = [int(v) for v in serving_pcis if v is not None]

    n = len(dt_rows)
    features = []

    avg_rsrp = sum(rsrps) / len(rsrps) if rsrps else 0
    avg_sinr = sum(sinrs) / len(sinrs) if sinrs else 0
    avg_tput = sum(tputs) / len(tputs) if tputs else 0
    low_tput = sum(1 for t in tputs if t < 600)
    features.append(
        f"Signal: avg RSRP={avg_rsrp:.1f}dBm, avg SINR={avg_sinr:.1f}dB, "
        f"avg throughput={avg_tput:.0f}Mbps, low-throughput samples(<600Mbps)={low_tput}/{len(tputs)}"
    )

    all_serving_pcis = set(srv_pcis)
    
    # C1: downtilt analysis for all serving PCIs
    c1_details = []
    for spci in all_serving_pcis:
        if spci in pci_to_eng:
            cell = pci_to_eng[spci]
            mech_dt = _safe_float(cell.get("Mechanical Downtilt"), 0)
            digi_tilt = _safe_float(cell.get("Digital Tilt"), 0)
            if digi_tilt == dt_default_code:
                digi_tilt = dt_default_angle
            total_dt = mech_dt + digi_tilt
            beam = cell.get("Beam Scenario", "DEFAULT")
            vbw = _get_vertical_beamwidth(beam, bw_rules)
            excess = total_dt - vbw
            if excess > 0:
                c1_details.append(f"PCI {spci}: total_dt={total_dt}deg (mech={mech_dt}, digi={digi_tilt}), beam={beam}, vbw={vbw}deg -> LARGE (excess {excess}deg)")
    
    if c1_details:
        features.append("C1: " + " | ".join(c1_details))
    else:
        features.append("C1: served by normal downtilt cells (total downtilt <= beamwidth)")

    # C2: coverage distance for all serving cells
    max_dist_any_serving = 0
    for spci in all_serving_pcis:
        if spci in pci_to_eng:
            cell = pci_to_eng[spci]
            clat = _safe_float(cell.get("Latitude"))
            clon = _safe_float(cell.get("Longitude"))
            if clat and clon:
                for lat, lon in ue_coords:
                    if lat and lon:
                        d = _haversine_km(lat, lon, clat, clon) * 1000
                        max_dist_any_serving = max(max_dist_any_serving, d)
    features.append(
        f"C2: UE-to-any-serving-cell max distance={max_dist_any_serving:.0f}m -> {'EXCEEDS 1km' if max_dist_any_serving > 1000 else 'within 1km'}"
    )

    # C3: neighbor provides higher signal + throughput/co-located neighbors info
    coloc_neighbor_count = 0
    non_coloc_neighbor_count = 0
    nbr_stronger = 0
    for r in dt_rows:
        srv = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        if srv is None or sp is None:
            continue
        srv_gnb = pci_to_gnb.get(int(sp), "")
        stronger_found = False
        for i in range(1, 6):
            npci = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"))
            nrsrp = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"))
            if npci is not None and int(npci) in pci_to_gnb:
                if pci_to_gnb[int(npci)] == srv_gnb:
                    coloc_neighbor_count += 1
                else:
                    non_coloc_neighbor_count += 1
            if nrsrp is not None and nrsrp > srv and not stronger_found:
                nbr_stronger += 1
                stronger_found = True

    total_nbr = coloc_neighbor_count + non_coloc_neighbor_count
    coloc_ratio = coloc_neighbor_count / total_nbr if total_nbr > 0 else 0
    
    coloc_handover_gain = 0
    non_coloc_handover_gain = 0
    valid_tp_rows = [(int(_safe_float(r.get("5G KPI PCell RF Serving PCI"))), _safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]"))) 
                     for r in dt_rows if _safe_float(r.get("5G KPI PCell RF Serving PCI")) is not None and _safe_float(r.get("5G KPI PCell Layer2 MAC DL Throughput [Mbps]")) is not None]
    
    for i in range(1, len(valid_tp_rows)):
        prev_pci, prev_tp = valid_tp_rows[i-1]
        curr_pci, curr_tp = valid_tp_rows[i]
        if prev_pci != curr_pci:
            start_before = max(0, i-2)
            before_tps = [x[1] for x in valid_tp_rows[start_before:i]]
            tput_before = sum(before_tps) / len(before_tps)
            end_after = min(len(valid_tp_rows), i+3)
            after_tps = [x[1] for x in valid_tp_rows[i:end_after]]
            tput_after = sum(after_tps) / len(after_tps)
            gain = tput_after - tput_before
            prev_gnb = pci_to_gnb.get(prev_pci, "A")
            curr_gnb = pci_to_gnb.get(curr_pci, "B")
            if prev_gnb == curr_gnb:
                coloc_handover_gain = max(coloc_handover_gain, gain)
            else:
                non_coloc_handover_gain = max(non_coloc_handover_gain, gain)

    features.append(
        f"C3: timestamps neighbor RSRP > serving = {nbr_stronger}/{n} | "
        f"co-located neighbor ratio={coloc_ratio:.0%} | "
        f"handover throughput gain (colocated={coloc_handover_gain:.0f}Mbps, non-colocated={non_coloc_handover_gain:.0f}Mbps)"
    )

    # C4: non-colocated co-frequency overlapping coverage
    non_coloc = 0
    for r in dt_rows:
        sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
        sr = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
        if sp is None or sr is None:
            continue
        srv_gnb = pci_to_gnb.get(int(sp), "")
        overlap = False
        for i in range(1, 6):
            npci = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"))
            nrsrp = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"))
            if npci is not None and nrsrp is not None:
                nbr_gnb = pci_to_gnb.get(int(npci), "")
                if nbr_gnb and nbr_gnb != srv_gnb and (sr - nrsrp) < 10:
                    overlap = True
                    break
        if overlap:
            non_coloc += 1
    features.append(
        f"C4: timestamps with non-colocated cell within 10dB of serving = {non_coloc}/{n} -> {'SEVERE OVERLAP' if non_coloc > n * 0.5 else 'low/moderate overlap'}"
    )

    # C5: frequent handovers
    ho_count = sum(1 for i in range(1, len(srv_pcis)) if srv_pcis[i] != srv_pcis[i - 1])
    features.append(f"C5: handovers={ho_count} in {n} samples -> {'FREQUENT' if ho_count >= 3 else 'not frequent'}")

    # C6: PCI mod 30 collision
    pci_mod30_collisions = []
    mod30_collision_strong_count = 0
    if srv_pcis:
        srv_mod30 = set(p % 30 for p in srv_pcis)
        all_nbr_pcis = set()
        for r in dt_rows:
            sp = _safe_float(r.get("5G KPI PCell RF Serving PCI"))
            sr = _safe_float(r.get("5G KPI PCell RF Serving SS-RSRP [dBm]"))
            if sp is None or sr is None:
                continue
            sp_mod30 = int(sp) % 30
            for i in range(1, 6):
                npci = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} PCI"))
                nrsrp = _safe_float(r.get(f"Measurement PCell Neighbor Cell Top Set(Cell Level) Top {i} Filtered Tx BRSRP [dBm]"))
                if npci is not None:
                    npci_val = int(npci)
                    all_nbr_pcis.add(npci_val)
                    if npci_val % 30 == sp_mod30 and npci_val not in all_serving_pcis:
                        if nrsrp is not None and sr is not None and (sr - nrsrp) < 10:
                            mod30_collision_strong_count += 1
                            break

        pci_mod30_collisions = sorted(p for p in all_nbr_pcis if p % 30 in srv_mod30 and p not in all_serving_pcis)
        
    serving_mod30_collision = False
    if len(all_serving_pcis) > 1:
        mods = [p % 30 for p in all_serving_pcis]
        if len(mods) != len(set(mods)):
            serving_mod30_collision = True

    features.append(
        f"C6: neighbor PCIs with same mod30 as serving = {pci_mod30_collisions if pci_mod30_collisions else 'NONE'}, "
        f"timestamps with strong colliding neighbor = {mod30_collision_strong_count}/{n}, "
        f"serving pci internal mod30 collision = {serving_mod30_collision}"
    )

    # C7: vehicle speed
    avg_spd = sum(speeds) / len(speeds) if speeds else 0
    max_spd = max(speeds) if speeds else 0
    features.append(f"C7: avg speed={avg_spd:.1f}km/h, max={max_spd:.1f}km/h -> {'EXCEEDS 40km/h' if max_spd > 40 else 'below 40km/h'}")

    # C8: average RBs
    avg_rb = sum(rbs) / len(rbs) if rbs else 0
    low_rb_count = sum(1 for r in rbs if r < 160)
    features.append(f"C8: avg scheduled RBs={avg_rb:.1f}, count < 160 = {low_rb_count}/{n} -> {'BELOW 160' if avg_rb < 160 else 'above 160'}")

    return "\n\nKey analysis:\n" + "\n".join(features)



# ══════════════════════════════════════════════════════════════
# CLI ARGUMENTS
# ══════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(
    description="Fine-tune Qwen2.5-1.5B-Instruct for telco classification"
)
parser.add_argument("--train_csv",  default="train.csv")
parser.add_argument("--test_csv",   default="phase_1_test.csv")
parser.add_argument("--truth_csv",  default="phase_1_test_truth.csv")
parser.add_argument("--epochs",     type=int,   default=3)
parser.add_argument("--lr",         type=float, default=2e-4)
parser.add_argument("--batch_size", type=int,   default=1)
parser.add_argument("--grad_accum", type=int,   default=8)
parser.add_argument("--lora_r",     type=int,   default=64)
parser.add_argument("--max_seq",    type=int,   default=4096)
parser.add_argument("--max_eval",   type=int,   default=None)
parser.add_argument("--skip_eval",  action="store_true")
parser.add_argument("--save_dir",   default="lora_model")
parser.add_argument("--output_dir", default="outputs")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
max_seq_length = args.max_seq
dtype = None
load_in_4bit = True

alpaca_prompt = """### Instruction:
{}

### Response:
{}"""


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load model & tokenizer
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading model")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# ══════════════════════════════════════════════════════════════
# STEP 2 — LoRA adapters
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Attaching LoRA adapters")
print("=" * 60)

model = FastLanguageModel.get_peft_model(
    model,
    r=args.lora_r,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=args.lora_r,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# ══════════════════════════════════════════════════════════════
# STEP 3 — Data preparation (with feature augmentation)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Preparing dataset (with feature extraction)")
print("=" * 60)

raw_dataset = load_dataset("csv", data_files=args.train_csv, split="train")
EOS_TOKEN = tokenizer.eos_token


def tokenize_and_mask(examples):
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    n_truncated = 0

    for q, a in zip(examples["question"], examples["answer"]):
        a = a.strip()
        if not a.startswith("\\boxed"):
            a = f"\\boxed{{{a}}}"

        q_augmented = q + compute_features(q)

        prompt_text = alpaca_prompt.format(q_augmented, "")
        answer_text = a + EOS_TOKEN

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        full_ids = prompt_ids + answer_ids

        if len(full_ids) > max_seq_length:
            n_truncated += 1
            head_budget = max_seq_length - len(answer_ids)
            full_ids = prompt_ids[:head_budget] + answer_ids

        answer_len = len(answer_ids)
        response_start = len(full_ids) - answer_len

        labels = [-100] * response_start + full_ids[response_start:]
        attention_mask = [1] * len(full_ids)

        out["input_ids"].append(full_ids)
        out["attention_mask"].append(attention_mask)
        out["labels"].append(labels)

    if n_truncated > 0:
        print(f"  [INFO] {n_truncated} samples truncated to {max_seq_length} tokens "
              f"(right-truncated, keeping head + answer)")

    return out


train_dataset = raw_dataset.map(
    tokenize_and_mask,
    batched=True,
    remove_columns=raw_dataset.column_names,
)
train_dataset = train_dataset.filter(lambda x: any(t != -100 for t in x["labels"]))

print(f"Train dataset size: {len(train_dataset)}")
assert len(train_dataset) > 0, "Train dataset is empty after preprocessing."

sample = train_dataset[0]
label_tokens = [t for t in sample["labels"] if t != -100]
print(f"Sample total tokens: {len(sample['input_ids'])}, answer tokens: {len(label_tokens)}")
print(f"Sample target: {tokenizer.decode(label_tokens, skip_special_tokens=False)}")
decoded_prompt = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
feat_start = decoded_prompt.find("Key analysis:")
if feat_start != -1:
    print(f"Sample features:\n{decoded_prompt[feat_start:feat_start+600]}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Train
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.03,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
    ),
)

print(f"Train dataset size: {len(trainer.train_dataset)}")
assert len(trainer.train_dataset) > 0, "train_dataset is empty before training."

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\nTraining completed. Peak GPU memory used: {used_memory} GB")
print(f"Training time: {trainer_stats.metrics['train_runtime']:.1f}s")


# ══════════════════════════════════════════════════════════════
# STEP 5 — Save LoRA adapters
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Saving LoRA adapters")
print("=" * 60)

model.save_pretrained(args.save_dir)
tokenizer.save_pretrained(args.save_dir)
print(f"Saved to {args.save_dir}/")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Evaluation
# ══════════════════════════════════════════════════════════════
if args.skip_eval:
    print("\nSkipping evaluation (--skip_eval flag).")
else:
    print("\n" + "=" * 60)
    print("STEP 6: Evaluating on test set")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    test_df = pd.read_csv(args.test_csv)
    truth_df = pd.read_csv(args.truth_csv)

    truth_cols = [c for c in truth_df.columns if c != "ID"]
    assert len(truth_cols) > 0, "No truth label column found"
    truth_col = truth_cols[0]

    if len(truth_cols) > 1:
        for c in truth_cols[1:]:
            mismatch = (
                truth_df[c].astype(str).str.upper()
                != truth_df[truth_col].astype(str).str.upper()
            ).sum()
            if mismatch:
                print(f"Warning: {mismatch} mismatches between {truth_col} and {c}")

    truth_df = truth_df.copy()
    truth_df["base_id"] = truth_df["ID"].astype(str).str.replace(
        r"_[0-9]+$", "", regex=True
    )
    truth_base = truth_df.groupby("base_id", as_index=False)[truth_col].first()

    merged = test_df.merge(truth_base, left_on="ID", right_on="base_id", how="inner")
    merged["gold"] = (
        merged[truth_col].astype(str).str.upper().str.extract(r"(C[1-8])", expand=False)
    )

    print(f"Test rows: {len(test_df)} | Truth rows: {len(truth_df)} | "
          f"Merged rows: {len(merged)}")
    assert len(merged) > 0, "Merge produced 0 rows."

    def extract_label(text):
        if not isinstance(text, str):
            return None
        m = re.search(r"\\boxed\s*\{\s*([Cc][1-8])\s*\}", text)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([Cc][1-8])\b", text)
        if m:
            return m.group(1).upper()
        return None

    allowed_texts = [f"\\boxed{{C{i}}}" for i in range(1, 9)]
    allowed_token_seqs = [
        tokenizer.encode(t, add_special_tokens=False) for t in allowed_texts
    ]
    eos_id = tokenizer.eos_token_id

    @torch.no_grad()
    def predict_label(question_text):
        q_aug = question_text + compute_features(question_text)
        inputs = tokenizer(
            [alpaca_prompt.format(q_aug, "")],
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
        ).to("cuda")

        prompt_len = inputs["input_ids"].shape[1]

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated = input_ids[prompt_len:].tolist()
            allowed_next = set()
            for seq in allowed_token_seqs:
                if len(generated) < len(seq) and generated == seq[: len(generated)]:
                    allowed_next.add(seq[len(generated)])
                elif len(generated) == len(seq):
                    allowed_next.add(eos_id)
            return list(allowed_next) if allowed_next else [eos_id]

        outputs = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=max(len(x) for x in allowed_token_seqs) + 1,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        generated_ids = outputs[0, prompt_len:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        pred = extract_label(raw)
        return pred, raw

    eval_df = merged if args.max_eval is None else merged.head(args.max_eval).copy()

    preds, raws = [], []
    for q in tqdm(eval_df["question"].tolist(), total=len(eval_df), desc="Evaluating"):
        p, r = predict_label(q)
        preds.append(p)
        raws.append(r)

    eval_df = eval_df.copy()
    eval_df["pred"] = preds
    eval_df["raw_output"] = raws

    valid_mask = eval_df["gold"].notna() & eval_df["pred"].notna()
    accuracy = (
        (eval_df.loc[valid_mask, "pred"] == eval_df.loc[valid_mask, "gold"]).mean()
        if valid_mask.any() else 0.0
    )

    print(f"\nRows evaluated: {len(eval_df)}")
    print(f"Rows with valid gold+pred labels: {int(valid_mask.sum())}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    mistakes = eval_df[valid_mask & (eval_df["pred"] != eval_df["gold"])][
        ["ID", "gold", "pred", "raw_output"]
    ]
    print(f"Mistakes: {len(mistakes)}")
    if len(mistakes) > 0:
        print(mistakes.head(10).to_string(index=False))


print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

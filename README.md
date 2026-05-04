# Telecom RCA with Tiny LLMs

Fine-tuning and evaluating small language models for **5G telecom root cause analysis (RCA)** on drive-test and engineering-parameter data.

This repo contains:
- a **rule-based baseline**
- a **feature-augmented fine-tuning pipeline**
- **Qwen 3.5** reasoning-based fine-tuning
- **Gemma 4** reasoning-based fine-tuning
- an **inference notebook** for batch evaluation

The task is an **8-class classification problem**. Given a telecom troubleshooting prompt containing drive-test data and engineering parameters, the model must predict the most likely reason for throughput dropping below 600 Mbps.

## Root Cause Classes

- `C1`: Large downtilt causing weak coverage
- `C2`: Overshooting due to coverage distance above 1 km
- `C3`: Neighboring cell provides higher throughput
- `C4`: Non-colocated neighboring cells create overlapping coverage
- `C5`: Frequent handovers
- `C6`: PCI mod 30 collision interference
- `C7`: High vehicle speed
- `C8`: Low available resource blocks

## Repository Files

### 1. `qwen_train_new_rules.py`
Rule-based telecom troubleshooting classifier.

What it does:
- Parses telecom tables from the question text
- Extracts numerical features using Python
- Applies hard rules and heuristic scoring
- Predicts one of `C1` to `C8`

Why it exists:
- Serves as a non-LLM baseline
- Encodes telecom logic explicitly
- Helps show how much of the task can be solved with engineered reasoning

Main logic:
- Hard rules first for easier classes like speed, RB shortage, distance, and handovers
- Scoring stage for harder classes like downtilt, overlap, neighbor preference, and PCI collision

### 2. `qwen_train.py`
Feature-augmented fine-tuning script for `unsloth/Qwen2.5-1.5B-Instruct`.

What it does:
- Parses the telecom prompt
- Computes class-specific features from the tables
- Appends a `"Key analysis:"` block to the original prompt
- Fine-tunes Qwen with LoRA
- Evaluates on the test set using constrained decoding

Why it exists:
- The raw task is numerically heavy
- Instead of forcing the model to parse all tables itself, the script adds structured evidence to the prompt

### 3. `Qwen_3_5_2B_Finetune (1).ipynb`
Notebook for reasoning-based fine-tuning of `unsloth/Qwen3.5-2B`.

What it does:
- Loads a reasoning dataset with columns:
  - `question`
  - `thinking`
  - `reasoning`
- Formats the assistant response as:
  - `<think> ... </think>` for chain-of-thought
  - followed by the final reasoning/answer
- Fine-tunes Qwen 3.5 using LoRA
- Saves LoRA adapters and merged checkpoints
- Pushes final models to Hugging Face

Notes:
- Designed for an A100 Colab setup
- Uses 16-bit LoRA instead of QLoRA
- Saves merged 16-bit, merged 4-bit, and LoRA-only variants

### 4. `Gemma4_(E2B)_Text_Training.ipynb`
Notebook for reasoning-based fine-tuning of `unsloth/gemma-4-E2B-it`.

What it does:
- Loads the same reasoning-style dataset
- Converts rows into chat-style conversations
- Fine-tunes Gemma 4 text layers using LoRA
- Saves local checkpoints and uploads merged models to Hugging Face

Notes:
- Configured for text-only fine-tuning
- Uses LoRA on language, attention, and MLP modules

### 5. `Inference_5000_tokens.ipynb`
Inference/evaluation notebook for running a trained model on the test set.

What it does:
- Installs inference dependencies
- Runs batch inference on `phase_1_test.csv`
- Evaluates predictions against `phase_1_test_truth.csv`
- Saves output predictions to CSV

Typical usage:
```bash
python run_inference.py \
    --test_csv phase_1_test.csv \
    --truth_csv phase_1_test_truth.csv \
    --output predictions.csv \
    --batch_size 32
```

## Project Workflow

This project was explored in multiple stages:

1. **Direct fine-tuning on labels only**
   - Input: raw telecom question
   - Output: class label
   - Result: poor performance, close to random guessing

2. **Rule-based baseline**
   - Extract telecom values explicitly
   - Apply domain rules
   - Much stronger than pure label-only fine-tuning

3. **Rule-based feature augmentation + LLM**
   - Convert extracted features into short textual summaries
   - Append them to the prompt
   - Fine-tune a small LLM on this richer input

4. **Reasoning-based fine-tuning**
   - Build a dataset with question, chain-of-thought, and final answer
   - Train Qwen 3.5 and Gemma 4 to reason before answering

## Data Format

The code expects telecom RCA samples in CSV form.

Typical files:
- `train.csv`
- `phase_1_test.csv`
- `phase_1_test_truth.csv`
- `lightning_output_combined.csv`

### `train.csv`
Expected columns:
- `question`
- `answer`

### `phase_1_test.csv`
Expected columns:
- `ID`
- `question`

### `phase_1_test_truth.csv`
Expected columns:
- `ID`
- one or more ground-truth label columns

### `lightning_output_combined.csv`
Expected columns:
- `question`
- `thinking`
- `reasoning`

This reasoning dataset is used for Qwen 3.5 and Gemma 4 fine-tuning.

## Setup

### Python scripts

Recommended environment:
- Python 3.10+
- NVIDIA GPU for fine-tuning

Install the main dependencies used across the scripts:

```bash
pip install torch pandas tqdm datasets transformers trl peft accelerate unsloth
```

Some notebooks/scripts may also require:

```bash
pip install bitsandbytes xformers huggingface_hub sentencepiece protobuf
```

### Notebook environment

The notebooks were designed for Google Colab:
- **Qwen 3.5 notebook**: A100 recommended
- **Gemma 4 notebook**: Colab GPU setup

## How to Run

### Rule-based baseline

```bash
python qwen_train_new_rules.py \
    --test_csv phase_1_test.csv \
    --truth_csv phase_1_test_truth.csv
```

### Feature-augmented Qwen 2.5 fine-tuning

```bash
python qwen_train.py \
    --train_csv train.csv \
    --test_csv phase_1_test.csv \
    --truth_csv phase_1_test_truth.csv
```

### Qwen 3.5 reasoning fine-tuning

Open:
- `Qwen_3_5_2B_Finetune (1).ipynb`

Upload or mount:
- `lightning_output_combined.csv`

Then run all notebook cells.

### Gemma 4 reasoning fine-tuning

Open:
- `Gemma4_(E2B)_Text_Training.ipynb`

Upload or mount:
- `lightning_output_combined.csv`

Then run all notebook cells.

### Inference

Open:
- `Inference_5000_tokens.ipynb`

Or run the inference script it calls after installing dependencies.

## Model Outputs

The training notebooks save multiple formats:
- LoRA adapters
- merged 16-bit checkpoints
- merged 4-bit checkpoints
- optional GGUF export in the Gemma notebook

Examples from the notebooks:
- `Paras014/qwen35_2b_finetune_16bit_final` (Main)
- `Paras014/gemma-4-finetune-final` (Main)
- `Paras014/qwen35_2b_finetune_4bit_final`
- `Paras014/qwen35_2b_lora_final`
- `Paras014/gemma_4_lora_final`


## Key Idea

The main lesson from this work is that **tiny LLMs struggle on raw telecom RCA prompts unless the reasoning is made more explicit**.

This repo explores three ways to improve that:
- **hand-crafted telecom rules**
- **prompt augmentation with computed features**
- **fine-tuning on reasoning traces**

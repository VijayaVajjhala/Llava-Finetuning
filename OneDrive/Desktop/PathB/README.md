# LLaVA-1.5-7b QLoRA Fine-Tuning — PathB

## Overview

Fine-tuned `liuhaotian/llava-v1.5-7b` on 20 IAM handwriting image-instruction pairs
using LoRA (r=128) on a single NVIDIA A100-SXM4-40GB GPU.

This README covers two experiments:

* **Part A:** 20-sample IAM handwriting fine-tune (OCR task)
* **Part B2:** 1-epoch fine-tune on LLaVA repo sample set (\~2,500 pairs) with LLaVA-Bench evaluation

**Date:** 2026-03-12  
**Task:** Handwritten text transcription (OCR)  
**Dataset:** IAM Handwriting (`Teklia/IAM-line`, 20 samples)

\---

## Directory Structure

```
PathB/
├─ instructions_20.json                   # 20 image-instruction pairs (Part A)
├─ generated_20.json                      # Model predictions vs ground truth (Part A)

├─ Homework5_LLaVA_LoRA_Finetune.ipynb    # Notebook with code execution for B1 and B2
├─ B2/
│   ├─ llava_bench_answers.jsonl          # B2 model answers on LLaVA-Bench-in-the-Wild
│   ├─ llava_bench_scores.jsonl           # GPT-4 evaluation scores (raw)
│   |─ benchmark_summary.json             # Final benchmark summary

├   └─ llava_2500.json                    # 2,500-pair subset used for B2 training
├─ full_run_logs/
│   ├─ gpu_info.txt                       # nvidia-smi output
│   ├─ training_log.json                  # Loss curve + hyperparameters
│   └─ training_command.sh                # deepspeed command used
│   └─ train_log_B2.txt                   # Finetuning logs
└─ README.md                              # This file
```

\---

## GPU Specs

|Property|Value|
|-|-|
|GPU|NVIDIA A100-SXM4-40GB|
|VRAM|40,960 MiB|
|CUDA Version|12.8|
|Driver Version|580.82.07|
|Form Factor|SXM4 (high-bandwidth)|

\---

## Part A — IAM Handwriting Fine-Tune

### Training Command

```bash
deepspeed /content/LLaVA/llava/train/train_mem.py \
    --lora_enable True \\
    --lora_r 128 \\
    --lora_alpha 256 \\
    --deepspeed /content/LLaVA/scripts/zero2.json \\
    --model_name_or_path liuhaotian/llava-v1.5-7b \\
    --version v1 \\
    --data_path /content/drive/MyDrive/TCSS590/Llava/data/train.json \\
    --image_folder /content/drive/MyDrive/TCSS590/Llava/data/images \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --bf16 True \\
    --bits 16 \\
    --output_dir /content/LLaVA/checkpoints/llava-lora-experiment \\
    --num_train_epochs 10 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 1 \\
    --evaluation_strategy no \\
    --save_strategy steps \\
    --save_steps 50 \\
    --save_total_limit 1 \\
    --learning_rate 2e-4 \\
    --weight_decay 0.0 \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type cosine \\
    --logging_steps 1 \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --lazy_preprocess True \\
    --report_to none
```

### Key Hyperparameters

|Parameter|Value|Notes|
|-|-|-|
|Base model|llava-v1.5-7b|7B parameter multimodal LLM|
|LoRA rank (r)|128|Higher = more trainable params|
|LoRA alpha|256|Scaling factor = alpha/r = 2.0|
|Precision|bf16|Brain float 16|
|Quantization|None (bits=16)|Full bf16, no 4-bit needed on A100|
|Learning rate|2e-4|With cosine decay|
|Warmup ratio|0.03|3% of steps for warmup|
|Epochs|10|Full passes over 20 samples|
|Batch size|4|Per device|
|Gradient accumulation|1|Effective batch = 4|
|DeepSpeed|ZeRO-2|Optimizer state offloading|
|Attention|Eager|flash_attn_2 skipped (compatibility)|

### Training Results

|Metric|Value|
|-|-|
|Final training loss|0.451 (avg), \~0.0003 (final epoch)|
|Training time|1 min 52 sec|
|Samples/sec|1.77|
|Convergence epoch|\~3.6 (loss < 0.01)|

### Loss Curve Summary

* **Epoch 0–1:** Rapid drop from \~5.0 → \~1.2 (model learning task format)
* **Epoch 1–3:** Continued drop to \~0.02 (learning transcriptions)
* **Epoch 3–10:** Near-zero loss (memorization of 20 samples)

\---

## Part B2 — Sample Set Fine-Tune + LLaVA-Bench Evaluation

### Overview

One additional epoch of LoRA fine-tuning on a \~2,500-pair subset of
`liuhaotian/LLaVA-Instruct-80K` with COCO train2017 images, followed by
evaluation on the built-in LLaVA-Bench-in-the-Wild benchmark scored by GPT-4.

**Date:** 2026-03-17  
**Dataset:** `llava_instruct_80k.json` (first 2,500 pairs)  
**Images:** COCO train2017  
**Benchmark:** LLaVA-Bench-in-the-Wild (60 questions)  
**Scorer:** GPT-4 via `eval_gpt_review_bench.py` (openai==0.28, model: gpt-4o-mini)

### Training Command

```bash
deepspeed /content/LLaVA/llava/train/train_mem.py \\
    --lora_enable True \\
    --lora_r 128 \\
    --lora_alpha 256 \\
    --deepspeed /content/LLaVA/scripts/zero2.json \\
    --model_name_or_path liuhaotian/llava-v1.5-7b \\
    --version v1 \\
    --data_path /content/drive/MyDrive/TCSS590/Llava/data/llava_2500.json \\
    --image_folder /content/drive/MyDrive/TCSS590/Llava/data/train2017 \\
    --vision_tower openai/clip-vit-large-patch14-336 \\
    --mm_projector_type mlp2x_gelu \\
    --mm_vision_select_layer -2 \\
    --mm_use_im_start_end False \\
    --mm_use_im_patch_token False \\
    --image_aspect_ratio pad \\
    --bf16 True \\
    --bits 16 \\
    --output_dir /content/drive/MyDrive/TCSS590/Llava/checkpoints/llava-lora-sample \\
    --num_train_epochs 1 \\
    --per_device_train_batch_size 4 \\
    --gradient_accumulation_steps 4 \\
    --evaluation_strategy no \\
    --save_strategy steps \\
    --save_steps 200 \\
    --save_total_limit 3 \\
    --learning_rate 2e-4 \\
    --weight_decay 0.0 \\
    --warmup_ratio 0.03 \\
    --lr_scheduler_type cosine \\
    --logging_steps 10 \\
    --report_to tensorboard \\
    --tf32 True \\
    --model_max_length 2048 \\
    --gradient_checkpointing True \\
    --lazy_preprocess True
```

### Key Hyperparameters

|Parameter|Value|Notes|
|-|-|-|
|Base model|llava-v1.5-7b|7B parameter multimodal LLM|
|LoRA rank (r)|128|Same as Part A|
|LoRA alpha|256|Scaling factor = alpha/r = 2.0|
|Precision|bf16|Brain float 16|
|Learning rate|2e-4|With cosine decay|
|Warmup ratio|0.03|3% of steps for warmup|
|Epochs|1|Single extra epoch|
|Batch size|4|Per device|
|Gradient accumulation|4|Effective batch = 16|
|DeepSpeed|ZeRO-2|Optimizer state offloading|
|Training pairs|\~2,500|Subset of LLaVA-Instruct-80K|

### Loss Curve Observations

* Loss range: **0.70 – 0.74** (narrow band, oscillating)
* No clear downward trend across the epoch
* Starting loss already low (\~0.73) — model close to pretrained baseline
* Oscillation likely due to learning rate being too high for a small dataset
and only 1 epoch being insufficient for convergence

### LLaVA-Bench-in-the-Wild Results

Evaluated on 60 questions across 3 categories, scored by GPT-4 on a 1–10 scale.

#### Overall Scores

|Metric|Value|
|-|-|
|Total questions|60|
|GPT-4 avg score|8.20|
|Our model avg score|5.70|
|**Relative score**|**69.5%**|

#### Scores by Category

|Category|GPT-4|Ours|Relative|
|-|-|-|-|
|Complex reasoning|7.86|6.54|83.2%|
|Conversation|8.94|5.12|57.2%|
|Detail description|8.00|4.80|60.0%|

#### Analysis

* **Complex reasoning (83.2%)** is the strongest category — the pretrained LLaVA base
retains general reasoning capability well after LoRA fine-tuning on a small dataset.
* **Conversation (57.2%)** and **Detail description (60.0%)** are weaker, likely because
1 epoch on 2,500 pairs is insufficient to meaningfully improve instruction-following
beyond the pretrained baseline.
* The 69.5% relative score is expected given the minimal training budget (1 epoch, \~2.5k pairs).
Further improvement would require more epochs, a larger dataset, or a lower learning rate
for stable convergence.

\---

## Dependencies

|Package|Version|
|-|-|
|torch|2.10.0+cu128|
|transformers|4.37.2|
|peft|0.9.0|
|deepspeed|0.15.4|
|bitsandbytes|0.49.2|
|accelerate|0.27.2|
|openai|0.28.0 (for bench scoring)|

\---

## Notes

* With only 20 samples (Part A) the model **memorizes** rather than generalizes
* Loss reaching near-zero in Part A is expected and confirms the pipeline works
* For real generalization, 500–5000+ diverse samples are recommended
* Part B2 loss oscillation suggests `learning_rate` should be reduced to `2e-5`
for a more stable single-epoch run on a small subset
* LoRA weights saved to:

  * Part A: `llava-lora-experiment/` checkpoint directory
  * Part B2: `checkpoints/llava-lora-sample/` checkpoint directory


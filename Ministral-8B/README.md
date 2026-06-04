# ministral-8b-ws3

A [Modal](https://modal.com) deployment of [`aitf-its-tim3-dfk/ministral-8b-ws3`](https://huggingface.co/aitf-its-tim3-dfk/ministral-8b-ws3) — a fine-tuned Ministral VLM served as a GPU-backed FastAPI endpoint.

The model performs social media content violation classification (DFK), returning a structured `Label:` and `Analisis:` given a post screenshot and metadata.

## Adapter

This deployment expects the LoRA adapter under:

```text
aitf-its-tim3-dfk/ministral-8b-ws3
subfolder: adapter
```

`modal_app_transformers.py` loads `adapter/adapter_config.json`, reads `base_model_name_or_path`, loads the base model, and applies the adapter with `PeftModel.from_pretrained(..., subfolder="adapter")`. If the model or adapter is private, provide `HF_TOKEN` through a Modal secret.

## Features

- **DFK classification** — detects violations from social media screenshots plus `ringkasan`, `klaim`, and `fakta`.
- **Captioning mode** — describes images with the LoRA adapter disabled. Optionally override with `caption_prompt`.
- **Free-form prompt** — bypasses the DFK template with a custom prompt, optionally with an image.
- **Mistral chat template fallback** — injects a local `[INST]/[/INST]` template if the tokenizer does not provide one.
- **Weave tracing** — records request metadata, latency, and generated output when `WANDB_API_KEY` is set.
- **GPU warmup tracking** — logs `move_to_gpu` time on cold start and includes `gpu_warmup_ms` in the first Weave trace.
- **CPU memory snapshot** — loads on CPU first, snapshots memory, then moves to GPU on container start.

## Setup

```bash
pip install modal
modal setup
```

For Hugging Face auth:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

The app also references `wandb-secret` for Weave/W&B logging. Without `WANDB_API_KEY`, Weave tracing is disabled at runtime.

## Commands

```bash
# One-shot inference
modal run Ministral-8B/modal_app_transformers.py --ringkasan "..." --klaim "..." --fakta "..."
modal run Ministral-8B/modal_app_transformers.py --prompt "Describe this image" --image-url "https://..."

# Dev server
modal serve Ministral-8B/modal_app_transformers.py

# Production deploy
modal deploy Ministral-8B/modal_app_transformers.py

# Stream logs
modal app logs ministral-8b-ws3
```

## API

**Endpoint:** `POST https://<your-modal-username>--ministral-8b-ws3-infer.modal.run`

### DFK Classification

```json
{
  "ringkasan": "Summary of the social media post",
  "klaim": "Claim made in the post",
  "fakta": "Verified fact for comparison",
  "image_url": "https://...",
  "max_new_tokens": 128,
  "temperature": 0.0
}
```

**Response:**

```json
{
  "text": "Label: UJARAN KEBENCIAN\n\nAnalisis: ...",
  "tokens_generated": 60,
  "infer_total_ms": 74240
}
```

`infer_total_ms` is the total end-to-end time measured at the endpoint level (includes network + queue + generation).

### Captioning

```json
{
  "captioning": true,
  "image_url": "https://...",
  "caption_prompt": "Describe everything you see in detail.",
  "max_new_tokens": 256
}
```

### Free-form Prompt

```json
{
  "prompt": "Your custom prompt here",
  "image_url": "https://...",
  "max_new_tokens": 256
}
```

### Mode Priority

| Priority | Trigger | Mode |
|----------|---------|------|
| 1 | `captioning: true` | Captioning (base model, adapter disabled) |
| 2 | `prompt` string | Free prompt (LoRA) |
| 3 | default | DFK classification (LoRA) |

Field aliases: `summary` → `ringkasan`, `claim` → `klaim`, `fact` → `fakta`. Images via `image_url` or `image_base64`.

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Max tokens to generate |
| `temperature` | 0.0 | Sampling temperature (`0` = greedy) |
| `top_p` | 0.8 | Nucleus sampling |
| `top_k` | 20 | Top-k sampling |
| `min_p` | 0.0 | Min-p sampling |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `caption_prompt` | — | Override captioning instruction |

## Architecture

**File:** `Ministral-8B/modal_app_transformers.py`

| Component | Description |
|-----------|-------------|
| `MinistralServer` | Modal class on L4 GPU. Loads processor/model on CPU, snapshots memory, then moves to GPU as `bfloat16`. |
| `infer` | FastAPI `POST` endpoint delegating to `MinistralServer().generate.remote(...)`. |
| HF Volume cache | `modal.Volume` named `ministral-8b-ws3-cache` persists downloaded base model and adapter weights. |

**Model loading flow:**
1. `snap=True` — loads processor and model to CPU concurrently, applies LoRA adapter, sets Mistral chat template fallback → **snapshot taken**
2. `snap=False` — moves model to GPU (`bfloat16`), logs warmup time, initializes Weave client if `WANDB_API_KEY` is set → ready to serve
3. Captioning requests use `self.model.disable_adapter()` to run the base model without LoRA

## Infrastructure

| Setting | Value |
|---------|-------|
| GPU | NVIDIA L4 |
| CPU | 4 vCPU |
| Memory | 24 GB |
| Timeout | 600s |
| Scale-down | 60s idle |
| Concurrency | 2 inputs per container |
| Snapshot | CPU memory snapshot enabled |

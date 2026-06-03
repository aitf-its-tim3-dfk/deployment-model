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
- **Logits label probe** — returns MTLA-style percentage scores for `NETRAL`, `DISINFORMASI`, `UJARAN KEBENCIAN`, and `FITNAH`.
- **Captioning mode** — describes images with the LoRA adapter disabled. Optionally override with `caption_prompt`.
- **Free-form prompt** — bypasses the DFK template with a custom prompt, optionally with an image.
- **Free-form messages** — accepts OpenAI-style `messages` array directly, optionally with an image via `image_url`.
- **Weave tracing** — records request metadata, latency, generated output, logits scores, and the rendered `model_prompt`.
- **Mistral chat template fallback** — injects a local template if the tokenizer does not provide one.
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
  "text": "Label: DISINFORMASI\n\nAnalisis: ...",
  "tokens_generated": 74,
  "logits_label": "DISINFORMASI",
  "logits_scores": {
    "NETRAL": 3.75,
    "DISINFORMASI": 77.68,
    "UJARAN KEBENCIAN": 11.35,
    "FITNAH": 7.22
  },
  "infer_total_ms": 33838
}
```

`logits_scores` is an experimental label probe. Scores are relative percentages, not calibrated probabilities. The probe is only added for normal DFK requests; captioning and free-form prompt requests keep the original response shape.

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

### Free-form Messages (OpenAI format)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Analyze this content..."}
  ],
  "max_new_tokens": 256
}
```

With image:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image"},
      {"type": "text", "text": "What is in this image?"}
    ]}
  ],
  "image_url": "https://...",
  "max_new_tokens": 256
}
```

### Mode Priority

| Priority | Trigger | Mode |
|----------|---------|------|
| 1 | `captioning: true` | Captioning (base model) |
| 2 | `messages` array | Free messages (LoRA) |
| 3 | `prompt` string | Free prompt (LoRA) |
| 4 | default | DFK classification (LoRA) |

Field aliases: `summary` → `ringkasan`, `claim` → `klaim`, `fact` → `fakta`. Images via `image_url` or `image_base64`.

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Max tokens to generate |
| `temperature` | 0.0 | Sampling temperature |
| `top_p` | 0.8 | Nucleus sampling |
| `top_k` | 20 | Top-k sampling |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `dfk_prompt` | — | Override DFK system instruction |
| `caption_prompt` | — | Override captioning instruction |
| `system_prompt` | — | Shortcut system message for any mode |

## Architecture

**File:** `Ministral-8B/modal_app_transformers.py`

| Component | Description |
|-----------|-------------|
| `MinistralServer` | Modal class on L4 GPU. Loads processor/model on CPU, snapshots memory, then moves to GPU as `bfloat16`. |
| `infer` | FastAPI `POST` endpoint delegating to `MinistralServer().generate.remote(...)`. |
| HF Volume cache | `modal.Volume` named `ministral-8b-ws3-cache` persists downloaded base model and adapter weights. |
| Logits probe | Scores fixed labels after the forced `Label: ` prefix using averaged label-token log probabilities, then softmaxes across labels. |
| Weave trace | Stores request fields, rendered `model_prompt`, generated output, latency, GPU warmup, and logits scores when available. |

**Logits scoring flow:**
1. Render the DFK chat template used for generation.
2. Append `Label: ` so logits are measured at the label position.
3. Score each fixed label token-by-token, averaging log probabilities for multi-token labels.
4. Softmax the averaged scores and return the top label as `logits_label`.

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

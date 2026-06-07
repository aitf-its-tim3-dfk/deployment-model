# ministral-8b-ws3

A [Modal](https://modal.com) deployment of [`aitf-its-tim3-dfk/ministral-8b-ws3`](https://huggingface.co/aitf-its-tim3-dfk/ministral-8b-ws3) — a fine-tuned Ministral VLM served as a GPU-backed FastAPI endpoint.

The model performs social media content violation classification (DFK), returning a structured `Label:` and `Analisis:` given a post screenshot and metadata.

## Adapter

```text
aitf-its-tim3-dfk/ministral-8b-ws3
subfolder: adapter
```

Loads `adapter/adapter_config.json`, reads `base_model_name_or_path` (`unsloth/Ministral-3-8B-Base-2512`), loads the base model, and applies the LoRA adapter with `PeftModel.from_pretrained(..., subfolder="adapter")`.

**Chat template:** Bundled as `templates/ministral_3.jinja` in this repo — copied into the Modal image at build time and loaded at startup. Identical to `adapter/chat_template.jinja` from the HF repo and `sita/templates/ministral_3.jinja` used during training.

## Features

- **DFK classification** — detects violations from social media screenshots plus `ringkasan`, `klaim`, and `fakta`.
- **Logits label probe** — returns softmax percentage scores for `NETRAL`, `DISINFORMASI`, `UJARAN KEBENCIAN`, and `FITNAH` alongside the generated response.
- **Captioning mode** — describes images with the LoRA adapter disabled. Optionally override with `caption_prompt`.
- **Free-form prompt** — bypasses the DFK template with a custom prompt, optionally with an image.
- **Free-form messages** — accepts OpenAI-style `messages` array with text and images.
- **Multi-image support** — `{"type": "image_url", "image_url": {"url": "..."}}` blocks in messages are downloaded and passed as a list to the model.
- **Custom system role** — inject `{"role": "system"}` into the conversation via `system_role` parameter.
- **Custom DFK instruction** — override the user-message instruction via `dfk_prompt` or the `system_prompt` shortcut.
- **Bundled chat template** — `templates/ministral_3.jinja` baked into the image, no HF download at runtime.
- **Weave tracing** — records request metadata, rendered `model_prompt`, latency, logits scores, and GPU warmup when `WANDB_API_KEY` is set.
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
  "text": "Label: ujaran kebencian\n\nAnalisis: ...",
  "tokens_generated": 39,
  "logits_label": "UJARAN KEBENCIAN",
  "logits_scores": {
    "NETRAL": 0.18,
    "DISINFORMASI": 14.22,
    "UJARAN KEBENCIAN": 85.38,
    "FITNAH": 0.22
  },
  "infer_total_ms": 28566
}
```

`logits_label` and `logits_scores` are only returned for DFK mode. Scores are relative percentages, not calibrated probabilities. `infer_total_ms` is end-to-end time at the endpoint level.

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

Single image via `image_url` block:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://..."}},
      {"type": "text", "text": "Analisis gambar ini dalam Bahasa Indonesia."}
    ]}
  ],
  "max_new_tokens": 256
}
```

Multiple images — pass multiple `image_url` blocks in content:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://...image1..."}},
      {"type": "image_url", "image_url": {"url": "https://...image2..."}},
      {"type": "text", "text": "Bandingkan kedua gambar ini."}
    ]}
  ],
  "max_new_tokens": 256
}
```

You can also still use `{"type": "image"}` placeholders combined with the top-level `image_url` parameter (legacy format).

### Custom System Role

Inject a system message into any mode via `system_role`:

```json
{
  "ringkasan": "...",
  "klaim": "...",
  "fakta": "...",
  "system_role": "Kamu adalah classifier konten DFK. Jawab singkat.",
  "max_new_tokens": 128
}
```

If the `messages` array already contains a `{"role": "system"}` entry, `system_role` is ignored.

### Mode Priority

| Priority | Trigger | Mode |
|----------|---------|------|
| 1 | `captioning: true` | Captioning (base model, adapter disabled) |
| 2 | `messages` array | Free messages (LoRA) |
| 3 | `prompt` string | Free prompt (LoRA) |
| 4 | default | DFK classification (LoRA) |

Field aliases: `summary` → `ringkasan`, `claim` → `klaim`, `fact` → `fakta`. Images via `image_url`, `image_base64`, or `image_url` blocks inside messages.

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Max tokens to generate |
| `temperature` | 0.0 | Sampling temperature (`0` = greedy) |
| `top_p` | 0.8 | Nucleus sampling |
| `top_k` | 20 | Top-k sampling |
| `min_p` | 0.0 | Min-p sampling |
| `repetition_penalty` | 1.0 | Repetition penalty |
| `system_role` | — | Inject `{"role": "system"}` into the conversation |
| `dfk_prompt` | — | Override DFK instruction in user message (also aliased as `dfk_system_prompt`, `dfk_instruction`) |
| `caption_prompt` | — | Override captioning instruction (also aliased as `caption_system_prompt`, `caption_instruction`) |
| `system_prompt` | — | Shortcut: routes to `dfk_prompt` for DFK mode or `caption_prompt` for captioning mode |

## Architecture

**File:** `Ministral-8B/modal_app_transformers.py`

| Component | Description |
|-----------|-------------|
| `MinistralServer` | Modal class on L4 GPU. Loads processor/model on CPU, snapshots memory, then moves to GPU as `bfloat16`. |
| `infer` | FastAPI `POST` endpoint delegating to `MinistralServer().generate.remote(...)`. |
| HF Volume cache | `modal.Volume` named `ministral-8b-ws3-cache` persists downloaded base model and adapter weights. |
| Chat template | `templates/ministral_3.jinja` bundled into the image. Uses `[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]` + `[INST]...[/INST]` format matching fine-tuning. |
| `_extract_images_from_messages` | Scans message content for `image_url` blocks, downloads each, replaces with `{"type": "image"}` placeholder. |
| Logits probe | Scores fixed labels at the `Label: ` position using averaged log probabilities, then softmaxes across labels. |
| Weave trace | Stores request fields, rendered `model_prompt`, generated output, latency, GPU warmup, and logits scores. Project: `Aitf-dfk-3/ministral`. |

**Model loading flow:**
1. `snap=True` — loads processor + model to CPU concurrently, loads `ministral_3.jinja` and sets it on the tokenizer, applies LoRA adapter, pre-computes label token sequences → **snapshot taken**
2. `snap=False` — moves model to GPU (`bfloat16`), logs warmup time, initializes Weave client if `WANDB_API_KEY` is set → ready to serve
3. Captioning requests use `self.model.disable_adapter()` to run the base model without LoRA

**Logits scoring flow:**
1. Render the DFK chat template used for generation.
2. Append `Label: ` so logits are measured at the label position.
3. Score each fixed label token-by-token, averaging log probabilities for multi-token labels.
4. Softmax the averaged scores into percentage-like values and return the top label as `logits_label`.

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

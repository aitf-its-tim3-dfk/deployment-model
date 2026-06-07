# qwen35-ws3

A [Modal](https://modal.com) deployment of [`aitf-its-tim3-dfk/qwen3.5-0.8B-ws3`](https://huggingface.co/aitf-its-tim3-dfk/qwen3.5-0.8B-ws3) — a fine-tuned Qwen VLM (vision-language model) served as a GPU-backed FastAPI endpoint.

The model performs social media content violation classification (DFK), returning a structured `Label:` and `Analisis:` given a post's screenshot and metadata.

## Features

- **DFK classification** — detects violations (hate speech, disinformation, etc.) from social media screenshots + metadata
- **Captioning mode** — describes images in detail in Bahasa Indonesia (uses base model, LoRA adapter disabled)
- **Free-form prompt** — bypass the DFK template entirely with a custom prompt
- **Weave tracing** — records request metadata, latency, and generated output
- **GPU warmup tracking** — logs `move_to_gpu` time on cold start and includes `gpu_warmup_ms` in the first Weave trace
- **CPU memory snapshot** — model loaded to CPU once, snapshotted, restored on cold start (~20s vs ~70s without)
- **Concurrent inputs** — one container handles up to 2 parallel requests before scaling

## Setup

```bash
pip install modal
modal setup
```

If the HF repo is private, create a Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

Then add `secrets=[modal.Secret.from_name("huggingface-secret")]` to both `@app.cls(...)` and `@app.function(...)` in `modal_app_transformers.py`.

## Commands

```bash
# One-shot inference (runs on remote GPU)
modal run Qwen-0.8B/modal_app_transformers.py --ringkasan "..." --klaim "..." --fakta "..."
modal run Qwen-0.8B/modal_app_transformers.py --prompt "Describe this image" --image-url "https://..."

# Dev server (hot-reload, temporary endpoint URL printed to console)
modal serve Qwen-0.8B/modal_app_transformers.py

# Production deploy
modal deploy Qwen-0.8B/modal_app_transformers.py

# Stream logs
modal app logs qwen35-ws3
```

## API

**Endpoint:** `POST https://<your-modal-username>--qwen35-ws3-infer.modal.run`

### DFK Classification

```json
{
  "ringkasan": "Summary of the social media post",
  "klaim": "Claim made in the post",
  "fakta": "Verified fact for comparison",
  "image_url": "https://...",
  "max_new_tokens": 128,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0.0,
  "repetition_penalty": 1.0
}
```

**Response:**
```json
{
  "text": "Label: DISINFORMASI\n\nAnalisis: ...",
  "tokens_generated": 74,
  "infer_total_ms": 4210
}
```

`infer_total_ms` is the total end-to-end time measured at the endpoint level (includes network + queue + generation).

### Image Captioning

```json
{
  "captioning": true,
  "image_url": "https://...",
  "max_new_tokens": 256
}
```

Optionally override the default captioning instruction:

```json
{
  "captioning": true,
  "image_url": "https://...",
  "caption_prompt": "What text is visible in this image?"
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

With multiple images using OpenAI-style `image_url` blocks:

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

| Priority | Trigger | Mode | Adapter |
|----------|---------|------|---------|
| 1 | `captioning: true` | Captioning | disabled |
| 2 | `messages` array | Free messages | disabled |
| 3 | `prompt` string | Free prompt | disabled |
| 4 | default | DFK classification | active |

### Field Aliases

| Field | Alias |
|-------|-------|
| `ringkasan` | `summary` |
| `klaim` | `claim` |
| `fakta` | `fact` |

### Input Options

| Field | Type | Description |
|-------|------|-------------|
| `image_url` | string | Public image URL |
| `image_base64` | string | Base64-encoded image (data URI prefix stripped automatically) |
| `messages[].content[].image_url` | object | OpenAI-style image URL block, supports multiple images in messages |

## Architecture

**File:** `Qwen-0.8B/modal_app_transformers.py`

| Component | Description |
|-----------|-------------|
| `QwenServer` | Modal class on L4 GPU. Loads model once via `@modal.enter(snap=True)` to CPU, snapshots memory, then moves to GPU via `@modal.enter(snap=False)`. |
| `infer` | FastAPI `POST` endpoint. Thin wrapper delegating to `QwenServer().generate.remote(...)`. |
| `main` | Local entrypoint for `modal run`. |
| HF Volume cache | `modal.Volume` named `qwen35-ws3-cache` persists downloaded weights across cold starts. |
| Chat template | Uses bundled `templates/qwen3.5_chatml.jinja` inside the Modal image. |

**Model loading flow:**
1. `snap=True` — downloads weights (cached in Volume), loads processor + model to CPU concurrently via `ThreadPoolExecutor`, wraps with `PeftModel` if LoRA adapter detected → **snapshot taken**
2. `snap=False` — moves model from CPU → GPU (`bfloat16`), logs warmup time, initializes Weave → ready to serve
3. Non-DFK requests (`captioning`, `messages`, and `prompt`) use `self.model.disable_adapter()` to run the base model without LoRA

## Generation Parameters

All generation parameters are optional and apply to both DFK and captioning modes.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Maximum tokens to generate |
| `temperature` | 0.0 | Sampling temperature. `0` = greedy/deterministic. Higher = more random |
| `top_p` | 0.8 | Nucleus sampling — only sample from top tokens whose cumulative probability ≥ `top_p`. Active when `temperature > 0` |
| `top_k` | 20 | Only sample from top-k most probable tokens. Active when `temperature > 0` |
| `min_p` | 0.0 | Minimum probability threshold relative to top token. Active when `temperature > 0` and `min_p > 0` |
| `repetition_penalty` | 1.0 | `1.0` = no penalty. `> 1.0` penalizes repeated tokens |

**Example with custom generation params:**

```json
{
  "ringkasan": "...",
  "klaim": "...",
  "fakta": "...",
  "image_url": "https://...",
  "max_new_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

> Note: `top_p`, `top_k`, and `min_p` are ignored when `temperature` is `0` (greedy decoding).

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

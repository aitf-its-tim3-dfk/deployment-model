# deployment-model

Modal deployments for DFK (social media content violation classification) models. Each model is served as a GPU-backed FastAPI endpoint supporting DFK classification, image captioning, free-form prompts, and OpenAI-compatible message format.

## Models

| Model | Description | Docs |
|-------|-------------|------|
| [Qwen-0.8B](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Qwen-0.8B) | Qwen3.5-0.8B VLM fine-tuned for DFK — lightweight, fast | [README](Qwen-0.8B/README.md) |
| [Qwen-0.8B-v2](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Qwen-0.8B-v2) | Qwen3.5-0.8B with updated adapter (`KomdigiITS-0.8B-DFK`) | [README](Qwen-0.8B-v2/README.md) |
| [Ministral-8B](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Ministral-8B) | Ministral-3-8B VLM fine-tuned for DFK — larger, more capable | [README](Ministral-8B/README.md) |

## Input Modes

All models support the same four input modes:

| Mode | Trigger |
|------|---------|
| DFK classification | `ringkasan`, `klaim`, `fakta` fields |
| Captioning | `captioning: true` + `image_url` |
| Free-form prompt | `prompt` field |
| Free-form messages | `messages` array (OpenAI format) |

## Setup

```bash
pip install modal
modal setup
```

Deploy any model:

```bash
modal deploy Qwen-0.8B/modal_app_transformers.py
modal deploy Ministral-8B/modal_app_transformers.py
```

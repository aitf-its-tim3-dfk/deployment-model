# deployment-model

Modal deployments for DFK (social media content violation classification) models. Each model is served as a GPU-backed FastAPI endpoint supporting DFK classification, image captioning, free-form prompts, and OpenAI-compatible message format.

## Models

| Model | Description | Docs |
|-------|-------------|------|
| [Qwen-0.8B](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Qwen-0.8B) | Qwen3.5-0.8B VLM fine-tuned for DFK — lightweight, fast | [README](Qwen-0.8B/README.md) |
| [Qwen-0.8B-v2](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Qwen-0.8B-v2) | Qwen3.5-0.8B with updated adapter (`KomdigiITS-0.8B-DFK`) | [README](Qwen-0.8B-v2/README.md) |
| [Ministral-8B](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Ministral-8B) | Ministral-3-8B VLM SFT fine-tuned for DFK — larger, more capable | [README](Ministral-8B/README.md) |
| [Ministral-8B-CPT](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Ministral-8B-CPT) | Ministral-3-8B VLM with CPT on domain data, then SFT fine-tuned for DFK | [README](Ministral-8B-CPT/README.md) |
| [Ministral-8B-Merged](https://github.com/aitf-its-tim3-dfk/deployment-model/tree/main/Ministral-8B-Merged) | Ministral adapter-mode deployment for `ministral-8b-merged-ws3`, with DFK-3 multimodal plus DFK-1 raw text classification | [README](Ministral-8B-Merged/README.md) |

## Input Modes

All transformer deployments support the standard DFK multimodal, captioning, prompt, and messages modes. `Ministral-8B-Merged` additionally supports DFK-1 text classification and experimental adapter-backed messages.

| Mode | Trigger | Adapter |
|------|---------|---------|
| DFK classification | `ringkasan`, `klaim`, `fakta` fields | active |
| Captioning | `captioning: true` + `image_url` | disabled |
| Free-form prompt | `prompt` field | disabled |
| Free-form messages | `messages` array (OpenAI format) | disabled |
| DFK-1 text classification (`Ministral-8B-Merged` only) | `text_classification: true` or `mode: "dfk_text"` | active |
| Experimental adapter messages (`Ministral-8B-Merged` only) | `adapter_messages: true`, `dfk_messages: true`, or `mode: "adapter_messages"` | active |

`messages` supports OpenAI-style image blocks such as `{"type": "image_url", "image_url": {"url": "https://..."}}` for multi-image requests. You can inject a system message into any mode with `system_role`. Use `adapter_messages` only for experiments where free-form messages should run with the DFK adapter enabled.

## Setup

```bash
pip install modal
modal setup
```

Deploy any model:

```bash
modal deploy Qwen-0.8B/modal_app_transformers.py
modal deploy Qwen-0.8B-v2/modal_app_transformers.py
modal deploy Ministral-8B/modal_app_transformers.py
modal deploy Ministral-8B-CPT/modal_app_transformers.py
modal deploy Ministral-8B-Merged/modal_app_transformers.py
```


## App Names

| Folder | Modal app | Hugging Face model |
|--------|-----------|--------------------|
| `Qwen-0.8B` | `qwen35-ws3` | `aitf-its-tim3-dfk/qwen3.5-0.8B-ws3` |
| `Qwen-0.8B-v2` | `qwen35-ws3-v2` | `aitf-its-tim3-dfk/KomdigiITS-0.8B-DFK-MultimodalClassification` |
| `Ministral-8B` | `ministral-8b-ws3` | `aitf-its-tim3-dfk/ministral-8b-ws3` |
| `Ministral-8B-CPT` | `ministral-cpt-8b-ws3` | `aitf-its-tim3-dfk/ministral-cpt-8b-ws3` |
| `Ministral-8B-Merged` | `ministral-8b-merged-ws3` | `aitf-its-tim3-dfk/ministral-8b-merged-ws3` |

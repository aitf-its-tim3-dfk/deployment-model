# ministral-8b-merged-ws3

A [Modal](https://modal.com) deployment of [`aitf-its-tim3-dfk/ministral-8b-merged-ws3`](https://huggingface.co/aitf-its-tim3-dfk/ministral-8b-merged-ws3) served as a GPU-backed FastAPI endpoint.

This deployment intentionally uses the repo's **LoRA adapter path** (`adapter/`) instead of loading the full merged `model.safetensors`. That keeps adapter control available: task adapters are active for DFK modes, and disabled for general captioning/prompt/messages modes.

## Adapter

```text
aitf-its-tim3-dfk/ministral-8b-merged-ws3
subfolder: adapter
base_model_name_or_path: aitf-komdigi/ministral-8b-dfk-cpt-last
```

The HF repo also contains a full merged `model.safetensors`, but this Modal app does **not** load it. Loading full merged weights would make `disable_adapter()` impossible because the adapter effect is already baked into model weights.

## Modes

| Priority | Trigger | Mode | Adapter | Training format |
|----------|---------|------|---------|-----------------|
| 1 | `text_classification: true`, `dfk_text: true`, `dfk1: true`, or `mode: "text_classification"` | DFK-1 text classification | active | raw `dfk_text_dataset.py` format |
| 2 | `adapter_messages: true`, `dfk_messages: true`, or `mode: "adapter_messages"` | Experimental free messages with adapter | active | OpenAI messages through DFK adapter |
| 3 | `captioning: true` | Captioning | disabled | base model |
| 4 | `messages` array | Free messages | disabled | base model |
| 5 | `prompt` string | Free prompt | disabled | base model |
| 6 | default | DFK-3 multimodal classification | active | existing `ringkasan` / `klaim` / `fakta` format |

## DFK-3 Multimodal Classification

Same format as the existing Ministral DFK deployment. The instruction stays in the user prompt, followed by metadata fields:

```json
{
  "ringkasan": "Caption and social media post summary",
  "klaim": "Core claim extracted from the content",
  "fakta": "Fact-check context or references",
  "image_url": "https://...",
  "max_new_tokens": 128,
  "temperature": 0.0
}
```

Generated prompt shape:

```text
[INST]Anda adalah seorang analis konten media sosial ahli. ...
Ringkasan: ...
Klaim: ...
Fakta: ...
[IMG][/INST]
```

Response includes generated text plus the 4-label logits probe for:

```text
NETRAL, DISINFORMASI, UJARAN KEBENCIAN, FITNAH
```

## DFK-1 Text Classification

This mode follows the **raw prompt** from [`sita/datasets/dfk_text_dataset.py`](https://github.com/aitf-its-tim3-dfk/SITA/blob/main/sita/datasets/dfk_text_dataset.py), not the merged-label prompt.

Use one of these triggers:

```json
{
  "text_classification": true,
  "klaim": "Klaim yang akan dibandingkan.",
  "fakta": "Artikel rujukan untuk fact-checking.",
  "max_new_tokens": 128,
  "temperature": 0.0
}
```

Aliases:

```json
{
  "mode": "dfk_text",
  "claim": "...",
  "fact": "..."
}
```

Prompt shape:

```text
[SYSTEM_PROMPT]
Anda adalah sistem deteksi konten DFK berbasis artikel rujukan. Tugas Anda adalah membandingkan klaim dengan artikel rujukan, lalu mengklasifikasikan teks ke dalam salah satu label: Fakta, Disinformasi, Fitnah, Ujaran Kebencian, atau Non-DFK. Jawab dengan format: Label: **NamaLabel.** penjelasan: ...
[/SYSTEM_PROMPT]
[INST]
{klaim}

Artikel Rujukan: {fakta}
[/INST]
```

Raw labels are kept as 5 classes:

```text
Fakta, Disinformasi, Fitnah, Ujaran Kebencian, Non-DFK
```

`Fakta` and `Non-DFK` are **not** merged into `Netral` in this mode.

Optional prompt override:

```json
{
  "text_classification": true,
  "klaim": "...",
  "fakta": "...",
  "text_classification_prompt": "Custom system prompt..."
}
```

## Other Modes

Captioning, free-form prompt, and free-form messages are still available and run with adapter disabled.

### Captioning

```json
{
  "captioning": true,
  "image_url": "https://...",
  "max_new_tokens": 256
}
```

### Free-form Prompt

```json
{
  "prompt": "Describe this image in Indonesian.",
  "image_url": "https://...",
  "max_new_tokens": 256
}
```

### Free-form Messages

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://..."}},
      {"type": "text", "text": "Analisis gambar ini."}
    ]}
  ],
  "max_new_tokens": 256
}
```

### Experimental Adapter Messages

Same input shape as free-form messages, but keeps the DFK adapter active for experiments:

```json
{
  "adapter_messages": true,
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://..."}},
      {"type": "text", "text": "Analisis konten ini dengan adapter DFK."}
    ]}
  ],
  "max_new_tokens": 256
}
```

Aliases: `dfk_messages: true` or `mode: "adapter_messages"` / `mode: "dfk_messages"`.

## Commands

```bash
modal run Ministral-8B-Merged/modal_app_transformers.py --ringkasan "..." --klaim "..." --fakta "..."
modal run Ministral-8B-Merged/modal_app_transformers.py --text-classification --klaim "..." --fakta "..."
modal deploy Ministral-8B-Merged/modal_app_transformers.py
modal app logs ministral-8b-merged-ws3
```

## Endpoint

```text
POST https://<your-modal-username>--ministral-8b-merged-ws3-infer.modal.run
```

## cURL Examples

DFK-1 text classification:

```bash
curl -X POST "https://<your-modal-username>--ministral-8b-merged-ws3-infer.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "text_classification": true,
    "klaim": "Selamat pagi semuanya, jangan lupa sarapan dan tetap semangat bekerja hari ini.",
    "fakta": "Teks tersebut adalah sapaan dan ajakan umum. Tidak terdapat klaim faktual yang perlu diverifikasi dan tidak terkait disinformasi, fitnah, atau ujaran kebencian.",
    "max_new_tokens": 128,
    "temperature": 0.0
  }'
```

DFK-3 multimodal/default:

```bash
curl -X POST "https://<your-modal-username>--ministral-8b-merged-ws3-infer.modal.run" \
  -H "Content-Type: application/json" \
  -d '{
    "ringkasan": "Unggahan berisi ucapan selamat pagi dan ajakan minum kopi tanpa klaim politik, kesehatan, atau tuduhan terhadap pihak tertentu.",
    "klaim": "Ucapan selamat pagi dan ajakan minum kopi.",
    "fakta": "Tidak ditemukan klaim faktual yang perlu diverifikasi. Konten bersifat sapaan umum dan tidak terkait kategori DFK.",
    "max_new_tokens": 128,
    "temperature": 0.0
  }'
```

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

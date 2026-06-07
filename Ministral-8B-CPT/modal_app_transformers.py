import base64
import io
from typing import Any

import modal


APP_NAME = "ministral-cpt-8b-ws3"
MODEL_ID = "aitf-its-tim3-dfk/ministral-cpt-8b-ws3"
CACHE_DIR = "/cache/huggingface"
ADAPTER_SUBFOLDER = None  # adapter is at repo root, no subfolder
LABELS = ["NETRAL", "DISINFORMASI", "UJARAN KEBENCIAN", "FITNAH"]
LABEL_CANDIDATES = {
    "NETRAL": "netral",
    "DISINFORMASI": "disinformasi",
    "UJARAN KEBENCIAN": "ujaran kebencian",
    "FITNAH": "fitnah",
}


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("ministral-cpt-8b-ws3-cache", create_if_missing=True)

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential", "ninja-build", "libz3-dev")
    .pip_install(
        "torch==2.8.0",
        "torchvision",
        "triton>=3.3.0",
        "einops",
        "accelerate",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
        "peft",
        "pillow",
        "requests",
        "sentencepiece",
        "ninja",
        "packaging",
        "weave",
        FLASH_ATTN_WHEEL,
        "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)

with image.imports():
    from huggingface_hub import hf_hub_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor


def _decode_image(image_base64: str):
    from PIL import Image

    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _load_image_url(image_url: str):
    import requests
    from PIL import Image

    response = requests.get(
        image_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# Fallback chat template — exact copy of adapter/chat_template.jinja from ministral-8b-ws3
MISTRAL_VLM_CHAT_TEMPLATE = """{#- Unsloth template fixes #}
{#- Default system message if no system prompt is passed. #}
{%- set default_system_message = 'You are Ministral-3-8B-Instruct-2512, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYou power an AI assistant called Le Chat.\\nYour knowledge base was last updated on 2023-10-01.\\nThe current date is {today}.\\n\\nWhen you\\'re not sure about some information or when the user\\'s request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don\\'t have the information and avoid making up anything.\\nIf the user\\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").\\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.\\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\\nNext sections describe the capabilities that you have.\\n\\n# WEB BROWSING INSTRUCTIONS\\n\\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\\n\\n# MULTI-MODAL INSTRUCTIONS\\n\\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\\nYou cannot read nor transcribe audio files or videos.\\n\\n# TOOL CALLING INSTRUCTIONS\\n\\nYou may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:\\n\\n1. When the request requires up-to-date information.\\n2. When the request requires specific data that you do not have in your knowledge base.\\n3. When the request involves actions that you cannot perform without tools.\\n\\nAlways prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment.' %}

{#- Begin of sequence token. #}
{{- bos_token }}

{#- Handle system prompt if it exists. #}
{#- System prompt supports text content or text chunks. #}
{%- if messages[0]['role'] == 'system' %}
    {{- '[SYSTEM_PROMPT]' -}}
    {%- if messages[0]['content'] is string %}
        {{- messages[0]['content'] -}}
    {%- else %}
        {%- for block in messages[0]['content'] %}
            {%- if block['type'] == 'text' %}
                {{- block['text'] }}
            {%- else %}
                {{- raise_exception('Only text chunks are supported in system message contents.') }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '[/SYSTEM_PROMPT]' -}}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set loop_messages = messages %}
    {%- if default_system_message != '' %}
        {{- '[SYSTEM_PROMPT]' + default_system_message + '[/SYSTEM_PROMPT]' }}
    {%- endif %}
{%- endif %}


{#- Tools definition #}
{%- set tools_definition = '' %}
{%- set has_tools = false %}
{%- if tools is defined and tools is not none and tools|length > 0 %}
    {%- set has_tools = true %}
    {%- set tools_definition = '[AVAILABLE_TOOLS]' + (tools| tojson) + '[/AVAILABLE_TOOLS]' %}
    {{- tools_definition }}
{%- endif %}

{#- Checks for alternating user/assistant messages. #}
{%- set ns = namespace(index=0) %}
{%- for message in loop_messages %}
    {%- if message.role == 'user' or (message.role == 'assistant' and (message.tool_calls is not defined or message.tool_calls is none or message.tool_calls | length == 0)) %}
        {%- if (message['role'] == 'user') != (ns.index % 2 == 0) %}
            {{- raise_exception('After the optional system message, conversation roles must alternate user and assistant roles except for tool calls and results.') }}
        {%- endif %}
        {%- set ns.index = ns.index + 1 %}
    {%- endif %}
{%- endfor %}

{#- Handle conversation messages. #}
{%- for message in loop_messages %}

    {#- User messages supports text content or text and image chunks. #}
    {%- if message['role'] == 'user' %}
        {%- if message['content'] is string %}
            {{- '[INST]' + message['content'] + '[/INST]' }}
        {%- elif message['content'] | length > 0 %}
            {{- '[INST]' }}
            {%- if message['content'] | length == 2 %}
                {%- set blocks = message['content'] | sort(attribute='type') %}
            {%- else %}
                {%- set blocks = message['content'] %}
            {%- endif %}
            {%- for block in blocks %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- elif block['type'] in ['image', 'image_url'] %}
                    {{- '[IMG]' }}
                {%- else %}
                    {{- raise_exception('Only text, image and image_url chunks are supported in user message content.') }}
                {%- endif %}
            {%- endfor %}
            {{- '[/INST]' }}
        {%- else %}
            {{- raise_exception('User message must have a string or a list of chunks in content') }}
        {%- endif %}

    {#- Assistant messages supports text content or text and image chunks. #}
    {%- elif message['role'] == 'assistant' %}

        {%- if message['content'] is string %}
            {{- message['content'] }}
        {%- elif message['content'] is iterable and message['content'] | length > 0 %}
            {%- for block in message['content'] %}
                {%- if block['type'] == 'text' %}
                    {{- block['text'] }}
                {%- else %}
                    {{- raise_exception('Only text chunks are supported in assistant message contents.') }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}

        {%- if message['tool_calls'] is defined and message['tool_calls'] is not none and message['tool_calls']|length > 0 %}
            {%- for tool in message['tool_calls'] %}
                {%- set arguments = tool['function']['arguments'] %}
                {%- if arguments is not string %}
                    {%- set arguments = arguments|tojson|safe %}
                {%- elif arguments == '' %}
                    {%- set arguments = '{}' %}
                {%- endif %}
                {{- '[TOOL_CALLS]' + tool['function']['name'] + '[ARGS]' + arguments }}
            {%- endfor %}
        {%- endif %}

        {#- End of sequence token for each assistant messages. #}
        {{- eos_token }}

    {#- Tool messages only supports text content. #}
    {%- elif message['role'] == 'tool' %}
        {{- '[TOOL_RESULTS]' + message['content']|string + '[/TOOL_RESULTS]' }}

    {#- Raise exception for unsupported roles. #}
    {%- else %}
        {{- raise_exception('Only user, assistant and tool roles are supported, got ' + message['role']) }}
    {%- endif %}
{%- endfor %}

{#- Copyright 2025-present Unsloth. Apache 2.0 License. #}"""

DFK_INSTRUCTION = (
    "Anda adalah seorang analis konten media sosial ahli. "
    "Diberikan tangkapan layar dari sebuah unggahan media sosial dan metadata berupa "
    "ringkasan, klaim, serta fakta pembanding. Tentukan label kategori pelanggaran "
    "dan berikan analisis detail mengenai pelanggaran yang ditemukan. "
    "Jawab hanya dengan format: Label: <label> lalu Analisis: <analisis>."
)

CAPTIONING_INSTRUCTION = (
    "Jika ada teks di gambar, kutip teksnya terlebih dahulu. "
    "Kemudian deskripsikan isi gambar dalam satu paragraf menggunakan Bahasa Indonesia."
)


def build_dfk_content(
    ringkasan: str = "",
    klaim: str = "",
    fakta: str = "",
    instruction: str | None = None,
) -> list[dict[str, str]]:
    context_parts = []
    if ringkasan.strip():
        context_parts.append(f"Ringkasan: {ringkasan.strip()}")
    if klaim.strip():
        context_parts.append(f"Klaim: {klaim.strip()}")
    if fakta.strip():
        context_parts.append(f"Fakta: {fakta.strip()}")

    content = [{"type": "text", "text": instruction or DFK_INSTRUCTION}]
    if context_parts:
        content.append({"type": "text", "text": "\n".join(context_parts)})
    return content


@app.cls(
    image=image,
    gpu="L4",
    cpu=4,
    memory=24 * 1024,
    timeout=600,
    scaledown_window=60,
    volumes={CACHE_DIR: hf_cache},
    enable_memory_snapshot=True,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
@modal.concurrent(max_inputs=2)
class MinistralCPTServer:
    @modal.enter(snap=True)
    def load_to_cpu(self):
        import json
        import os
        from concurrent.futures import ThreadPoolExecutor

        token = os.environ.get("HF_TOKEN")

        base_model_id = MODEL_ID
        adapter_model_id = None
        adapter_subfolder = ADAPTER_SUBFOLDER

        try:
            adapter_config_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="adapter_config.json",
                subfolder=ADAPTER_SUBFOLDER,
                token=token,
                cache_dir=CACHE_DIR,
            )
            with open(adapter_config_path) as f:
                adapter_config = json.load(f)
            base_model_id = adapter_config.get("base_model_name_or_path") or base_model_id
            adapter_model_id = MODEL_ID
        except Exception as e:
            print(f"[INIT] adapter_config.json not found: {e}")

        def load_processor():
            return AutoProcessor.from_pretrained(
                base_model_id,
                token=token,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
            )

        def load_model():
            return AutoModelForImageTextToText.from_pretrained(
                base_model_id,
                token=token,
                trust_remote_code=True,
                dtype="bfloat16",
                device_map="cpu",
                cache_dir=CACHE_DIR,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            f_processor = executor.submit(load_processor)
            f_model = executor.submit(load_model)
            self.processor = f_processor.result()
            self.model = f_model.result()

        # Try to load chat_template.jinja from adapter repo root
        chat_template_loaded = False
        if adapter_model_id:
            try:
                chat_template_path = hf_hub_download(
                    repo_id=MODEL_ID,
                    filename="chat_template.jinja",
                    subfolder=ADAPTER_SUBFOLDER,
                    token=token,
                    cache_dir=CACHE_DIR,
                )
                with open(chat_template_path) as f:
                    self.processor.tokenizer.chat_template = f.read()
                chat_template_loaded = True
                print("[INIT] chat_template loaded from repo root chat_template.jinja")
            except Exception:
                pass

        if not chat_template_loaded and not getattr(self.processor.tokenizer, "chat_template", None):
            self.processor.tokenizer.chat_template = MISTRAL_VLM_CHAT_TEMPLATE
            print("[INIT] using hardcoded MISTRAL_VLM_CHAT_TEMPLATE fallback")

        if adapter_model_id:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_model_id,
                token=token,
                cache_dir=CACHE_DIR,
                subfolder=adapter_subfolder,
            )

        self.model.eval()

        self._label_token_seqs = [
            self.processor.tokenizer.encode(label, add_special_tokens=False)
            for label in LABELS
        ]
        self._label_all_tokens = list(
            {token_id for token_ids in self._label_token_seqs for token_id in token_ids}
        )
        print(f"[INIT] logits labels: {dict(zip(LABELS, self._label_token_seqs))}")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        import torch
        import os
        import time as _time

        t0 = _time.time()
        self.model = self.model.to("cuda", dtype=torch.bfloat16)
        self._gpu_warmup_ms = int((_time.time() - t0) * 1000)
        print(f"[COLD_START] move_to_gpu elapsed_ms={self._gpu_warmup_ms}")

        self._weave = None
        self._weave_client = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import weave
                self._weave_client = weave.init("Aitf-dfk-3/ministral-cpt")
                self._weave = weave
            except Exception as e:
                print(f"[WEAVE] init failed: {e}")
        else:
            print("[WEAVE] disabled: WANDB_API_KEY is not set")

    def _score_logits_labels(self, text: str, images) -> dict[str, Any]:
        import torch

        device = next(self.model.parameters()).device
        base_inputs = self.processor(
            text=[text + "Label:"],
            images=images,
            return_tensors="pt",
        ).to(device)
        base_ids = base_inputs["input_ids"][0]

        candidates = []
        allowed_tokens = set()
        for label in LABELS:
            label_text = LABEL_CANDIDATES[label]
            candidate_inputs = self.processor(
                text=[text + f"Label: {label_text}"],
                images=images,
                return_tensors="pt",
            ).to(device)
            candidate_ids = candidate_inputs["input_ids"][0]
            prefix_len = base_ids.shape[0]
            if not torch.equal(candidate_ids[:prefix_len], base_ids):
                prefix_len = 0
                max_prefix_len = min(base_ids.shape[0], candidate_ids.shape[0])
                while (
                    prefix_len < max_prefix_len
                    and base_ids[prefix_len].item() == candidate_ids[prefix_len].item()
                ):
                    prefix_len += 1
            suffix_ids = candidate_ids[prefix_len:]
            candidates.append((label, candidate_inputs, suffix_ids, prefix_len))
            allowed_tokens.update(suffix_ids.tolist())

        vocab_size = self.model.get_input_embeddings().weight.shape[0]
        label_mask = torch.full((vocab_size,), float("-inf"), device=device)
        label_mask[list(allowed_tokens)] = 0.0

        def score_candidate(candidate_inputs, suffix_ids, prefix_len: int) -> float:
            with torch.inference_mode():
                out = self.model(**candidate_inputs)

            logits = out.logits[0]
            log_probs_sum = 0.0
            for step, token_id in enumerate(suffix_ids.tolist()):
                step_logits = logits[prefix_len - 1 + step]
                filtered = step_logits + label_mask
                step_log_probs = torch.nn.functional.log_softmax(filtered, dim=-1)
                log_probs_sum += step_log_probs[token_id].item()

            return log_probs_sum / max(len(suffix_ids), 1)

        raw_scores = {
            label: score_candidate(candidate_inputs, suffix_ids, prefix_len)
            for label, candidate_inputs, suffix_ids, prefix_len in candidates
        }
        max_score = max(raw_scores.values())
        exp_scores = {
            label: torch.exp(torch.tensor(score - max_score)).item()
            for label, score in raw_scores.items()
        }
        total = sum(exp_scores.values())
        scores = {
            label: round((score / total) * 100, 2)
            for label, score in exp_scores.items()
        }
        predicted = max(scores, key=scores.get)
        return {"logits_label": predicted, "logits_scores": scores}

    @modal.method()
    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        image_url: str | None = None,
        image_base64: str | None = None,
        ringkasan: str = "",
        klaim: str = "",
        fakta: str = "",
        dfk_prompt: str | None = None,
        captioning: bool = False,
        caption_prompt: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
    ) -> dict[str, Any]:
        if image_url and image_base64:
            raise ValueError("Provide either image_url or image_base64, not both.")

        import time as _time
        t_total = _time.time()

        mode = "captioning" if captioning else ("messages" if messages else ("prompt" if prompt else "dfk"))
        img_ref = image_url or ("[base64]" if image_base64 else None)
        print(f"[INPUT] mode={mode} image={img_ref} ringkasan={ringkasan!r} klaim={klaim!r} fakta={fakta!r} prompt={prompt!r} max_new_tokens={max_new_tokens} temperature={temperature}")

        pil_image = None
        if image_url:
            pil_image = _load_image_url(image_url)
        elif image_base64:
            pil_image = _decode_image(image_base64)

        if messages:
            pass
        elif captioning:
            if pil_image is None:
                return {"text": "Error: image_url or image_base64 is required for captioning."}
            content: list[dict[str, Any]] = [
                {"type": "image"},
                {"type": "text", "text": caption_prompt or CAPTIONING_INSTRUCTION},
            ]
        elif prompt:
            content = [{"type": "text", "text": prompt}]
            if pil_image is not None:
                content.append({"type": "image"})
        else:
            content = build_dfk_content(
                ringkasan=ringkasan,
                klaim=klaim,
                fakta=fakta,
                instruction=dfk_prompt,
            )
            if pil_image is not None:
                content.append({"type": "image"})

        if not messages:
            messages = [{"role": "user", "content": content}]

        text = self.processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        images = [pil_image] if pil_image is not None else None
        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        ).to(next(self.model.parameters()).device)

        logits_result = None
        if mode == "dfk":
            logits_result = self._score_logits_labels(text, images)

        generation_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            repetition_penalty=repetition_penalty,
        )
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k
            if min_p > 0:
                generation_kwargs["min_p"] = min_p

        import torch
        import time

        weave_call = None
        try:
            if self._weave_client:
                from datetime import datetime, timezone, timedelta
                wib = timezone(timedelta(hours=7))
                ts = datetime.now(wib).strftime("%Y-%m-%d %H:%M WIB")
                weave_call = self._weave_client.create_call(
                    f"ministral-cpt-8b-ws3-generate | {ts}",
                    inputs={
                        "mode": mode, "image": img_ref,
                        "ringkasan": ringkasan, "klaim": klaim, "fakta": fakta,
                        "prompt": prompt, "dfk_prompt": dfk_prompt,
                        "caption_prompt": caption_prompt,
                        "model_prompt": text,
                        "messages_input": messages if mode == "messages" else None,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                )
        except Exception as e:
            print(f"[WEAVE] create_call failed: {e}")

        t0 = time.time()
        with torch.inference_mode():
            if captioning:
                with self.model.disable_adapter():
                    generated_ids = self.model.generate(**inputs, **generation_kwargs)
            else:
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
        elapsed_ms = int((time.time() - t0) * 1000)

        new_token_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
        output = self.processor.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        result_text = output.strip()
        tokens = new_token_ids.shape[-1]
        total_ms = int((_time.time() - t_total) * 1000)
        print(f"[OUTPUT] tokens={tokens} elapsed_ms={elapsed_ms} total_ms={total_ms} text={result_text!r}")
        if logits_result:
            print(f"[LOGITS] label={logits_result['logits_label']} scores={logits_result['logits_scores']}")

        try:
            if self._weave_client and weave_call:
                out = {"text": result_text, "tokens_generated": tokens, "elapsed_ms": elapsed_ms, "total_request_ms": total_ms}
                if logits_result:
                    out.update(logits_result)
                if self._gpu_warmup_ms is not None:
                    out["gpu_warmup_ms"] = self._gpu_warmup_ms
                    self._gpu_warmup_ms = None
                self._weave_client.finish_call(weave_call, output=out)
        except Exception as e:
            print(f"[WEAVE] finish_call failed: {e}")

        result = {"text": result_text, "tokens_generated": tokens}
        if logits_result:
            result.update(logits_result)
        return result


@app.function(image=image, timeout=600, secrets=[modal.Secret.from_name("wandb-secret")])
@modal.fastapi_endpoint(method="POST", docs=True)
def infer(payload: dict[str, Any]) -> dict[str, Any]:
    import time
    t0 = time.time()
    captioning = bool(payload.get("captioning", False))
    system_prompt = payload.get("system_prompt") or None
    dfk_prompt = (
        payload.get("dfk_prompt")
        or payload.get("dfk_system_prompt")
        or payload.get("dfk_instruction")
        or (system_prompt if not captioning and not payload.get("prompt") else None)
    )
    caption_prompt = (
        payload.get("caption_prompt")
        or payload.get("caption_system_prompt")
        or payload.get("caption_instruction")
        or (system_prompt if captioning else None)
    )
    result = MinistralCPTServer().generate.remote(
        prompt=payload.get("prompt"),
        messages=payload.get("messages") or None,
        image_url=payload.get("image_url"),
        image_base64=payload.get("image_base64"),
        ringkasan=str(payload.get("ringkasan") or payload.get("summary") or ""),
        klaim=str(payload.get("klaim") or payload.get("claim") or ""),
        fakta=str(payload.get("fakta") or payload.get("fact") or ""),
        dfk_prompt=dfk_prompt,
        captioning=captioning,
        caption_prompt=caption_prompt,
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        temperature=float(payload.get("temperature", 0.0)),
        top_p=float(payload.get("top_p", 0.8)),
        top_k=int(payload.get("top_k", 20)),
        min_p=float(payload.get("min_p", 0.0)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
    )
    result["infer_total_ms"] = int((time.time() - t0) * 1000)
    return result


@app.local_entrypoint()
def main(
    prompt: str | None = None,
    image_url: str | None = None,
    ringkasan: str = "",
    klaim: str = "",
    fakta: str = "",
    dfk_prompt: str | None = None,
    captioning: bool = False,
    caption_prompt: str | None = None,
    max_new_tokens: int = 256,
):
    result = MinistralCPTServer().generate.remote(
        prompt=prompt,
        image_url=image_url,
        ringkasan=ringkasan,
        klaim=klaim,
        fakta=fakta,
        dfk_prompt=dfk_prompt,
        captioning=captioning,
        caption_prompt=caption_prompt,
        max_new_tokens=max_new_tokens,
    )
    print(result["text"])

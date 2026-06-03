import base64
import io
from typing import Any

import modal


APP_NAME = "qwen35-ws3-v2"
MODEL_ID = "aitf-komdigi/KomdigiITS-0.8B-DFK-MultimodalClassification"
CACHE_DIR = "/cache/huggingface"
ADAPTER_SUBFOLDER = "adapter"
LABELS = ["NETRAL", "DISINFORMASI", "UJARAN KEBENCIAN", "FITNAH"]


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("qwen35-ws3-v2-cache", create_if_missing=True)

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
) -> list[dict[str, str]]:
    context_parts = []
    if ringkasan.strip():
        context_parts.append(f"Ringkasan: {ringkasan.strip()}")
    if klaim.strip():
        context_parts.append(f"Klaim: {klaim.strip()}")
    if fakta.strip():
        context_parts.append(f"Fakta: {fakta.strip()}")

    content = [{"type": "text", "text": DFK_INSTRUCTION}]
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
class QwenServer:
    @modal.enter(snap=True)
    def load_to_cpu(self):
        import json
        import os
        from concurrent.futures import ThreadPoolExecutor

        token = os.environ.get("HF_TOKEN")

        base_model_id = MODEL_ID
        adapter_model_id = None
        adapter_subfolder = None

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
            adapter_subfolder = ADAPTER_SUBFOLDER
        except Exception:
            pass

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
        import time as _time
        import os
        t0 = _time.time()
        self.model = self.model.to("cuda", dtype=torch.bfloat16)
        self._gpu_warmup_ms = int((_time.time() - t0) * 1000)
        print(f"[COLD_START] move_to_gpu elapsed_ms={self._gpu_warmup_ms}")

        self._weave = None
        self._weave_client = None
        if os.environ.get("WANDB_API_KEY"):
            try:
                import weave
                self._weave_client = weave.init("Aitf-dfk-3/log-qwen-v2")
                self._weave = weave
            except Exception as e:
                print(f"[WEAVE] init failed: {e}")
        else:
            print("[WEAVE] disabled: WANDB_API_KEY is not set")

    def _score_logits_labels(self, inputs) -> dict[str, Any]:
        import torch

        device = inputs["input_ids"].device
        prefix_ids = self.processor.tokenizer.encode("Label: ", add_special_tokens=False)
        prefix_tensor = torch.tensor([prefix_ids], device=device)
        prefix_attn = torch.ones_like(prefix_tensor)
        base_ids = torch.cat([inputs["input_ids"], prefix_tensor], dim=1)
        base_attn = torch.cat([inputs["attention_mask"], prefix_attn], dim=1)

        config = self.model.config
        vocab_size = (
            config.vocab_size
            if hasattr(config, "vocab_size")
            else config.text_config.vocab_size
        )
        label_mask = torch.full((vocab_size,), float("-inf"), device=device)
        label_mask[self._label_all_tokens] = 0.0

        def score_label(label_token_ids: list[int]) -> float:
            label_tensor = torch.tensor([label_token_ids], device=device)
            combined_ids = torch.cat([base_ids, label_tensor], dim=1)
            combined_attn = torch.cat(
                [base_attn, torch.ones_like(label_tensor)],
                dim=1,
            )
            model_inputs = dict(inputs)
            model_inputs["input_ids"] = combined_ids
            model_inputs["attention_mask"] = combined_attn

            with torch.inference_mode():
                out = self.model(**model_inputs)

            logits = out.logits[0]
            prompt_len = base_ids.shape[1]
            log_probs_sum = 0.0
            for step, token_id in enumerate(label_token_ids):
                step_logits = logits[prompt_len - 1 + step]
                filtered = step_logits + label_mask
                step_log_probs = torch.nn.functional.log_softmax(filtered, dim=-1)
                log_probs_sum += step_log_probs[token_id].item()

            return log_probs_sum / len(label_token_ids)

        raw_scores = {
            label: score_label(token_ids)
            for label, token_ids in zip(LABELS, self._label_token_seqs)
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
        image_url: str | None = None,
        image_base64: str | None = None,
        ringkasan: str = "",
        klaim: str = "",
        fakta: str = "",
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

        mode = "captioning" if captioning else ("prompt" if prompt else "dfk")
        img_ref = image_url or ("[base64]" if image_base64 else None)
        print(f"[INPUT] mode={mode} image={img_ref} ringkasan={ringkasan!r} klaim={klaim!r} fakta={fakta!r} prompt={prompt!r} max_new_tokens={max_new_tokens} temperature={temperature}")

        pil_image = None
        if image_url:
            pil_image = _load_image_url(image_url)
        elif image_base64:
            pil_image = _decode_image(image_base64)

        if captioning:
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
            )
            if pil_image is not None:
                content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
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
            logits_result = self._score_logits_labels(inputs)

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
                    f"qwen35-ws3-v2-generate | {ts}",
                    inputs={
                        "mode": mode, "image": img_ref,
                        "ringkasan": ringkasan, "klaim": klaim, "fakta": fakta,
                        "prompt": prompt, "model_prompt": text,
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
            print(
                f"[LOGITS] label={logits_result['logits_label']} "
                f"scores={logits_result['logits_scores']}"
            )

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
    result = QwenServer().generate.remote(
        prompt=payload.get("prompt"),
        image_url=payload.get("image_url"),
        image_base64=payload.get("image_base64"),
        ringkasan=str(payload.get("ringkasan") or payload.get("summary") or ""),
        klaim=str(payload.get("klaim") or payload.get("claim") or ""),
        fakta=str(payload.get("fakta") or payload.get("fact") or ""),
        captioning=bool(payload.get("captioning", False)),
        caption_prompt=payload.get("caption_prompt") or None,
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
    captioning: bool = False,
    caption_prompt: str | None = None,
    max_new_tokens: int = 256,
):
    result = QwenServer().generate.remote(
        prompt=prompt,
        image_url=image_url,
        ringkasan=ringkasan,
        klaim=klaim,
        fakta=fakta,
        captioning=captioning,
        caption_prompt=caption_prompt,
        max_new_tokens=max_new_tokens,
    )
    print(result["text"])

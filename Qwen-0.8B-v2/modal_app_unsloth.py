import base64
import io
from typing import Any

import modal


APP_NAME = "qwen35-ws3-v2"
MODEL_ID = "aitf-komdigi/KomdigiITS-0.8B-DFK-MultimodalClassification"
CACHE_DIR = "/cache/huggingface"
ADAPTER_SUBFOLDER = "adapter"


app = modal.App(APP_NAME)

hf_cache = modal.Volume.from_name("qwen35-ws3-v2-cache", create_if_missing=True)

FLASH_ATTN_WHEEL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "ninja-build", "libz3-dev")
    .pip_install(
        "torch==2.8.0",
        "triton>=3.3.0",
        "torchvision",
        "bitsandbytes",
        "xformers==0.0.32.post2",
        "ninja",
        "packaging",
        "fastapi[standard]",
        "huggingface_hub[hf_transfer]",
        "pillow",
        "requests",
        FLASH_ATTN_WHEEL,
    )
    .run_commands(
        "pip install -q --no-deps 'unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo'",
        "pip install -q --no-deps 'unsloth[base] @ git+https://github.com/unslothai/unsloth'",
        "pip install -q 'transformers==5.2.0' 'tokenizers>=0.22.0,<=0.23.0' 'trl==0.22.2' unsloth unsloth_zoo",
        "pip install -q --no-build-isolation flash-linear-attention 'causal_conv1d==1.6.0'",
        "pip install -q --no-deps 'apache-tvm-ffi==0.1.9' 'tilelang==0.1.8'",
        "pip install -q --no-deps --upgrade 'torchao>=0.16.0'",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_HOME": CACHE_DIR,
            "TRANSFORMERS_CACHE": CACHE_DIR,
        }
    )
)


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
    "Anda adalah asisten yang membantu menganalisis gambar. "
    "Pengguna akan memberikan URL gambar. "
    "Tugas Anda: "
    "1. Salin SEMUA teks yang terlihat di gambar secara persis (verbatim). "
    "2. Deskripsikan secara singkat konteks visual gambar (siapa, di mana, apa yang terjadi). "
    "Format: tulis teks yang terlihat terlebih dahulu, lalu deskripsi visual. "
    "Gunakan Bahasa Indonesia. JANGAN tambahkan pembukaan apapun."
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
)
class QwenServer:
    @modal.enter()
    def load_model(self):
        import os
        from unsloth import FastVisionModel

        token = os.environ.get("HF_TOKEN")

        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_name=MODEL_ID,
            load_in_4bit=False,
            token=token,
        )
        FastVisionModel.for_inference(self.model)

    def _build_inputs(self, content: list[dict], pil_image=None):
        messages = [{"role": "user", "content": content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if pil_image is not None:
            inputs = self.tokenizer(
                images=pil_image,
                text=text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(
                text=text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self.model.device)
        return inputs

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
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 20,
        min_p: float = 0.0,
        presence_penalty: float = 2.0,
        repetition_penalty: float = 1.0,
    ) -> dict[str, Any]:
        import torch

        if image_url and image_base64:
            raise ValueError("Provide either image_url or image_base64, not both.")

        pil_image = None
        if image_url:
            pil_image = _load_image_url(image_url)
        elif image_base64:
            pil_image = _decode_image(image_base64)

        generation_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
        )

        if captioning:
            if pil_image is None:
                return {"text": "Error: image_url or image_base64 is required for captioning."}
            content: list[dict[str, Any]] = [
                {"type": "text", "text": caption_prompt or CAPTIONING_INSTRUCTION},
                {"type": "image"},
            ]
            inputs = self._build_inputs(content, pil_image)
            with torch.inference_mode():
                with self.model.disable_adapter():
                    generated_ids = self.model.generate(**inputs, **generation_kwargs)
        else:
            if prompt:
                content = [{"type": "text", "text": prompt}]
            else:
                content = build_dfk_content(
                    ringkasan=ringkasan,
                    klaim=klaim,
                    fakta=fakta,
                )
            if pil_image is not None:
                content.append({"type": "image"})
            inputs = self._build_inputs(content, pil_image)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **generation_kwargs)

        new_token_ids = generated_ids[:, inputs["input_ids"].shape[-1]:]
        output = self.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"text": output.strip(), "tokens_generated": new_token_ids.shape[-1]}


@app.function(image=image, timeout=600)
@modal.fastapi_endpoint(method="POST", docs=True)
def infer(payload: dict[str, Any]) -> dict[str, Any]:
    return QwenServer().generate.remote(
        prompt=payload.get("prompt"),
        image_url=payload.get("image_url"),
        image_base64=payload.get("image_base64"),
        ringkasan=str(payload.get("ringkasan") or ""),
        klaim=str(payload.get("klaim") or ""),
        fakta=str(payload.get("fakta") or ""),
        captioning=bool(payload.get("captioning", False)),
        caption_prompt=payload.get("caption_prompt") or None,
        max_new_tokens=int(payload.get("max_new_tokens", 128)),
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 1.0)),
        top_k=int(payload.get("top_k", 20)),
        min_p=float(payload.get("min_p", 0.0)),
        presence_penalty=float(payload.get("presence_penalty", 2.0)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
    )


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

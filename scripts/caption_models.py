"""Image-only captioners. Loading a backend once returns a reusable callable."""

from collections.abc import Callable
from pathlib import Path

MODEL_IDS = {
    "molmo": "allenai/Molmo-7B-O-0924",
    "blip": "Salesforce/blip-image-captioning-base",
}
MODEL_REVISIONS = {
    "molmo": "7a8c4bf80c839c243a6908c6ebbb0f1ee576d7ca",
    "blip": "82a37760796d32b1411fe092ab5d4e227313294b",
}
# BLIP is an unconditional captioner; only Molmo receives this instruction.
PROMPT = (
    "Describe this artwork in English in two or three sentences. Focus on the "
    "visible subjects, actions, colors, lighting, and composition. If no subject "
    "is recognizable, describe shapes and contrast instead. Do not guess "
    "the artist, title, or viewer's emotional response."
)


def load_captioner(backend: str) -> Callable[[Path], str]:
    """Load the selected model on CUDA if available, otherwise on CPU."""
    if backend not in MODEL_IDS:
        raise ValueError(f"Unknown backend {backend!r}; choose one of {', '.join(MODEL_IDS)}")

    import torch
    from PIL import Image, ImageOps

    model_id = MODEL_IDS[backend]
    revision = MODEL_REVISIONS[backend]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if backend == "molmo":
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

        # Molmo uses custom processor and model code from the same pinned repo.
        source = dict(revision=revision, code_revision=revision, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_id, **source)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, **source, device_map={"": device}, torch_dtype=dtype
        ).eval()
        generation = GenerationConfig(max_new_tokens=128, do_sample=False, stop_strings="<|endoftext|>")

        def generate(image: Image.Image) -> str:
            single = processor.process(images=[image], text=PROMPT)
            batch = {
                key: value.unsqueeze(0).to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                for key, value in single.items()
            }
            output = model.generate_from_batch(batch, generation, tokenizer=processor.tokenizer)
            new_tokens = output[0, batch["input_ids"].shape[1]:]
            return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    else:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(model_id, revision=revision)
        model = BlipForConditionalGeneration.from_pretrained(
            model_id, revision=revision, torch_dtype=dtype
        ).to(device).eval()

        def generate(image: Image.Image) -> str:
            pixels = processor(images=image, return_tensors="pt").pixel_values.to(device=device, dtype=dtype)
            output = model.generate(pixel_values=pixels, max_new_tokens=64, do_sample=False)
            return processor.decode(output[0], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def caption(path: Path) -> str:
        with Image.open(path) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        return generate(image)

    return caption

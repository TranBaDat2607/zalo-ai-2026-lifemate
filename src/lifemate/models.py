"""Model loading shared by training and inference."""
import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

DEFAULT_LLM = "Qwen/Qwen3.5-4B"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_and_dtype():
    # Kaggle's T4 has no native bf16, so fp16 is the GPU dtype here.
    if torch.cuda.is_available():
        return "cuda", torch.float16
    # CPU is only for smoke tests. float32 needs 17 GB for the LLM; LIFEMATE_CPU_DTYPE=bfloat16 halves that.
    return "cpu", getattr(torch, os.environ.get("LIFEMATE_CPU_DTYPE", "float32"))


def load_llm(name: str, adapter: str = None):
    device, dtype = device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(name)
    kwargs = dict(dtype=dtype, device_map={"": device})
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except ValueError:  # multimodal checkpoints only register under the image-text auto class
        model = AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return tokenizer, model


def count_parameters(model) -> dict:
    """Total, and without any vision tower that the agent never runs."""
    total = sum(p.numel() for p in model.parameters())
    vision = sum(p.numel() for n, p in model.named_parameters() if "visual" in n or "vision" in n)
    return {"total": total, "vision": vision, "text": total - vision}

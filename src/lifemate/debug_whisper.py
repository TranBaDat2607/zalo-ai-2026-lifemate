"""Compare Whisper decoding paths on a few clips, to find why the short-input path gave garbage. Runs on CPU.

    python -m lifemate.debug_whisper --out outputs/debug
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration

from . import asr, data


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/debug")
    ap.add_argument("--model", default="vinai/PhoWhisper-tiny")
    ap.add_argument("--clips", type=int, default=4)
    args = ap.parse_args(argv)
    root = data.find_root("train")
    items = data.load_split("train", root)[:args.clips]
    audio = [asr.read_audio(str(Path(root) / "train" / i["audio"])) for i in items]
    refs = [i["audio_text"] for i in items]
    processor = AutoProcessor.from_pretrained(args.model)
    tok = processor.tokenizer
    tok.set_prefix_tokens(language="vi", task="transcribe")
    print("prefix", tok.prefix_tokens, tok.convert_ids_to_tokens(tok.prefix_tokens), "eos", tok.eos_token_id, flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32).eval()
    print("generation suppress", model.generation_config.suppress_tokens, model.generation_config.begin_suppress_tokens,
          "forced", getattr(model.generation_config, "forced_decoder_ids", None), flush=True)

    full = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    print("full features", full.shape, float(full.min()), float(full.max()), flush=True)
    with torch.inference_mode():
        ids = model.generate(input_features=full, language="vi", task="transcribe", do_sample=False, num_beams=1, max_new_tokens=100)
    out = {"ref": refs, "generate_full": processor.batch_decode(ids, skip_special_tokens=True)}
    print("generate ids", ids[0][:12].tolist(), flush=True)

    raw = asr.greedy_decode(model, full, tok.prefix_tokens, tok.eos_token_id, asr.token_limits(audio))
    print("greedy full ids", raw[0][:12], flush=True)
    out["greedy_full"] = processor.batch_decode(raw, skip_special_tokens=True)

    short = asr.log_mel(processor.feature_extractor, audio)
    print("short features", short.shape, float(short.min()), float(short.max()), flush=True)
    asr.enable_short_context(model)
    raw = asr.greedy_decode(model, short, tok.prefix_tokens, tok.eos_token_id, asr.token_limits(audio))
    print("greedy short ids", raw[0][:12], flush=True)
    out["greedy_short"] = processor.batch_decode(raw, skip_special_tokens=True)
    enc = model.get_encoder()
    print("encoder positions", enc.embed_positions.num_embeddings, enc.embed_positions.weight.dtype, float(enc.embed_positions.weight.abs().mean()),
          "full table", float(enc._full_positions.weight.abs().mean()), flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

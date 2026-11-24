#!/usr/bin/env python3
"""Qari v2 isolated experiments on the Visual Benchmark crops (RunPod /workspace).

Experiment A (structured): structured label|value prompt, greedy, max_new_tokens=4096.
Experiment B (antirepeat): "Free OCR." greedy (4096); if a repetition loop is suspected
(token limit hit, or a short token sequence repeated >= 5 times consecutively), the same
image is rerun exactly once with repetition_penalty=1.1, max_new_tokens=2048, greedy.

Inputs (read only):  /workspace/benchmark_ar/crops (20), /workspace/benchmark_en/crops (19)
Outputs (new only):  /workspace/results_v2_structured/{AR,EN}/, /workspace/results_v2_antirepeat/{AR,EN}/
Raw model text is saved unchanged (UTF-8, no trimming). Nothing existing is overwritten.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
ADAPTER = "NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct"
WORKSPACE = Path("/workspace")
LANGUAGES = {"AR": (WORKSPACE / "benchmark_ar" / "crops", 20), "EN": (WORKSPACE / "benchmark_en" / "crops", 19)}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

ORIGINAL_PROMPT = "Free OCR."
STRUCTURED_PROMPT = (
    "Transcribe all text in this image exactly as printed, in its natural reading order.\n"
    "If the image is a chart, KPI card, form, or table, also output one line per label and value pair, "
    "in this format:\n"
    "label | value\n"
    "Only output numbers that are visibly printed in the image. Do not estimate values from bar heights, "
    "pie slices, or positions. If a label has no printed numeric value, leave the value empty.\n"
    "Output only the transcription, without commentary."
)
GREEDY = {"do_sample": False, "max_new_tokens": 4096}
RETRY = {"do_sample": False, "max_new_tokens": 2048, "repetition_penalty": 1.1}
LOOP_MIN_REPEATS = 5
LOOP_MAX_PATTERN_TOKENS = 20

VARIANTS = {
    "structured": {"root": WORKSPACE / "results_v2_structured", "prompt": STRUCTURED_PROMPT},
    "antirepeat": {"root": WORKSPACE / "results_v2_antirepeat", "prompt": ORIGINAL_PROMPT},
}


# ------------------------------------------------------------------ validation


def crops(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def validate() -> dict[str, list[Path]]:
    found = {}
    for language, (folder, expected) in LANGUAGES.items():
        assert folder.is_dir(), f"missing crop folder {folder}"
        images = crops(folder)
        assert len(images) == expected, f"{language}: expected {expected} crops in {folder}, found {len(images)}"
        ids = [p.stem for p in images]
        assert len(set(ids)) == len(ids), f"{language}: region IDs are not unique: {sorted(ids)}"
        found[language] = images
    for variant in VARIANTS.values():
        for language in LANGUAGES:
            target = variant["root"] / language
            if target.exists() and any(target.iterdir()):
                raise SystemExit(f"REFUSED: {target} already exists and contains files; nothing was run")
    return found


# -------------------------------------------------------------------- helpers


def write_new(path: Path, text: str) -> None:
    """Write UTF-8 exactly as given; never overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def repeated_pattern(tokens: list[int]) -> dict | None:
    """A short token sequence (1..20 tokens) repeated >= 5 times consecutively."""
    count = len(tokens)
    for size in range(1, LOOP_MAX_PATTERN_TOKENS + 1):
        need = size * LOOP_MIN_REPEATS
        if need > count:
            break
        for start in range(0, count - need + 1):
            pattern = tokens[start:start + size]
            if all(tokens[start + k * size:start + (k + 1) * size] == pattern for k in range(1, LOOP_MIN_REPEATS)):
                return {"pattern_tokens": size, "start_token": start, "repeats_at_least": LOOP_MIN_REPEATS}
    return None


# ----------------------------------------------------------------------- model


def load_model(args):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    base = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, revision=args.base_revision, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER, revision=args.adapter_revision)
    model.eval()
    processor = AutoProcessor.from_pretrained(BASE_MODEL, revision=args.base_revision)
    return model, processor


def generate(model, processor, image: Path, prompt: str, params: dict) -> dict:
    import torch

    messages = [{"role": "user", "content": [{"type": "image", "image": str(image)},
                                             {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **params)
    seconds = time.perf_counter() - started
    generated = output[0][inputs["input_ids"].shape[1]:].tolist()
    text = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    del inputs, output
    torch.cuda.empty_cache()
    return {"text": text, "tokens": len(generated), "seconds": round(seconds, 3), "token_ids": generated}


def loop_check(result: dict, params: dict) -> dict:
    hit_limit = result["tokens"] >= params["max_new_tokens"]
    pattern = repeated_pattern(result["token_ids"])
    return {"suspected": bool(hit_limit or pattern), "hit_token_limit": hit_limit, "repetition": pattern}


# ---------------------------------------------------------------- experiments


def header(variant: str, language: str, prompt: str, args, retry: dict | None) -> dict:
    import torch
    import transformers

    return {"variant": variant, "language": language, "base_model": BASE_MODEL, "adapter": ADAPTER,
            "base_revision": args.base_revision or "default branch", "adapter_revision": args.adapter_revision
            or "default branch", "prompt": prompt, "generation": GREEDY, "retry_settings": retry,
            "loop_rule": (f"token limit hit, or a sequence of 1..{LOOP_MAX_PATTERN_TOKENS} tokens repeated "
                          f">= {LOOP_MIN_REPEATS} times consecutively") if retry else None,
            "started_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
            "platform": platform.platform(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "regions": []}


def run_structured(model, processor, language: str, images: list[Path], args) -> None:
    out = VARIANTS["structured"]["root"] / language
    log = header("structured", language, STRUCTURED_PROMPT, args, None)
    for index, image in enumerate(images, start=1):
        rid = image.stem
        result = generate(model, processor, image, STRUCTURED_PROMPT, GREEDY)
        write_new(out / f"{rid}.txt", result["text"])
        check = loop_check(result, GREEDY)
        entry = {"region_id": rid, "source_image": str(image), "tokens": result["tokens"], "seconds": result["seconds"],
                 "params": GREEDY, "loop_suspected": check["suspected"], "loop_check": check, "output": f"{rid}.txt"}
        log["regions"].append(entry)
        append_jsonl(out / "run_log.partial.jsonl", entry)
        print(f"[structured {language} {index}/{len(images)}] {rid}: {result['tokens']} tokens, "
              f"{result['seconds']} s, loop suspected={check['suspected']}", flush=True)
    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_new(out / "run_log.json", json.dumps(log, ensure_ascii=False, indent=2))


def run_antirepeat(model, processor, language: str, images: list[Path], args) -> None:
    out = VARIANTS["antirepeat"]["root"] / language
    log = header("antirepeat", language, ORIGINAL_PROMPT, args, RETRY)
    for index, image in enumerate(images, start=1):
        rid = image.stem
        first = generate(model, processor, image, ORIGINAL_PROMPT, GREEDY)
        write_new(out / f"{rid}.attempt1.txt", first["text"])
        check = loop_check(first, GREEDY)
        entry = {"region_id": rid, "source_image": str(image), "loop_suspected": check["suspected"],
                 "attempt1": {"file": f"{rid}.attempt1.txt", "tokens": first["tokens"], "seconds": first["seconds"],
                              "params": GREEDY, "loop_check": check},
                 "attempt2": None, "final": "attempt1"}
        final_text = first["text"]
        if check["suspected"]:
            second = generate(model, processor, image, ORIGINAL_PROMPT, RETRY)
            write_new(out / f"{rid}.attempt2.txt", second["text"])
            entry["attempt2"] = {"file": f"{rid}.attempt2.txt", "tokens": second["tokens"],
                                 "seconds": second["seconds"], "params": RETRY,
                                 "loop_check": loop_check(second, RETRY)}
            entry["final"] = "attempt2"
            final_text = second["text"]
        write_new(out / f"{rid}.txt", final_text)
        entry["final_file"] = f"{rid}.txt"
        entry["tokens"] = (entry["attempt2"] or entry["attempt1"])["tokens"]
        entry["seconds"] = round(first["seconds"] + ((entry["attempt2"] or {}).get("seconds") or 0.0), 3)
        entry["params"] = (entry["attempt2"] or entry["attempt1"])["params"]
        log["regions"].append(entry)
        append_jsonl(out / "run_log.partial.jsonl", entry)
        print(f"[antirepeat {language} {index}/{len(images)}] {rid}: loop suspected={check['suspected']}, "
              f"final={entry['final']}", flush=True)
    log["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_new(out / "run_log.json", json.dumps(log, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-revision", default=None, help="optional pinned base model revision")
    parser.add_argument("--adapter-revision", default=None, help="optional pinned adapter revision")
    args = parser.parse_args()
    images = validate()  # before loading the model
    print(f"Validated: AR {len(images['AR'])} crops, EN {len(images['EN'])} crops; output folders are new.", flush=True)
    model, processor = load_model(args)
    for language in LANGUAGES:
        run_structured(model, processor, language, images[language], args)
    for language in LANGUAGES:
        run_antirepeat(model, processor, language, images[language], args)
    print("ALL DONE. Zip /workspace/results_v2_structured and /workspace/results_v2_antirepeat and send them.")


if __name__ == "__main__":
    main()

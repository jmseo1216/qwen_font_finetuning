from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.qwen_font_pipeline.prompting import build_messages, format_prompt
from src.qwen_font_pipeline.svg_utils import extract_svg_path, path_to_svg, repair_svg_path, validate_svg_path


def load_base_model(adapter_dir: Path, override: str | None) -> str:
    if override:
        return override
    meta = adapter_dir / "train_meta.json"
    if not meta.exists():
        raise ValueError("Provide --model_name_or_path or use an adapter directory containing train_meta.json")
    return json.loads(meta.read_text(encoding="utf-8"))["base_model"]


def parse_generated_path(text: str, repair: bool = False) -> str:
    candidate = text.strip().replace("<|im_end|>", "").strip()
    try:
        validate_svg_path(candidate)
        return candidate
    except ValueError:
        if not repair:
            raise
        return repair_svg_path(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a skeleton path from an outline SVG using a Qwen LoRA adapter.")
    parser.add_argument("--input_svg", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter_dir", required=True, type=Path)
    parser.add_argument("--model_name_or_path")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--device_map", default="auto")
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    outline = extract_svg_path(args.input_svg)
    base_model = load_base_model(args.adapter_dir, args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quantization, device_map=args.device_map)
    model = PeftModel.from_pretrained(model, args.adapter_dir).eval()
    prompt = tokenizer.apply_chat_template(build_messages(outline.path), tokenize=False, add_generation_prompt=True) if hasattr(tokenizer, "apply_chat_template") else format_prompt(outline.path)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    answer_ids = generated[0, encoded["input_ids"].shape[1]:]
    output_path = parse_generated_path(tokenizer.decode(answer_ids, skip_special_tokens=False), args.repair)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(path_to_svg(output_path, outline.view_box) if args.output.suffix.lower() == ".svg" else output_path + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from dataset.qwen_causal_dataset import AssistantOnlyDataCollator, QwenFontDataset


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen with 4-bit QLoRA for outline-to-skeleton conversion.")
    parser.add_argument("--config", default="configs/qwen2.5-coder-1.5b-qlora.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=cfg.get("trust_remote_code", False))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], quantization_config=quantization, device_map="auto", torch_dtype=dtype, trust_remote_code=cfg.get("trust_remote_code", False))
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora = LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg.get("lora_dropout", 0.05), bias="none", task_type="CAUSAL_LM", target_modules=cfg["lora_target_modules"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    train_ds = QwenFontDataset(cfg["train_jsonl"], tokenizer, cfg["max_seq_length"], cfg.get("strict_length", False))
    val_ds = QwenFontDataset(cfg["val_jsonl"], tokenizer, cfg["max_seq_length"], cfg.get("strict_length", False))
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"], num_train_epochs=cfg["epochs"], learning_rate=float(cfg["learning_rate"]),
        per_device_train_batch_size=cfg["batch_size"], per_device_eval_batch_size=cfg.get("eval_batch_size", 1),
        gradient_accumulation_steps=cfg["grad_accumulation_steps"], gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False), logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps"), eval_steps=cfg.get("eval_steps", 100), save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3), warmup_ratio=cfg.get("warmup_ratio", 0.03),
        optim=cfg.get("optim", "paged_adamw_8bit"), report_to=cfg.get("report_to", "none"), remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=AssistantOnlyDataCollator(tokenizer))
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["output_dir"]) / "train_meta.json").write_text(json.dumps({"base_model": cfg["model_name_or_path"], "config": cfg}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

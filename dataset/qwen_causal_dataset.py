from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.qwen_font_pipeline.prompting import build_messages, format_prompt


class QwenFontDataset(Dataset):
    """Tokenize ChatML examples and mask prompt labels for assistant-only loss."""
    def __init__(self, jsonl_path: str | Path, tokenizer: Any, max_seq_length: int = 2048, strict_length: bool = False):
        self.tokenizer, self.max_seq_length, self.strict_length = tokenizer, max_seq_length, strict_length
        self.rows = [json.loads(line) for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def _render(self, row: dict) -> tuple[str, str]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(build_messages(row["input_path"]), tokenize=False, add_generation_prompt=True)
            full = self.tokenizer.apply_chat_template(build_messages(row["input_path"], row["target_path"]), tokenize=False, add_generation_prompt=False)
            return prompt, full
        return format_prompt(row["input_path"]), format_prompt(row["input_path"], row["target_path"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt, full = self._render(self.rows[index])
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = self.tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(input_ids) > self.max_seq_length and self.strict_length:
            raise ValueError(f"Example {self.rows[index].get('id', index)} has {len(input_ids)} tokens; max_seq_length={self.max_seq_length}")
        input_ids = input_ids[: self.max_seq_length]
        labels = input_ids.copy()
        for pos in range(min(len(prompt_ids), len(labels))):
            labels[pos] = -100
        if not any(token != -100 for token in labels):
            raise ValueError(
                f"Example {self.rows[index].get('id', index)} leaves no assistant tokens after truncation; "
                "increase max_seq_length or shorten the input path"
            )
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.ones(len(input_ids), dtype=torch.long), "labels": torch.tensor(labels)}


class AssistantOnlyDataCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int | None = 8):
        self.tokenizer, self.pad_to_multiple_of = tokenizer, pad_to_multiple_of

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        def padded(key: str, value: int) -> torch.Tensor:
            return torch.stack([torch.cat([item[key], torch.full((max_len - len(item[key]),), value, dtype=item[key].dtype)]) for item in features])
        return {"input_ids": padded("input_ids", self.tokenizer.pad_token_id), "attention_mask": padded("attention_mask", 0), "labels": padded("labels", -100)}

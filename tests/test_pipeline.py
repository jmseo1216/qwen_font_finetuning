import json
from pathlib import Path

import pytest

from preprocess.build_dataset import infer_metadata, split_rows
from src.qwen_font_pipeline.prompting import format_prompt
from src.qwen_font_pipeline.svg_utils import extract_svg_path, repair_svg_path, validate_svg_path


class TinyTokenizer:
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text.split()) + 1))}


def test_validate_and_repair_svg_path():
    assert validate_svg_path("M 0 0 L 10 10 Z")
    assert repair_svg_path("```svg\nM 0 0 L 10 10 junk\n```") == "M 0 0 L 10 10"
    with pytest.raises(ValueError):
        validate_svg_path("L 0 0")


def test_extract_svg_path(tmp_path: Path):
    svg = tmp_path / "font_65.svg"
    svg.write_text('<svg viewBox="0 0 20 30"><path d="M0,0 L10,10"/></svg>', encoding="utf-8")
    result = extract_svg_path(svg)
    assert result.path == "M0,0 L10,10"
    assert (result.width, result.height) == (20.0, 30.0)


def test_prompt_and_assistant_masking(tmp_path: Path):
    pytest.importorskip("torch")
    from dataset.qwen_causal_dataset import AssistantOnlyDataCollator, QwenFontDataset
    path = tmp_path / "sample.jsonl"
    path.write_text(json.dumps({"input_path": "M 0 0 L 1 1", "target_path": "M 0 0"}) + "\n", encoding="utf-8")
    dataset = QwenFontDataset(path, TinyTokenizer(), max_seq_length=100)
    item = dataset[0]
    prompt_len = len(TinyTokenizer()(format_prompt("M 0 0 L 1 1"))["input_ids"])
    assert item["labels"][:prompt_len].tolist() == [-100] * prompt_len
    assert any(value != -100 for value in item["labels"].tolist())
    batch = AssistantOnlyDataCollator(TinyTokenizer())([item, item])
    assert batch["input_ids"].shape[1] % 8 == 0


def test_holdout_split_and_filename_metadata():
    assert infer_metadata("Roboto_65") == ("Roboto", 65)
    rows = [{"id": f"{font}_{code}", "font": font, "code": code} for font in ["A", "B", "C"] for code in [65, 66, 67]]
    train, val, test = split_rows(rows, "font", 0.1, 0.34, 42)
    assert test and all(row["generalization_split"] == "unseen" for row in test)
    assert not ({row["font"] for row in train} & {row["font"] for row in test})

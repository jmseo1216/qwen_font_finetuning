from __future__ import annotations

SYSTEM_PROMPT = (
    "You convert font outline SVG path into skeleton SVG path. "
    "Output only valid SVG path commands."
)


def build_messages(input_path: str, target_path: str | None = None) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Outline path:\n{input_path.strip()}"},
    ]
    if target_path is not None:
        messages.append({"role": "assistant", "content": target_path.strip()})
    return messages


def format_prompt(input_path: str, target_path: str | None = None) -> str:
    """Render Qwen's ChatML format without requiring transformers at import time."""
    chunks = []
    for message in build_messages(input_path, target_path):
        chunks.append(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n")
    if target_path is None:
        chunks.append("<|im_start|>assistant\n")
    return "".join(chunks)

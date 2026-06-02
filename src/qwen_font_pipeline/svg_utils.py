from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

COMMANDS = "MmLlHhVvCcSsQqTtAaZz"
COMMAND_RE = re.compile(f"[{COMMANDS}]")
TOKEN_RE = re.compile(rf"([{COMMANDS}])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
ALLOWED_RE = re.compile(rf"^[\s,0-9eE+\-.{COMMANDS}]+$")
ARITY = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7, "Z": 0}


@dataclass(frozen=True)
class SVGPath:
    path: str
    width: float
    height: float
    view_box: str


def extract_svg_path(svg_file: str | Path) -> SVGPath:
    """Extract and concatenate path data from an SVG while preserving its canvas."""
    path = Path(svg_file)
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not parse SVG file {path}: {exc}") from exc
    view_box = root.attrib.get("viewBox", "0 0 1024 1024")
    parts = view_box.replace(",", " ").split()
    if len(parts) != 4:
        raise ValueError(f"Invalid viewBox in {path}: {view_box!r}")
    try:
        width, height = float(parts[2]), float(parts[3])
    except ValueError as exc:
        raise ValueError(f"Non-numeric viewBox in {path}: {view_box!r}") from exc
    paths = [elem.attrib["d"].strip() for elem in root.iter() if elem.tag.endswith("path") and elem.attrib.get("d")]
    if not paths:
        raise ValueError(f"No SVG path with a d attribute found in {path}")
    joined = " ".join(paths)
    validate_svg_path(joined)
    return SVGPath(joined, width, height, view_box)


def tokenize_svg_path(path_d: str) -> list[str]:
    return [command or number for command, number in TOKEN_RE.findall(path_d)]


def validate_svg_path(path_d: str) -> bool:
    """Perform dependency-free syntax checks for generated SVG path data."""
    if not path_d or not path_d.strip():
        raise ValueError("SVG path is empty")
    if not ALLOWED_RE.fullmatch(path_d.strip()):
        raise ValueError("SVG path contains unsupported characters")
    tokens = tokenize_svg_path(path_d)
    if not tokens or tokens[0] not in "Mm":
        raise ValueError("SVG path must start with M or m")
    command: str | None = None
    numbers = 0
    for token in tokens + ["END"]:
        if token == "END" or COMMAND_RE.fullmatch(token):
            if command is not None:
                arity = ARITY[command.upper()]
                if arity == 0 and numbers != 0:
                    raise ValueError(f"Command {command} does not accept coordinates")
                if arity and (numbers < arity or numbers % arity != 0):
                    raise ValueError(f"Command {command} received {numbers} coordinates; expected a multiple of {arity}")
            if token == "END":
                break
            command, numbers = token, 0
        else:
            if command is None:
                raise ValueError("SVG coordinates must follow a command")
            numbers += 1
    return True


def repair_svg_path(text: str) -> str:
    """Apply conservative repair: strip wrappers and truncate an incomplete final command."""
    text = text.strip().replace("```svg", "").replace("```", "").strip()
    match = re.search(r"[Mm]", text)
    if not match:
        raise ValueError("Cannot repair SVG path without a move command")
    candidate = text[match.start():]
    candidate = re.sub(rf"[^\s,0-9eE+\-.{COMMANDS}].*$", "", candidate).strip(" ,")
    tokens = tokenize_svg_path(candidate)
    while tokens:
        repaired = " ".join(tokens)
        try:
            validate_svg_path(repaired)
            return repaired
        except ValueError:
            tokens.pop()
    raise ValueError("Could not repair generated SVG path")


def path_to_svg(path_d: str, view_box: str = "0 0 1024 1024") -> str:
    validate_svg_path(path_d)
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}"><path d="{path_d}" fill="none" stroke="black"/></svg>\n'


def command_sequence(path_d: str) -> list[str]:
    return COMMAND_RE.findall(path_d)

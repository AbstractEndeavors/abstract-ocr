from __future__ import annotations
from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable
from abstract_ocr import *
from abstract_utilities import *
from abstract_apis import *


#!/usr/bin/env python3
"""
abstract_ocr.ocr_utils.paddle_manager
-------------------------------------
Provides a CPU-only, cached PaddleOCR interface usable across all modules.
"""



@dataclass
class OCRToken:
    text: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _to_int_box(box: Any) -> tuple[int, int, int, int]:
    """
    Converts PaddleOCR rec_boxes row to x1, y1, x2, y2.
    Handles numpy arrays/lists.
    """
    values = list(map(int, box))
    if len(values) != 4:
        raise ValueError(f"Expected box of length 4, got: {values}")

    x1, y1, x2, y2 = values
    return x1, y1, x2, y2


def extract_tokens(
    paddle_result: list[dict[str, Any]],
    min_score: float = 0.50,
) -> list[OCRToken]:
    """
    Extract OCR tokens from PaddleOCR 3.x predict()/ocr() result.
    """
    tokens: list[OCRToken] = []

    for page in paddle_result:
        texts = page.get("rec_texts") or []
        scores = page.get("rec_scores") or []
        boxes = page.get("rec_boxes",[])

        for text, score, box in zip(texts, scores, boxes):
            if not text or float(score) < min_score:
                continue

            x1, y1, x2, y2 = _to_int_box(box)

            tokens.append(
                OCRToken(
                    text=str(text).strip(),
                    score=float(score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return tokens


def group_tokens_into_lines(
    tokens: list[OCRToken],
    y_tolerance: int | None = None,
) -> list[list[OCRToken]]:
    """
    Groups OCR tokens into visual text lines using their y-center.
    """
    if not tokens:
        return []

    tokens = sorted(tokens, key=lambda t: (t.cy, t.x1))

    if y_tolerance is None:
        heights = [t.height for t in tokens if t.height > 0]
        y_tolerance = max(8, int(median(heights) * 0.55)) if heights else 12

    lines: list[list[OCRToken]] = []

    for token in tokens:
        placed = False

        for line in lines:
            line_cy = median([t.cy for t in line])
            if abs(token.cy - line_cy) <= y_tolerance:
                line.append(token)
                placed = True
                break

        if not placed:
            lines.append([token])

    for line in lines:
        line.sort(key=lambda t: t.x1)

    lines.sort(key=lambda line: median([t.cy for t in line]))

    return lines


def lines_to_plain_text(lines: list[list[OCRToken]]) -> str:
    """
    Converts grouped lines to simple readable text.
    """
    return "\n".join(" ".join(token.text for token in line) for line in lines)


def lines_to_spaced_text(
    lines: list[list[OCRToken]],
    image_width: int | None = None,
    chars_per_line: int = 100,
) -> str:
    """
    Converts grouped lines to monospaced-ish formatted text.

    This preserves indentation and rough horizontal positioning.
    """
    if not lines:
        return ""

    if image_width is None:
        image_width = max(token.x2 for line in lines for token in line)

    output_lines: list[str] = []

    for line in lines:
        current = ""
        cursor = 0

        for token in line:
            target_col = int((token.x1 / image_width) * chars_per_line)
            spaces = max(1, target_col - cursor)

            current += " " * spaces + token.text
            cursor = len(current)

        output_lines.append(current.rstrip())

    return "\n".join(output_lines)


def paddle_result_to_formatted_text(
    paddle_result: list[dict[str, Any]],
    min_score: float = 0.50,
    preserve_spacing: bool = True,
) -> str:
    """
    Main helper.

    Use preserve_spacing=True for OCR layout reconstruction.
    Use preserve_spacing=False for normal readable paragraphs/lines.
    """
    tokens = extract_tokens(paddle_result, min_score=min_score)
    lines = group_tokens_into_lines(tokens)

    if preserve_spacing:
        return lines_to_spaced_text(lines)

    return lines_to_plain_text(lines)

def lines_to_markdown(lines: list[list[OCRToken]]) -> str:
    output: list[str] = []

    for line in lines:
        text = " ".join(token.text for token in line).strip()

        if not text:
            continue

        # Heuristic: short title-like side labels.
        if len(text) < 40 and "-" in text and text.count(" ") <= 3:
            output.append(f"\n### {text}\n")
        else:
            output.append(text)

    return "\n".join(output)

def test_result(result):
    tokens = extract_tokens(result, min_score=0.60)
    lines = group_tokens_into_lines(tokens)
    return lines_to_markdown(lines)



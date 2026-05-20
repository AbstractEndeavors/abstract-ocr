from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any


@dataclass
class OCRToken:
    text: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


def extract_tokens_from_cleaned(
    cleaned_result: list[dict[str, Any]],
    min_score: float = 0.40,
) -> list[OCRToken]:
    tokens: list[OCRToken] = []

    for page in cleaned_result:
        for item in page.get("items", []):
            box = item.get("box") or []

            if len(box) != 4:
                continue

            text = str(item.get("text", "")).strip()
            score = float(item.get("score", 0.0))

            if not text or score < min_score:
                continue

            x1, y1, x2, y2 = map(int, box)

            tokens.append(
                OCRToken(
                    text=text,
                    score=score,
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
    if not tokens:
        return []

    tokens = sorted(tokens, key=lambda t: (t.cy, t.x1))

    if y_tolerance is None:
        heights = [t.height for t in tokens]
        y_tolerance = max(10, int(median(heights) * 0.60))

    lines: list[list[OCRToken]] = []

    for token in tokens:
        for line in lines:
            line_y = median([t.cy for t in line])

            if abs(token.cy - line_y) <= y_tolerance:
                line.append(token)
                break
        else:
            lines.append([token])

    for line in lines:
        line.sort(key=lambda t: t.x1)

    lines.sort(key=lambda line: median([t.cy for t in line]))
    return lines


def cleaned_result_to_text(
    cleaned_result: list[dict[str, Any]],
    min_score: float = 0.40,
) -> str:
    tokens = extract_tokens_from_cleaned(cleaned_result, min_score=min_score)
    lines = group_tokens_into_lines(tokens)

    return "\n".join(
        " ".join(token.text for token in line)
        for line in lines
    )

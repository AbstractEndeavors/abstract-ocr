from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image


def _shift_box_y(box: Any, y_offset: int) -> list[int]:
    values = [int(v) for v in list(box)]

    if len(values) != 4:
        return values

    x1, y1, x2, y2 = values
    return [x1, y1 + y_offset, x2, y2 + y_offset]


def _page_to_clean_items(page: dict[str, Any], y_offset: int = 0) -> list[dict[str, Any]]:
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    boxes = page.get("rec_boxes",[])

    items: list[dict[str, Any]] = []

    for text, score, box in zip(texts, scores, boxes):
        text = str(text).strip()

        if not text:
            continue

        items.append(
            {
                "text": text,
                "score": float(score),
                "box": _shift_box_y(box, y_offset),
            }
        )

    return items


def _dedupe_items(items: list[dict[str, Any]], y_tolerance: int = 12) -> list[dict[str, Any]]:
    """
    Basic duplicate remover for overlapping tile OCR.
    Removes same text found at nearly the same y-position.
    """
    kept: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for item in sorted(items, key=lambda i: (i["box"][1], i["box"][0])):
        box = item.get("box") or []

        if len(box) != 4:
            continue

        text = item["text"]
        y_bucket = round(int(box[1]) / y_tolerance)
        key = (text, y_bucket)

        if key in seen:
            continue

        seen.add(key)
        kept.append(item)

    return kept


def ocr_image_tiled_clean(
    mgr,
    image_path: str,
    tile_count: int = 4,
    overlap_ratio: float = 0.18,
    min_score: float = 0.40,
) -> list[dict[str, Any]]:
    """
    Run PaddleOCR over overlapping vertical tiles.

    Returns cleaned lightweight OCR result:
        [
            {
                "input_path": "...",
                "page_index": None,
                "items": [...]
            }
        ]
    """
    image_path = str(image_path)

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size

        tile_h = max(1, height // tile_count)
        overlap = int(tile_h * overlap_ratio)

        all_items: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            for index in range(tile_count):
                y1 = max(0, index * tile_h - overlap)
                y2 = min(height, (index + 1) * tile_h + overlap)

                if y2 <= y1:
                    continue

                tile = img.crop((0, y1, width, y2))
                tile_path = tmpdir_path / f"tile_{index:03d}.jpg"
                tile.save(tile_path, quality=95)

                raw = None

                try:
                    raw = mgr.ocr_image(str(tile_path))

                    for page in raw:
                        for item in _page_to_clean_items(page, y_offset=y1):
                            if item["score"] >= min_score:
                                all_items.append(item)

                finally:
                    if raw is not None:
                        del raw
                    gc.collect()

    all_items = _dedupe_items(all_items)

    return [
        {
            "input_path": image_path,
            "page_index": None,
            "items": all_items,
        }
    ]

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any

from typing import Any

from abstract_apis import *

from abstract_ocr.layout_ocr.pipeline import run_on_image
from abstract_ocr import *
from abstract_utilities import *
from .utils import test_result

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
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


def extract_tokens_from_cleaned(
    cleaned_result: list[dict[str, Any]],
    min_score: float = 0.50,
) -> list[OCRToken]:
    tokens: list[OCRToken] = []

    for page in cleaned_result:
        for item in page.get("items", []):
            box = item.get("box") or []

            if len(box) != 4:
                continue

            score = float(item.get("score", 0.0))
            text = str(item.get("text", "")).strip()

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


def group_tokens_into_lines(tokens: list[OCRToken]) -> list[list[OCRToken]]:
    if not tokens:
        return []

    tokens = sorted(tokens, key=lambda t: (t.cy, t.x1))
    heights = [t.height for t in tokens]
    y_tolerance = max(8, int(median(heights) * 0.55))

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
    min_score: float = 0.50,
) -> str:
    tokens = extract_tokens_from_cleaned(cleaned_result, min_score=min_score)
    lines = group_tokens_into_lines(tokens)

    return "\n".join(
        " ".join(token.text for token in line)
        for line in lines
    )


def clean_paddle_result(result: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert PaddleOCR/PaddleX result into a lightweight serializable structure.

    Keeps only text, confidence, and rectangular boxes.
    Drops image arrays, polygons, font objects, model metadata, and visual outputs.
    """
    cleaned_pages: list[dict[str, Any]] = []

    for page in result:
        texts = page.get("rec_texts",[]) or []
        scores = page.get("rec_scores",[]) or []
        boxes = page.get("rec_boxes",[])

        items: list[dict[str, Any]] = []

        for text, score, box in zip(texts, scores, boxes):
            try:
                box_values = [int(v) for v in list(box)]
            except Exception:
                box_values = []

            items.append(
                {
                    "text": str(text),
                    "score": float(score),
                    "box": box_values,
                }
            )

        cleaned_pages.append(
            {
                "input_path": page.get("input_path"),
                "page_index": page.get("page_index"),
                "items": items,
            }
        )

    return cleaned_pages



class PaddleManager(metaclass=SingletonMeta):
    def __init__(self, lang: str = "en"):
        if not hasattr(self, 'initialized'):
            self.lang = lang
            self.ocr = None
            self.initialized = False
            self._initialize_ocr()

    @staticmethod
    def _configure_cpu_runtime() -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

    @staticmethod
    def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
        sig = inspect.signature(callable_obj)
        params = sig.parameters

        # If constructor accepts **kwargs, pass everything.
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs

        return {k: v for k, v in kwargs.items() if k in params}

    def _initialize_ocr(self) -> None:
        if self.initialized:
            return

        try:
            self._configure_cpu_runtime()

            paddle = lazy_import("paddle")
            paddle.device.set_device("cpu")

            PaddleOCR = get_lazy_attr("paddleocr", "PaddleOCR")

            raw_kwargs = {
    "lang": self.lang,
    "device": "cpu",

    # Important for your oneDNN crash.
    "enable_mkldnn": False,

    # PaddleOCR 3.x flags.
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,

    # Do NOT include use_angle_cls with use_textline_orientation.
    # "use_angle_cls": False,
}
            kwargs = self._filter_supported_kwargs(PaddleOCR, raw_kwargs)
            self.ocr = PaddleOCR(**kwargs)

            self.initialized = True
            logger.info(f"✅ PaddleOCR initialized with args: {kwargs}")

        except Exception as e:
            logger.exception(f"❌ Failed to initialize PaddleOCR: {e}")
            self.ocr = None
            self.initialized = False

    @classmethod
    @lru_cache(maxsize=8)
    def get_instance(cls, lang: str = "en"):
        return cls(lang=lang)

    def ocr_image(self, image_path: str, **kwargs):
        if not self.initialized or not self.ocr:
            self._initialize_ocr()

        if not self.ocr:
            return []

        try:
            # New PaddleOCR wants predict().
            if hasattr(self.ocr, "predict"):
                return list(self.ocr.predict(image_path, **kwargs))

            return self.ocr.ocr(image_path, **kwargs)

        except Exception as e:
            logger.exception(f"⚠️ Paddle OCR failed on {image_path}: {e}")
            return []

def debug_ocr_coverage(raw):
    for page in raw:
        boxes = page.get("rec_boxes") or []

        if len(boxes) == 0:
            print("No boxes detected")
            continue

        y_values = []

        for box in boxes:
            values = [int(v) for v in list(box)]
            if len(values) == 4:
                y_values.extend([values[1], values[3]])

        print("detected y-range:", min(y_values), "to", max(y_values))
        print("detected boxes:", len(boxes))
def paddle_image(image_path):
    import gc

    from .tiled_ocr import ocr_image_tiled_clean
    from .clean_formatter import cleaned_result_to_text

    mgr = PaddleManager()

    cleaned = None

    try:
        cleaned = ocr_image_tiled_clean(
            mgr,
            image_path,
            tile_count=4,
            overlap_ratio=0.20,
            min_score=0.35,
        )

        return cleaned_result_to_text(cleaned, min_score=0.35)

    finally:
        if cleaned is not None:
            del cleaned
        gc.collect()

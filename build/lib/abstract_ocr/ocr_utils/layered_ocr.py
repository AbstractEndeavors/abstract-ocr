#!/usr/bin/env python3
"""
abstract_ocr.ocr_utils.layered_ocr
----------------------------------
Unified OCR orchestration for multi-engine text extraction:
- Tesseract
- EasyOCR
- PaddleOCR (via PaddleManager, CPU-only)
"""

from .imports import *
from .paddle_funcs import PaddleManager
# -----------------------------------------------------
# Image Preprocessing
# -----------------------------------------------------
def preprocess_image(input_path: Path, output_path: Path) -> None:
    """Convert to grayscale and apply adaptive thresholding."""
    img = cv2.imread(str(input_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    cv2.imwrite(str(output_path), thresh)


# -----------------------------------------------------
# OCR Backends
# -----------------------------------------------------
def tesseract_ocr_img(img: np.ndarray) -> pd.DataFrame:
    """Perform OCR using Tesseract."""
    pil = Image.fromarray(img)
    df = pytesseract.image_to_data(
        pil,
        config=Ocr_Config.TESS_PSM,
        output_type=pytesseract.Output.DATAFRAME
    )
    return df[df.text.notnull()]


def easyocr_ocr(path: Path) -> pd.DataFrame:
    """Perform OCR using EasyOCR (GPU optional)."""
    reader = easyocr.Reader(Ocr_Config.EASY_LANGS, gpu=False)
    recs = []
    for bbox, text, conf in reader.readtext(str(path)):
        xs, ys = zip(*bbox)
        recs.append({
            "text": text,
            "conf": conf * 100,
            "left": min(xs), "top": min(ys),
            "width": max(xs) - min(xs),
            "height": max(ys) - min(ys),
        })
    return pd.DataFrame(recs)


def paddleocr_ocr(path: Path) -> pd.DataFrame:
    """Perform OCR using PaddleOCR (CPU-only), with Tesseract fallback."""
    from .layered_ocr import tesseract_ocr_img  # local import to avoid circular


    recs = []
    ocr = getattr(PaddleManager.get_instance(), "ocr", None)

    # Attempt PaddleOCR first
    if ocr:
        try:
            results = ocr.ocr(str(path))
            for page in results or []:
                if not page:
                    continue
                for bbox, (text, conf) in page:
                    xs, ys = zip(*bbox)
                    recs.append({
                        "text": text,
                        "conf": conf,
                        "left": min(xs), "top": min(ys),
                        "width": max(xs) - min(xs),
                        "height": max(ys) - min(ys),
                    })
            if recs:
                return pd.DataFrame(recs)
            else:
                logger.warning(f"⚠️ PaddleOCR returned no text for {path}")
        except Exception as e:
            logger.error(f"❌ PaddleOCR failed on {path}: {e}")

    # --- fallback to Tesseract ---
    try:
        img = cv2.imread(str(path))
        df = tesseract_ocr_img(img)
        logger.info(f"✅ Fallback to Tesseract succeeded for {path}")
        return df
    except Exception as e:
        logger.error(f"❌ Both PaddleOCR and Tesseract failed for {path}: {e}")
        return pd.DataFrame(columns=["text", "conf", "left", "top", "width", "height"])

# -----------------------------------------------------
# Layered OCR Logic
# -----------------------------------------------------
def layered_ocr_img(img: np.ndarray, engine: str = "tesseract") -> pd.DataFrame:
    """Perform OCR with multiple backends (layered OCR)."""
    tmp = Path("/tmp/ocr_tmp.png")
    cv2.imwrite(str(tmp), img)

    if engine == "tesseract":
        df = tesseract_ocr_img(img)
    elif engine == "easy":
        df = easyocr_ocr(tmp)
    elif engine == "paddle":
        df = paddleocr_ocr(tmp)
    else:
        logger.warning(f"⚠️ Unknown OCR engine '{engine}', skipping.")
        return pd.DataFrame(columns=["text", "left", "top", "width", "height", "conf"])

    if df is None or df.empty:
        return pd.DataFrame(columns=["text", "left", "top", "width", "height", "conf"])

    # ✅ Force strict top-to-bottom order (no L→R mixing)
    try:
        df = df.sort_values("top").reset_index(drop=True)
    except Exception as e:
        logger.warning(f"Could not sort OCR output for engine {engine}: {e}")

    return df

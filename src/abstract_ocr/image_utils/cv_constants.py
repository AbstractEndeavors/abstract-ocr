"""
abstract_cv.cv_constants
------------------------
Fuzzy resolver for OpenCV constant names (e.g., 'gray', 'binary_inv', 'otsu').
Provides dynamic matching and introspection for cv2 constants.
"""

from ..init_imports import *

CV_SCALE: Set[str] = {
    "THRESH_OTSU",
    "THRESH_BINARY_INV",
    "THRESH_BINARY",
    "THRESH_TRUNC",
    "IMREAD_GRAYSCALE",
    "COLOR_GRAY2BGR",
    "COLOR_BGR2GRAY",
    "COLOR_RGB2BGR",
    "COLOR_BGR2RGB",
}
# Configuration
class Ocr_Config:
    GAUSSIAN_BLUR_KERNEL = (5, 5)
    MORPH_KERNEL = (3, 3)
    THRESH_METHOD = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    OCR_ENGINES = ['tesseract', 'easy', 'paddle']
    TESS_PSM = "--psm 6"
    EASY_LANGS = ['en']
    PADDLE_USE_ANGLE_CLS = True
    PADDLE_LANG = 'en'
    DEEPCODER_MODEL_PATH = "./DeepCoder-14B"  # Path to DeepCoder-14B-Preview
    DEEPCODER_MAX_TOKENS = 512
    
def get_cv_scale(query: str) -> Optional[Tuple[int, int]]:
    """
    Fuzzy resolver for OpenCV constant names (e.g., 'gray', 'binary_inv').
    Returns (cv_constant_value, confidence_score) or None if not found.
    """
    if not query:
        return None

    query_upper = re.sub(r'[^A-Z0-9_]', '', query.upper())
    tokens = query_upper.split('_')

    # 1️⃣ Exact match
    if query_upper in CV_SCALE:
        return getattr(cv2, query_upper), 100

    # 2️⃣ Partial / token-based scoring
    scores = {}
    for candidate in CV_SCALE:
        c_tokens = candidate.split('_')
        overlap = len(set(tokens) & set(c_tokens))
        seq = difflib.SequenceMatcher(None, query_upper, candidate).ratio()
        scores[candidate] = overlap * 20 + int(seq * 80)

    best_name, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score < 40:
        print(f"⚠️ No strong match for '{query}' (best={best_name}, score={best_score})")
        return None

    print(f"✅ Matched '{query}' → {best_name} ({best_score}%)")
    return getattr(cv2, best_name), best_score

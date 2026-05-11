"""
abstract_cv.image_readers
-------------------------
Utility for reading and verifying images with fuzzy OpenCV constant resolution.
"""

from ..init_imports import *
from .cv_constants import get_cv_scale

def read_image(image_path: str, imread_cv_scale: Optional[str] = None):
    """
    Reads an image safely using OpenCV with a flexible imread mode string.
    Falls back to grayscale if the mode is invalid or not specified.
    """
    if not image_path:
        logger.warning("⚠️ No image path provided.")
        return None

    imread_cv_scale = imread_cv_scale or "IMREAD_GRAYSCALE"
    cv_scale_tuple = get_cv_scale(imread_cv_scale)
    cv_scale = cv_scale_tuple[0] if cv_scale_tuple else cv2.IMREAD_GRAYSCALE

    img = cv2.imread(str(image_path), cv_scale)
    if img is None:
        logger.warning(f"⚠️ Could not read image at: {image_path}")
        return None
    return img

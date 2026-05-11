"""
abstract_cv.threshold_utils
---------------------------
Provides thresholding utilities using dynamic cv2 constants.
Includes ink-pixel thresholding and binary shape extraction.
"""

from ..init_imports import *
from .cv_constants import get_cv_scale
from .image_readers import read_image

def get_threshold_for_ink_pixels(
    image_path: str,
    *thresh_cv_scales: str,
    px_x: Optional[int] = None,
    px_y: Optional[int] = None,
    imread_cv_scale: Optional[str] = None,
) -> Tuple[float, "np.ndarray"]:
    """
    Apply OpenCV thresholding based on fuzzy-matched scales (e.g. 'binary_inv', 'otsu').
    Returns (threshold_value, binary_image)
    """
    img = read_image(image_path, imread_cv_scale)
    if img is None:
        return 0, None

    thresh_cv_scales = thresh_cv_scales or ["THRESH_BINARY_INV", "THRESH_OTSU"]
    thresh_scales = 0
    for scale in thresh_cv_scales:
        cv_scale = get_cv_scale(scale)
        if cv_scale:
            thresh_scales |= cv_scale[0]

    px_x = 0 if not is_number(px_x) else px_x
    px_y = 255 if not is_number(px_y) else px_y

    _, bin_img = cv2.threshold(img, px_x, px_y, thresh_scales)
    return _, bin_img


def get_binary_shape(
    image_path: str,
    *thresh_cv_scales: str,
    px_x: Optional[int] = None,
    px_y: Optional[int] = None,
    imread_cv_scale: Optional[str] = None,
) -> Tuple[int, int]:
    """
    Get (height, width) from a thresholded binary version of an image.
    """
    _, bin_img = get_threshold_for_ink_pixels(
        image_path,
        *thresh_cv_scales,
        px_x=px_x,
        px_y=px_y,
        imread_cv_scale=imread_cv_scale,
    )
    if bin_img is None:
        return 0, 0
    return bin_img.shape

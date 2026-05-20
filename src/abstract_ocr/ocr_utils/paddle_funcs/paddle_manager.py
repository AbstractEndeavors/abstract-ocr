#!/usr/bin/env python3
"""
abstract_ocr.ocr_utils.paddle_manager
-------------------------------------
Provides a CPU-only, cached PaddleOCR interface usable across all modules.

Usage:
    from abstract_ocr.ocr_utils.paddle_manager import PaddleManager
    ocr = PaddleManager.get_instance()
    result = ocr.ocr("/path/to/image.png", cls=False)
"""

import logging
from ..imports import *
class PaddleManager(metaclass=SingletonMeta):
    """
    CPU-locked PaddleOCR manager.
    Ensures only one PaddleOCR model instance is ever loaded.
    """

    
    
    def __init__(self, lang: str = "en",cls=False):
        self.lang = lang
        self.ocr = None
        self.initialized = False
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize PaddleOCR in CPU mode."""
        if self.initialized:
            return
        try:
            # 🔒 Lock Paddle to CPU globally
            paddle = lazy_import("paddle")
            paddle.device.set_device("cpu")
            PaddleOCR = get_lazy_attr("paddleocr", "PaddleOCR")
            self.ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
            self.initialized = True
            logger.info("✅ PaddleOCR initialized (CPU mode)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize PaddleOCR: {e}")
            self.ocr = None

    # ------------------------------------------------------
    # Public API
    # ------------------------------------------------------
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls, lang: str = "en"):
        """Return a cached PaddleManager instance."""
        return cls(lang=lang)

    def ocr_image(self, image_path: str, **kwargs):
        """Run OCR on an image (safe wrapper)."""
        if not self.initialized or not self.ocr:
            self._initialize_ocr()
        try:
            result = self.ocr.ocr(image_path, **kwargs)
            return result
        except Exception as e:
            logger.error(f"⚠️ Paddle OCR failed on {image_path}: {e}")
            return []

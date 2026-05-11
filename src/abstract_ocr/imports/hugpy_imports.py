from .init_imports import *


logger = get_logFile(__name__)

# ─────────────────────────────────────────────────────────────
# Lazy model accessors
# ─────────────────────────────────────────────────────────────

_MODELS = {}
def get_abstract_hugpy():
    from abstract_hugpy import (
        get_transformers,
        get_keybert,
        get_pytesseract,
        get_pypdf2,
        get_easyocr,
        get_speech_recognition,
        get_pydub,
        get_paddleocr,
        get_pdf2image,
        get_spacy,
        require,
    )


def get_summarizer():
    """Text summarizer (lazy-loaded)."""
    if "summarizer" not in _MODELS:
        get_abstract_hugpy()
        pipeline = get_transformers("pipeline")
        _MODELS["summarizer"] = pipeline(
            "text-generation",
            model="Falconsai/text_summarization"
        )
    return _MODELS["summarizer"]


def get_keyword_extractor():
    """Keyword extractor (lazy-loaded)."""
    if "keyword_extractor" not in _MODELS:
        get_abstract_hugpy()
        pipeline = get_transformers("pipeline")
        _MODELS["keyword_extractor"] = pipeline(
            "feature-extraction",
            model="distilbert-base-uncased"
        )
    return _MODELS["keyword_extractor"]


def get_generator():
    """Text generator (lazy-loaded)."""
    if "generator" not in _MODELS:
        get_abstract_hugpy()
        pipeline = get_transformers("pipeline")
        _MODELS["generator"] = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1
        )
    return _MODELS["generator"]


def get_kw_model():
    """KeyBERT instance (lazy-loaded)."""
    if "kw_model" not in _MODELS:
        get_abstract_hugpy()
        KeyBERT = get_keybert()
        extractor = get_keyword_extractor()
        _MODELS["kw_model"] = KeyBERT(model=extractor.model)
    return _MODELS["kw_model"]


def get_led():
    """LED tokenizer and model for long documents (lazy-loaded)."""
    if "led" not in _MODELS:
        get_abstract_hugpy()
        LEDTokenizer = get_transformers("LEDTokenizer")
        LEDForConditionalGeneration = get_transformers("LEDForConditionalGeneration")
        tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
        model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
        _MODELS["led"] = (tokenizer, model)
    return _MODELS["led"]

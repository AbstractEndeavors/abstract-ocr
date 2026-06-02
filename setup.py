from time import time
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="abstract_ocr",
    version='0.0.1.69',
    author="putkoff",
    author_email="partners@abstractendeavors.com",
    description=(
        "A structured OCR pipeline designed for layout-aware text extraction "
        "from complex documents, combining preprocessing, column detection, "
        "region classification, PaddleOCR, and ordered OCR assembly."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    install_requires=[
        # abstract packages
        "abstract_utilities",
        "abstract_pdfs>=0.0.29",

        # primary OCR backend
        "paddleocr>=3.5.0",
        "paddlepaddle>=3.2.2",
        "paddlex>=3.5.2",


        # PDF and document processing
        "pdf2image>=1.17.0",
        "PyPDF2>=3.0.1",
        "PyMuPDF>=1.27.2",
        "pypdfium2>=5.6.0",
        "pdfplumber>=0.11.9",
        "pdfminer.six>=20251230",
        "python-docx>=1.2.0",
        "openpyxl>=3.1.5",

        # image / CV processing
        "opencv-python-headless>=4.13.0.92",
        "pillow>=12.1.1",
        "numpy",
        "scikit-image>=0.26.0",

        # NLP / text processing
        "spacy>=3.8.11",
        "nltk>=3.9.4",
        "beautifulsoup4>=4.14.3",
        "lxml>=6.0.2",

        # general runtime support
        "requests>=2.32.5",
        "tqdm>=4.67.3",
        "PyYAML>=6.0.2",
        "packaging",
        "typing_extensions>=4.15.0",

        # video/media support, if abstract_ocr still uses it
        "moviepy>=1.0.3",
    ],
    python_requires=">=3.9",
    extras_require={
        "easyocr": ["easyocr>=1.7.2"],
        "tesseract": ["pytesseract>=0.3.13"],
        "abstract_hugpy":["abstract_hugpy"],
        "all": [
            "easyocr>=1.7.2",
            "pytesseract>=0.3.13",
        ],
    },
    setup_requires=["wheel"],
)

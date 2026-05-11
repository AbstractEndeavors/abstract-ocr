from time import time
import setuptools
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setuptools.setup(
    name='abstract_ocr',
    version='0.0.1.65',
    author='putkoff',
    author_email='partners@abstractendeavors.com',
    description='A structured OCR pipeline designed for **layout-aware text extraction from complex documents**, combining preprocessing, column detection, region classification, and ordered OCR assembly.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'abstract_utilities',
        'requests',
        'pdf2image',
        'PyPDF2',
        'easyocr',
        'pytest',
        'pytesseract',
        'lxml',
        'moviepy==1.0.3',
        'spacy',
        'abstract_hugpy'
        ],
    python_requires=">=3.6",
    # Add this line to include wheel format in your distribution
    setup_requires=['wheel'],
)

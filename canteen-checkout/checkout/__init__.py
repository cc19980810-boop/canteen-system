"""Canteen visual checkout: class-agnostic YOLO detector + vector-retrieval recognizer.

Heavy dependencies (torch / ultralytics / timm) are imported lazily inside the
modules that need them, so the data tools, the vector index and the evaluation
code work in a plain numpy + OpenCV environment.
"""

__version__ = "0.1.0"

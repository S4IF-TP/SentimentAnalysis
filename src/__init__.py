"""
Sentiment Analysis Package
==========================

A production-grade sentiment analysis toolkit supporting:
- Classical ML models (Logistic Regression, SVM, Random Forest, etc.)
- Deep Learning models (MLP, BERT, LoRA fine-tuning)
- Text preprocessing pipelines
- Comprehensive evaluation and visualization

Author: Seif
"""

from .config import Config
from .data.preprocessing import TextPreprocessor
from .data.dataset import SentimentDataset, TextDatasetWithEmbeddings, TokenizedDataset
from .models.classical import ClassicalModelPipeline
from .models.deep_learning import MLP, SentimentClassifier
from .training.trainer import Trainer
from .evaluation.metrics import ModelEvaluator
from .visualization.plots import Visualizer

__version__ = "1.0.0"
__all__ = [
    "Config",
    "TextPreprocessor",
    "SentimentDataset",
    "TextDatasetWithEmbeddings",
    "TokenizedDataset",
    "ClassicalModelPipeline",
    "MLP",
    "SentimentClassifier",
    "Trainer",
    "ModelEvaluator",
    "Visualizer",
]

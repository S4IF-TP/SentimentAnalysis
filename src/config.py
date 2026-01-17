"""
Configuration Module
====================

Centralized configuration for the sentiment analysis project.
All hyperparameters, paths, and settings are defined here for easy management.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    train_path: str = ""
    test_path: str = ""
    
    # Encoding
    encoding: str = "ISO-8859-1"
    
    # Missing value threshold (percentage)
    missing_threshold: float = 5.0
    
    # Text columns
    text_column: str = "text"
    label_column: str = "sentiment"
    
    # Train/validation split
    validation_size: float = 0.2
    random_state: int = 1770


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""
    
    # Stopwords to keep (important for sentiment)
    stopwords_to_keep: set = field(default_factory=lambda: {
        'not', 'no', 'nor', "don't", "isn't", "aren't", "couldn't", "didn't",
        "doesn't", "hadn't", "hasn't", "haven't", "mightn't", "mustn't",
        "needn't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
        'but', 'however', 'although', 'though'
    })
    
    # Regex patterns for cleaning
    url_pattern: str = r'https?://[A-Za-z0-9./_?=#]+'
    punctuation_pattern: str = r"[^a-zA-Z0-9\s!?*$]"
    whitespace_pattern: str = r'\s+'


@dataclass
class ModelConfig:
    """Configuration for models."""
    
    # BERT configuration
    pretrained_model: str = "bert-base-uncased"
    max_length: int = 128
    
    # TF-IDF configuration
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.95
    
    # Number of sentiment classes
    num_classes: int = 3
    
    # Sentiment mappings
    sentiment_mapping: Dict[str, int] = field(default_factory=lambda: {
        "positive": 2, "neutral": 1, "negative": 0
    })
    
    @property
    def reverse_sentiment_mapping(self) -> Dict[int, str]:
        return {v: k for k, v in self.sentiment_mapping.items()}


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Batch sizes
    batch_size_train: int = 64
    batch_size_eval: int = 256
    
    # Learning rate
    learning_rate: float = 2e-5
    
    # Epochs
    num_epochs: int = 20
    
    # Early stopping
    patience: int = 5
    bad_patience: int = 4
    
    # Scheduler
    scheduler_factor: float = 0.1
    scheduler_patience: int = 3
    
    # Checkpoints
    checkpoint_dir: str = "checkpoints"
    
    # LoRA configuration
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1


@dataclass
class Config:
    """Main configuration class combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Device configuration
    device: torch.device = field(default_factory=lambda: 
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Random seed for reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create config from dictionary."""
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            preprocessing=PreprocessingConfig(**config_dict.get("preprocessing", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "data": self.data.__dict__,
            "preprocessing": self.preprocessing.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "device": str(self.device),
            "seed": self.seed,
        }


# Default configuration instance
default_config = Config()

"""
Deep Learning Models
====================

PyTorch-based neural network architectures for sentiment classification.
Includes MLP, BERT classifiers, and LoRA-based fine-tuning.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification on BERT embeddings.
    
    A production-ready MLP with:
    - Batch normalization for stable training
    - Dropout for regularization
    - Kaiming initialization for ReLU activations
    
    Parameters
    ----------
    input_dim : int
        Dimension of input features (e.g., 768 for BERT-base).
    output_dim : int
        Number of output classes.
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default: [input_dim, 512, 256, 128]
    dropout_rates : list of float, optional
        Dropout rates for each hidden layer.
        
    Example
    -------
    >>> model = MLP(input_dim=768, output_dim=3)
    >>> embeddings = torch.randn(32, 768)  # Batch of BERT embeddings
    >>> logits = model(embeddings)  # Shape: (32, 3)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rates: Optional[List[float]] = None
    ):
        """Initialize the MLP architecture."""
        super(MLP, self).__init__()
        
        # Default architecture
        if hidden_dims is None:
            hidden_dims = [input_dim, 512, 256, 128]
        if dropout_rates is None:
            dropout_rates = [0.2, 0.2, 0.1, 0.05]
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Apply Kaiming initialization for ReLU activations."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch_size, input_dim).
            
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, output_dim).
        """
        return self.layers(x)


class SentimentClassifier(nn.Module):
    """
    Flexible sentiment classifier wrapper.
    
    Can wrap different backbone models (MLP, BERT, etc.)
    with a unified interface.
    
    Parameters
    ----------
    backbone : nn.Module
        The backbone network (e.g., MLP or BERT).
    num_classes : int
        Number of output classes.
    freeze_backbone : bool, default=False
        Whether to freeze backbone weights.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        freeze_backbone: bool = False
    ):
        """Initialize the classifier."""
        super(SentimentClassifier, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone."""
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


class BertClassifier(nn.Module):
    """
    BERT-based sentiment classifier.
    
    Wraps HuggingFace's BertForSequenceClassification with
    additional utilities for training and inference.
    
    Parameters
    ----------
    pretrained_model : str, default='bert-base-uncased'
        Name of the pretrained BERT model.
    num_classes : int, default=3
        Number of output classes.
    freeze_bert : bool, default=False
        Whether to freeze BERT weights (feature extraction mode).
        
    Example
    -------
    >>> model = BertClassifier(num_classes=3)
    >>> inputs = tokenizer("I love this!", return_tensors="pt")
    >>> logits = model(**inputs)
    """
    
    def __init__(
        self,
        pretrained_model: str = "bert-base-uncased",
        num_classes: int = 3,
        freeze_bert: bool = False
    ):
        """Initialize the BERT classifier."""
        super(BertClassifier, self).__init__()
        
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes
        )
        
        if freeze_bert:
            self._freeze_bert_layers()
    
    def _freeze_bert_layers(self, unfreeze_last_n: int = 0):
        """
        Freeze BERT layers.
        
        Parameters
        ----------
        unfreeze_last_n : int, default=0
            Number of last transformer layers to keep unfrozen.
        """
        # Freeze embeddings
        for param in self.bert.bert.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        n_layers = len(self.bert.bert.encoder.layer)
        for i, layer in enumerate(self.bert.bert.encoder.layer):
            if i < n_layers - unfreeze_last_n:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs from tokenizer.
        attention_mask : torch.Tensor
            Attention mask.
        token_type_ids : torch.Tensor, optional
            Token type IDs for sentence pairs.
            
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.logits
    
    def predict(self, **inputs) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(**inputs)
        return torch.argmax(logits, dim=1)
    
    @classmethod
    def with_lora(
        cls,
        pretrained_model: str = "bert-base-uncased",
        num_classes: int = 3,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1
    ):
        """
        Create a BERT classifier with LoRA adapters.
        
        Parameters
        ----------
        pretrained_model : str
            Pretrained model name.
        num_classes : int
            Number of output classes.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha scaling factor.
        lora_dropout : float
            LoRA dropout rate.
            
        Returns
        -------
        peft model
            BERT model with LoRA adapters.
        """
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError(
                "PEFT library required for LoRA. "
                "Install with: pip install peft"
            )
        
        # Create base model
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="all"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        
        return model


class EmbeddingClassifier(nn.Module):
    """
    Simple classifier for pre-computed embeddings.
    
    A lightweight classifier when embeddings are pre-computed.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of input embeddings.
    num_classes : int
        Number of output classes.
    hidden_dim : int, default=256
        Hidden layer dimension.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dim: int = 256
    ):
        """Initialize the classifier."""
        super(EmbeddingClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(x)

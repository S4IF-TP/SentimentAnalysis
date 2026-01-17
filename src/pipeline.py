"""
Pipeline Module
===============

End-to-end pipelines for training and inference.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from .data.loader import SentimentDataLoader
from .data.dataset import TextDatasetWithEmbeddings, TokenizedDataset
from .data.preprocessing import SentimentLabelEncoder
from .models.classical import ClassicalModelPipeline, TFIDFPipeline
from .models.deep_learning import MLP, BertClassifier
from .training.trainer import Trainer, BertTrainer
from .evaluation.metrics import ModelEvaluator
from .visualization.plots import Visualizer
from .config import Config


class SentimentAnalysisPipeline:
    """
    End-to-end pipeline for sentiment analysis.
    
    Provides a unified interface for:
    - Data loading and preprocessing
    - Model training (classical ML, MLP, BERT)
    - Evaluation and comparison
    - Visualization
    
    Parameters
    ----------
    config : Config
        Configuration object.
        
    Example
    -------
    >>> pipeline = SentimentAnalysisPipeline(config)
    >>> pipeline.load_data('train.csv', 'test.csv')
    >>> pipeline.train_classical_models()
    >>> pipeline.train_mlp()
    >>> pipeline.compare_models()
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the pipeline."""
        self.config = config or Config()
        self.device = self.config.device
        
        # Components
        self.data_loader: Optional[SentimentDataLoader] = None
        self.label_encoder = SentimentLabelEncoder()
        self.visualizer = Visualizer(self.label_encoder.classes)
        self.evaluator = ModelEvaluator(self.label_encoder.classes, self.device)
        
        # Data containers
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Model containers
        self.tfidf_pipeline: Optional[ClassicalModelPipeline] = None
        self.embedding_pipeline: Optional[ClassicalModelPipeline] = None
        self.mlp_model: Optional[MLP] = None
        self.bert_model = None
        
        # Results
        self.results: List[Dict[str, Any]] = []
        
        # BERT components (lazy loaded)
        self._bert_tokenizer = None
        self._bert_encoder = None
        self._train_embeddings = None
        self._val_embeddings = None
        self._test_embeddings = None
    
    @property
    def bert_tokenizer(self):
        """Lazy load BERT tokenizer."""
        if self._bert_tokenizer is None:
            print(f"Loading BERT tokenizer: {self.config.model.pretrained_model}")
            self._bert_tokenizer = BertTokenizer.from_pretrained(
                self.config.model.pretrained_model
            )
        return self._bert_tokenizer
    
    @property
    def bert_encoder(self):
        """Lazy load BERT model for embeddings."""
        if self._bert_encoder is None:
            print(f"Loading BERT encoder: {self.config.model.pretrained_model}")
            self._bert_encoder = BertModel.from_pretrained(
                self.config.model.pretrained_model
            ).to(self.device)
        return self._bert_encoder
    
    # ==========================================================================
    # Data Loading
    # ==========================================================================
    
    def load_data(
        self,
        train_path: str,
        test_path: str,
        val_size: float = 0.2,
        random_state: int = 1770
    ) -> Tuple:
        """
        Load and preprocess data.
        
        Parameters
        ----------
        train_path : str
            Path to training CSV.
        test_path : str
            Path to test CSV.
        val_size : float
            Validation split size.
        random_state : int
            Random seed.
            
        Returns
        -------
        tuple
            (train_df, val_df, test_df)
        """
        self.data_loader = SentimentDataLoader(train_path, test_path)
        self.train_df, self.val_df, self.test_df = self.data_loader.load_and_prepare(
            val_size=val_size,
            random_state=random_state
        )
        return self.train_df, self.val_df, self.test_df
    
    def get_data_splits(self) -> Tuple:
        """Get train/val/test texts and labels."""
        train_texts, train_labels = self.data_loader.get_texts_and_labels("train")
        val_texts, val_labels = self.data_loader.get_texts_and_labels("val")
        test_texts, test_labels = self.data_loader.get_texts_and_labels("test")
        return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)
    
    # ==========================================================================
    # EDA
    # ==========================================================================
    
    def run_eda(self):
        """Run exploratory data analysis with visualizations."""
        if self.train_df is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Sentiment distribution
        self.visualizer.plot_sentiment_distribution(self.train_df, 'sentiment')
        
        # Text length distribution
        self.visualizer.plot_text_length_distribution(
            self.train_df, 'processed_text', 'sentiment'
        )
        
        # Top words
        self.visualizer.plot_top_words(self.train_df, 'processed_text', 'sentiment')
    
    # ==========================================================================
    # Model Training
    # ==========================================================================
    
    def train_tfidf_models(self) -> Dict[str, float]:
        """
        Train classical ML models with TF-IDF features.
        
        Returns
        -------
        dict
            Model names mapped to accuracies.
        """
        print("=" * 60)
        print("TRAINING TF-IDF BASED MODELS")
        print("=" * 60)
        
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = self.get_data_splits()
        
        # Initialize and train
        self.tfidf_pipeline = ClassicalModelPipeline(use_tfidf=True)
        self.tfidf_pipeline.fit(train_texts, train_labels)
        
        # Evaluate
        results = self.tfidf_pipeline.evaluate_all(
            test_texts, test_labels,
            target_names=self.label_encoder.classes
        )
        
        # Store results
        for name, metrics in results.items():
            self.results.append({
                'model_name': f"TF-IDF + {name}",
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score']
            })
        
        return {name: metrics['accuracy'] for name, metrics in results.items()}
    
    def train_embedding_models(self) -> Dict[str, float]:
        """
        Train classical ML models with BERT embeddings.
        
        Returns
        -------
        dict
            Model names mapped to accuracies.
        """
        print("=" * 60)
        print("TRAINING EMBEDDING BASED MODELS")
        print("=" * 60)
        
        # Get embeddings
        train_embed = self._get_embedding_dataset("train")
        val_embed = self._get_embedding_dataset("val")
        test_embed = self._get_embedding_dataset("test")
        
        X_train, y_train = train_embed.get_numpy()
        X_test, y_test = test_embed.get_numpy()
        
        # Train models
        self.embedding_pipeline = ClassicalModelPipeline(use_tfidf=False)
        self.embedding_pipeline.fit(X_train, y_train)
        
        # Evaluate
        results = self.embedding_pipeline.evaluate_all(
            X_test, y_test,
            target_names=self.label_encoder.classes
        )
        
        # Store results
        for name, metrics in results.items():
            self.results.append({
                'model_name': f"BERT Embed + {name}",
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score']
            })
        
        return {name: metrics['accuracy'] for name, metrics in results.items()}
    
    def train_mlp(
        self,
        epochs: int = 20,
        learning_rate: float = 2e-5,
        batch_size_train: int = 64,
        batch_size_eval: int = 256
    ) -> Dict[str, Any]:
        """
        Train MLP on BERT embeddings.
        
        Parameters
        ----------
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate.
        batch_size_train : int
            Training batch size.
        batch_size_eval : int
            Evaluation batch size.
            
        Returns
        -------
        dict
            Training results and history.
        """
        print("=" * 60)
        print("TRAINING MLP MODEL")
        print("=" * 60)
        
        # Get embedding datasets
        train_embed = self._get_embedding_dataset("train")
        val_embed = self._get_embedding_dataset("val")
        test_embed = self._get_embedding_dataset("test")
        
        # Create data loaders
        train_loader = DataLoader(train_embed, batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(val_embed, batch_size=batch_size_eval, shuffle=False)
        test_loader = DataLoader(test_embed, batch_size=batch_size_eval, shuffle=False)
        
        # Initialize model
        embedding_dim = train_embed.embeddings.shape[1]
        self.mlp_model = MLP(embedding_dim, self.label_encoder.num_classes)
        
        # Train
        trainer = Trainer(
            self.mlp_model,
            self.device,
            learning_rate=learning_rate,
            checkpoint_dir=self.config.training.checkpoint_dir
        )
        history = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Plot training history
        self.visualizer.plot_training_history(history.to_dict())
        
        # Evaluate
        metrics = self.evaluator.evaluate_pytorch(
            self.mlp_model, test_loader, "MLP"
        )
        
        self.results.append({
            'model_name': 'MLP',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1-score': metrics['f1-score']
        })
        
        self.evaluator.print_report(metrics)
        
        return {
            'metrics': metrics,
            'history': history.to_dict()
        }
    
    def train_bert(
        self,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 32
    ) -> Dict[str, Any]:
        """
        Fine-tune BERT for sentiment classification.
        
        Parameters
        ----------
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate.
        use_lora : bool
            Whether to use LoRA for efficient fine-tuning.
        lora_r : int
            LoRA rank.
        lora_alpha : int
            LoRA alpha.
            
        Returns
        -------
        dict
            Training results and history.
        """
        print("=" * 60)
        print(f"TRAINING BERT {'with LoRA' if use_lora else ''}")
        print("=" * 60)
        
        # Get tokenized datasets
        train_dataset = self._get_tokenized_dataset("train")
        val_dataset = self._get_tokenized_dataset("val")
        test_dataset = self._get_tokenized_dataset("test")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size_train,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size_eval,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size_eval,
            shuffle=False
        )
        
        # Initialize model
        if use_lora:
            self.bert_model = BertClassifier.with_lora(
                self.config.model.pretrained_model,
                self.label_encoder.num_classes,
                lora_r=lora_r,
                lora_alpha=lora_alpha
            ).to(self.device)
        else:
            self.bert_model = BertClassifier(
                self.config.model.pretrained_model,
                self.label_encoder.num_classes
            ).bert.to(self.device)
        
        # Train
        trainer = BertTrainer(
            self.bert_model,
            self.device,
            learning_rate=learning_rate,
            checkpoint_dir=self.config.training.checkpoint_dir
        )
        history = trainer.fit(train_loader, val_loader, epochs=epochs)
        
        # Plot training history
        self.visualizer.plot_training_history(history.to_dict())
        
        # Evaluate
        model_name = "BERT + LoRA" if use_lora else "BERT Fine-tuned"
        metrics = self.evaluator.evaluate_bert(
            self.bert_model, test_loader, model_name
        )
        
        self.results.append({
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1-score': metrics['f1-score']
        })
        
        self.evaluator.print_report(metrics)
        
        return {
            'metrics': metrics,
            'history': history.to_dict()
        }
    
    # ==========================================================================
    # Model Comparison
    # ==========================================================================
    
    def compare_models(self):
        """Compare all trained models."""
        if not self.results:
            print("No results to compare. Train some models first.")
            return
        
        print("=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        
        self.visualizer.plot_model_comparison(self.results)
    
    def get_best_model(self) -> Tuple[str, float]:
        """Get the best performing model."""
        if not self.results:
            raise RuntimeError("No results available. Train some models first.")
        
        best = max(self.results, key=lambda x: x['accuracy'])
        return best['model_name'], best['accuracy']
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _get_embedding_dataset(self, split: str) -> TextDatasetWithEmbeddings:
        """Get or create embedding dataset for a split."""
        texts, labels = self.data_loader.get_texts_and_labels(split)
        
        return TextDatasetWithEmbeddings(
            texts=texts,
            labels=labels,
            tokenizer=self.bert_tokenizer,
            max_length=self.config.model.max_length,
            model=self.bert_encoder,
            device=self.device
        )
    
    def _get_tokenized_dataset(self, split: str) -> TokenizedDataset:
        """Get tokenized dataset for a split."""
        texts, labels = self.data_loader.get_texts_and_labels(split)
        
        return TokenizedDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.bert_tokenizer,
            max_length=self.config.model.max_length
        )


class InferencePipeline:
    """
    Pipeline for production inference.
    
    Loads trained models and provides prediction APIs.
    
    Parameters
    ----------
    model_path : str
        Path to saved model.
    model_type : str
        Type of model ('tfidf', 'mlp', 'bert').
    """
    
    def __init__(
        self,
        model_path: str,
        model_type: str = 'bert',
        device: torch.device = None
    ):
        """Initialize the inference pipeline."""
        self.model_path = model_path
        self.model_type = model_type
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = SentimentLabelEncoder()
        
        self._load_model()
    
    def _load_model(self):
        """Load the model from disk."""
        # Implementation depends on model type
        pass
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text.
        
        Parameters
        ----------
        text : str
            Input text.
            
        Returns
        -------
        dict
            Prediction with label and confidence.
        """
        pass
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts.
        
        Parameters
        ----------
        texts : list of str
            Input texts.
            
        Returns
        -------
        list of dict
            Predictions with labels and confidences.
        """
        pass

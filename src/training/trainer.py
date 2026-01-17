"""
Training Module
===============

Comprehensive training utilities with:
- Early stopping
- Learning rate scheduling
- Model checkpointing
- Training history tracking
"""

import os
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainingHistory:
    """Container for training history metrics."""
    
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_accuracy: List[float] = field(default_factory=list)
    val_accuracy: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary."""
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_accuracy': self.train_accuracy,
            'val_accuracy': self.val_accuracy,
            'learning_rates': self.learning_rates
        }


class EarlyStopping:
    """
    Early stopping callback for training.
    
    Parameters
    ----------
    patience : int, default=5
        Number of epochs to wait for improvement.
    min_delta : float, default=0.0
        Minimum change to qualify as improvement.
    mode : str, default='min'
        'min' for loss, 'max' for accuracy.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min'
    ):
        """Initialize early stopping."""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.
        
        Returns True if improvement, False otherwise.
        """
        if self.best_value is None:
            self.best_value = value
            return True
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class Trainer:
    """
    General-purpose trainer for embedding-based models.
    
    Supports training MLP and other models on pre-computed embeddings.
    
    Parameters
    ----------
    model : nn.Module
        The model to train.
    device : torch.device
        Device for training.
    learning_rate : float, default=2e-5
        Initial learning rate.
    checkpoint_dir : str, default='checkpoints'
        Directory for saving checkpoints.
        
    Example
    -------
    >>> trainer = Trainer(model, device)
    >>> history = trainer.fit(train_loader, val_loader, epochs=20)
    >>> trainer.plot_history()
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-5,
        checkpoint_dir: str = 'checkpoints'
    ):
        """Initialize the trainer."""
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize optimizer and criterion
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=3
        )
        
        # Training history
        self.history = TrainingHistory()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 5,
        bad_patience: int = 4,
        verbose: bool = True
    ) -> TrainingHistory:
        """
        Train the model.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader
            Validation data loader.
        epochs : int, default=20
            Maximum number of epochs.
        patience : int, default=5
            Early stopping patience.
        bad_patience : int, default=4
            Patience before rollback to best model.
        verbose : bool, default=True
            Whether to print progress.
            
        Returns
        -------
        TrainingHistory
            Training history object.
        """
        early_stopping = EarlyStopping(patience=patience)
        
        # Save initial model
        self._save_checkpoint('initial_model.pth')
        
        for epoch in range(epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, epoch, epochs, verbose
            )
            self.history.train_loss.append(train_loss)
            self.history.train_accuracy.append(train_acc)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(
                val_loader, epoch, epochs, verbose
            )
            self.history.val_loss.append(val_loss)
            self.history.val_accuracy.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history.learning_rates.append(current_lr)
            
            # Check for improvement
            if early_stopping(val_loss):
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self._save_checkpoint(f'best_model_epoch_{epoch+1}.pth')
                if verbose:
                    print("✓ Validation loss improved. Model saved.")
            else:
                if verbose:
                    print(f"✗ No improvement for {early_stopping.counter} epochs")
                
                # Rollback if needed
                if early_stopping.counter >= bad_patience:
                    if verbose:
                        print(f"Rolling back to best model (epoch {self.best_epoch})")
                    self._load_checkpoint(f'best_model_epoch_{self.best_epoch}.pth')
            
            # Early stopping check
            if early_stopping.should_stop:
                if verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
            
            if verbose:
                print(f"Current LR: {current_lr:.2e}\n")
        
        # Load best model at the end
        self._load_checkpoint(f'best_model_epoch_{self.best_epoch}.pth')
        
        return self.history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", disable=not verbose)
        
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(loader)
        accuracy = correct / total
        
        if verbose:
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Run one validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", disable=not verbose)
        
        with torch.no_grad():
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(loader)
        accuracy = correct / total
        
        if verbose:
            print(f"Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)
    
    def _load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(path))


class BertTrainer(Trainer):
    """
    Trainer specialized for BERT-based models.
    
    Handles tokenized inputs with attention masks.
    
    Example
    -------
    >>> trainer = BertTrainer(bert_model, device)
    >>> history = trainer.fit(train_loader, val_loader, epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-5,
        checkpoint_dir: str = 'checkpoints',
        use_huggingface_output: bool = True
    ):
        """
        Initialize BERT trainer.
        
        Parameters
        ----------
        use_huggingface_output : bool, default=True
            If True, expects model output to have .logits attribute.
        """
        super().__init__(model, device, learning_rate, checkpoint_dir)
        self.use_huggingface_output = use_huggingface_output
    
    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Run one training epoch for BERT."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]", disable=not verbose)
        
        for batch in pbar:
            inputs, labels = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_huggingface_output:
                outputs = self.model(**inputs).logits
            else:
                outputs = self.model(**inputs)
            
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = running_loss / len(loader)
        accuracy = correct / total
        
        if verbose:
            print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        verbose: bool
    ) -> Tuple[float, float]:
        """Run one validation epoch for BERT."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Val]", disable=not verbose)
        
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)
                
                if self.use_huggingface_output:
                    outputs = self.model(**inputs).logits
                else:
                    outputs = self.model(**inputs)
                
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(loader)
        accuracy = correct / total
        
        if verbose:
            print(f"Val Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        
        return avg_loss, accuracy

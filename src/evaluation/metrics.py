"""
Evaluation Metrics Module
=========================

Comprehensive evaluation utilities for sentiment analysis models.
Supports both sklearn and PyTorch models with unified interface.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:
    """
    Unified evaluator for sentiment analysis models.
    
    Supports evaluation of:
    - Sklearn models (with pre-computed features)
    - PyTorch models (with DataLoader)
    - BERT-based models (with tokenized inputs)
    
    Parameters
    ----------
    class_names : list of str
        Names of the classes (e.g., ['negative', 'neutral', 'positive']).
    device : torch.device, optional
        Device for PyTorch models.
        
    Example
    -------
    >>> evaluator = ModelEvaluator(['negative', 'neutral', 'positive'])
    >>> metrics = evaluator.evaluate_sklearn(model, X_test, y_test)
    >>> metrics = evaluator.evaluate_pytorch(model, test_loader)
    """
    
    def __init__(
        self,
        class_names: List[str] = None,
        device: torch.device = None
    ):
        """Initialize the evaluator."""
        self.class_names = class_names or ['negative', 'neutral', 'positive']
        self.device = device or torch.device('cpu')
    
    def evaluate_sklearn(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate an sklearn model.
        
        Parameters
        ----------
        model : sklearn estimator
            Trained sklearn model.
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            True labels.
        model_name : str
            Name for identification.
            
        Returns
        -------
        dict
            Evaluation metrics.
        """
        predictions = model.predict(X)
        return self._compute_metrics(y, predictions, model_name)
    
    def evaluate_pytorch(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Evaluate a PyTorch model on embedding-based data.
        
        Parameters
        ----------
        model : nn.Module
            Trained PyTorch model.
        data_loader : DataLoader
            DataLoader with (embeddings, labels) batches.
        model_name : str
            Name for identification.
            
        Returns
        -------
        dict
            Evaluation metrics.
        """
        predictions, labels = self._get_predictions_pytorch(model, data_loader)
        return self._compute_metrics(labels, predictions, model_name)
    
    def evaluate_bert(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        model_name: str = "BERT",
        use_huggingface_output: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a BERT-based model.
        
        Parameters
        ----------
        model : nn.Module
            Trained BERT model.
        data_loader : DataLoader
            DataLoader with tokenized inputs.
        model_name : str
            Name for identification.
        use_huggingface_output : bool
            If True, expects model output to have .logits attribute.
            
        Returns
        -------
        dict
            Evaluation metrics.
        """
        predictions, labels = self._get_predictions_bert(
            model, data_loader, use_huggingface_output
        )
        return self._compute_metrics(labels, predictions, model_name)
    
    def _get_predictions_pytorch(
        self,
        model: nn.Module,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a PyTorch model."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(target.numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _get_predictions_bert(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        use_huggingface_output: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from a BERT model."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                if use_huggingface_output:
                    outputs = model(**inputs).logits
                else:
                    outputs = model(**inputs)
                
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.
        model_name : str
            Name for identification.
            
        Returns
        -------
        dict
            Dictionary containing all metrics.
        """
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'true_labels': y_true
        }
    
    def compare_models(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple model results.
        
        Parameters
        ----------
        results : list of dict
            List of evaluation results from different models.
        metrics : list of str, optional
            Metrics to compare. Default: accuracy, precision, recall, f1-score.
            
        Returns
        -------
        dict
            Comparison dictionary {model_name: {metric: value}}.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1-score']
        
        comparison = {}
        for result in results:
            name = result['model_name']
            comparison[name] = {
                metric: result[metric] for metric in metrics
            }
        
        return comparison
    
    def get_best_model(
        self,
        results: List[Dict[str, Any]],
        metric: str = 'accuracy'
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Find the best performing model.
        
        Parameters
        ----------
        results : list of dict
            List of evaluation results.
        metric : str, default='accuracy'
            Metric to use for comparison.
            
        Returns
        -------
        tuple
            (model_name, full_results_dict)
        """
        best_idx = np.argmax([r[metric] for r in results])
        best_result = results[best_idx]
        return best_result['model_name'], best_result
    
    def print_report(
        self,
        result: Dict[str, Any],
        show_confusion_matrix: bool = True
    ):
        """
        Print a formatted evaluation report.
        
        Parameters
        ----------
        result : dict
            Evaluation result dictionary.
        show_confusion_matrix : bool
            Whether to print confusion matrix.
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {result['model_name']}")
        print(f"{'='*60}")
        print(f"Accuracy:  {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall:    {result['recall']:.4f}")
        print(f"F1-Score:  {result['f1-score']:.4f}")
        
        if show_confusion_matrix:
            print(f"\nConfusion Matrix:")
            print(result['confusion_matrix'])
        
        print(f"\nClassification Report:")
        report = result['classification_report']
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-" * 55)
        for class_name in self.class_names:
            if class_name in report:
                r = report[class_name]
                print(f"{class_name:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {int(r['support']):>10}")


def evaluate_on_splits(
    evaluator: ModelEvaluator,
    model: Any,
    test_data: Tuple,
    val_data: Tuple,
    model_type: str = 'sklearn',
    model_name: str = 'Model'
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a model on both test and validation splits.
    
    Parameters
    ----------
    evaluator : ModelEvaluator
        The evaluator instance.
    model : Any
        The trained model.
    test_data : tuple
        (X_test, y_test) or test_loader.
    val_data : tuple
        (X_val, y_val) or val_loader.
    model_type : str
        'sklearn', 'pytorch', or 'bert'.
    model_name : str
        Name for identification.
        
    Returns
    -------
    dict
        Results for both test and validation.
    """
    if model_type == 'sklearn':
        X_test, y_test = test_data
        X_val, y_val = val_data
        test_results = evaluator.evaluate_sklearn(model, X_test, y_test, model_name)
        val_results = evaluator.evaluate_sklearn(model, X_val, y_val, model_name)
    elif model_type == 'pytorch':
        test_results = evaluator.evaluate_pytorch(model, test_data, model_name)
        val_results = evaluator.evaluate_pytorch(model, val_data, model_name)
    elif model_type == 'bert':
        test_results = evaluator.evaluate_bert(model, test_data, model_name)
        val_results = evaluator.evaluate_bert(model, val_data, model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return {
        'test': test_results,
        'validation': val_results
    }

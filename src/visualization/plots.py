"""
Visualization Module
====================

Comprehensive plotting utilities for sentiment analysis.
Includes EDA visualizations, training curves, and model comparisons.
"""

from typing import Dict, List, Optional, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Visualizer:
    """
    Visualization toolkit for sentiment analysis.
    
    Provides methods for:
    - Exploratory Data Analysis (EDA)
    - Training history plots
    - Model comparison charts
    - Confusion matrices
    - Embedding visualizations
    
    Parameters
    ----------
    class_names : list of str
        Names of sentiment classes.
    style : str, default='whitegrid'
        Seaborn style for plots.
    figsize : tuple, default=(12, 6)
        Default figure size.
        
    Example
    -------
    >>> viz = Visualizer(['negative', 'neutral', 'positive'])
    >>> viz.plot_sentiment_distribution(df, 'sentiment')
    >>> viz.plot_training_history(history)
    """
    
    # Color palettes for consistent styling
    SENTIMENT_COLORS = {
        'positive': 'deepskyblue',
        'neutral': 'grey',
        'negative': 'coral'
    }
    
    def __init__(
        self,
        class_names: List[str] = None,
        style: str = 'whitegrid',
        figsize: Tuple[int, int] = (12, 6)
    ):
        """Initialize the visualizer."""
        self.class_names = class_names or ['negative', 'neutral', 'positive']
        self.figsize = figsize
        sns.set_theme(style=style)
    
    # ==========================================================================
    # Exploratory Data Analysis (EDA) Plots
    # ==========================================================================
    
    def plot_sentiment_distribution(
        self,
        df: pd.DataFrame,
        label_column: str = 'sentiment',
        title: str = "Distribution of Sentiment Labels"
    ):
        """
        Plot the distribution of sentiment labels.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        label_column : str
            Name of the sentiment column.
        title : str
            Plot title.
        """
        plt.figure(figsize=self.figsize)
        ax = sns.countplot(x=label_column, data=df, palette=self.SENTIMENT_COLORS)
        
        # Add count labels on bars
        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom'
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()
    
    def plot_text_length_distribution(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_text',
        label_column: str = 'sentiment',
        title: str = "Text Length Distribution by Sentiment"
    ):
        """
        Plot the distribution of text lengths by sentiment.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        text_column : str
            Name of the text column.
        label_column : str
            Name of the sentiment column.
        title : str
            Plot title.
        """
        # Calculate text lengths
        df_copy = df.copy()
        df_copy['text_length'] = df_copy[text_column].apply(lambda x: len(str(x).split()))
        
        plt.figure(figsize=self.figsize)
        sns.kdeplot(
            data=df_copy,
            x='text_length',
            hue=label_column,
            fill=True,
            common_norm=False,
            palette=self.SENTIMENT_COLORS
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.show()
    
    def plot_top_words(
        self,
        df: pd.DataFrame,
        text_column: str = 'processed_text',
        label_column: str = 'sentiment',
        top_n: int = 20,
        title: str = "Most Frequent Words by Sentiment"
    ):
        """
        Plot the most frequent words for each sentiment.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        text_column : str
            Name of the text column.
        label_column : str
            Name of the sentiment column.
        top_n : int
            Number of top words to show.
        title : str
            Plot title.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 8))
        
        sentiments = ['positive', 'negative', 'neutral']
        colors = [self.SENTIMENT_COLORS[s] for s in sentiments]
        
        for ax, sentiment, color in zip(axes, sentiments, colors):
            # Get texts for this sentiment
            texts = df[df[label_column] == sentiment][text_column]
            
            # Count words
            word_counts = Counter(" ".join(texts).split())
            top_words = pd.DataFrame(
                word_counts.most_common(top_n),
                columns=['Word', 'Count']
            )
            
            # Plot
            sns.barplot(y='Word', x='Count', data=top_words, color=color, ax=ax)
            ax.set_title(f'{sentiment.capitalize()} Tweets', fontsize=12, fontweight='bold')
            ax.set_xlabel('Count')
            ax.set_ylabel('')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    def plot_categorical_sentiment(
        self,
        df: pd.DataFrame,
        category_column: str,
        label_column: str = 'sentiment',
        title: str = None
    ):
        """
        Plot sentiment distribution across a categorical variable.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the data.
        category_column : str
            Name of the categorical column.
        label_column : str
            Name of the sentiment column.
        title : str
            Plot title.
        """
        if title is None:
            title = f"Sentiment Distribution by {category_column}"
        
        plt.figure(figsize=self.figsize)
        sns.countplot(
            x=category_column,
            data=df,
            hue=label_column,
            palette=self.SENTIMENT_COLORS
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(category_column)
        plt.ylabel('Count')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.show()
    
    # ==========================================================================
    # Training History Plots
    # ==========================================================================
    
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History"
    ):
        """
        Plot training and validation loss/accuracy curves.
        
        Parameters
        ----------
        history : dict
            Dictionary with keys: train_loss, val_loss, train_accuracy, val_accuracy.
        title : str
            Plot title.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss plot
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'o-', label='Training Loss', color='coral')
        axes[0].plot(epochs, history['val_loss'], 'o-', label='Validation Loss', color='deepskyblue')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, history['train_accuracy'], 'o-', label='Training Accuracy', color='coral')
        axes[1].plot(epochs, history['val_accuracy'], 'o-', label='Validation Accuracy', color='deepskyblue')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # ==========================================================================
    # Model Comparison Plots
    # ==========================================================================
    
    def plot_model_comparison(
        self,
        results: List[Dict[str, Any]],
        metrics: List[str] = None,
        title: str = "Model Performance Comparison"
    ):
        """
        Plot comparison of multiple models across metrics.
        
        Parameters
        ----------
        results : list of dict
            List of evaluation results from different models.
        metrics : list of str
            Metrics to compare.
        title : str
            Plot title.
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1-score']
        
        # Create DataFrame for plotting
        df_results = pd.DataFrame([
            {
                'model_name': r['model_name'],
                **{m: r[m] for m in metrics}
            }
            for r in results
        ])
        
        # Create subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        colors = sns.color_palette("deep", n_metrics)
        
        for ax, metric, color in zip(axes, metrics, colors):
            sns.barplot(
                x='model_name', y=metric,
                data=df_results, color=color, ax=ax
            )
            ax.set_title(metric.replace('-', ' ').title(), fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xlabel('')
            ax.set_ylabel('Score')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels
            for p in ax.patches:
                ax.annotate(
                    f'{p.get_height():.3f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9
                )
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Print best model
        best_idx = df_results['accuracy'].idxmax()
        best_model = df_results.loc[best_idx, 'model_name']
        best_acc = df_results.loc[best_idx, 'accuracy']
        print(f"\n🏆 Best Model: {best_model} with {best_acc*100:.2f}% accuracy")
    
    def plot_accuracy_bar(
        self,
        results: Dict[str, float],
        title: str = "Model Accuracy Comparison"
    ):
        """
        Simple bar plot of model accuracies.
        
        Parameters
        ----------
        results : dict
            Dictionary mapping model names to accuracies.
        title : str
            Plot title.
        """
        plt.figure(figsize=self.figsize)
        
        names = list(results.keys())
        accuracies = list(results.values())
        
        bars = plt.bar(names, accuracies, color='skyblue', edgecolor='navy')
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f'{acc:.4f}',
                ha='center', va='bottom'
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    # ==========================================================================
    # Confusion Matrix Plots
    # ==========================================================================
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        highlight_diagonal: bool = True
    ):
        """
        Plot a single confusion matrix.
        
        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix array.
        title : str
            Plot title.
        highlight_diagonal : bool
            Whether to highlight diagonal cells.
        """
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(
            cm, annot=True, fmt='d', cbar=True,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap='Blues'
        )
        
        if highlight_diagonal:
            for i in range(cm.shape[0]):
                ax.add_patch(patches.Rectangle(
                    (i, i), 1, 1,
                    fill=False, edgecolor='coral', linewidth=2
                ))
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices_comparison(
        self,
        cm_test: np.ndarray,
        cm_val: np.ndarray,
        model_name: str = "Model"
    ):
        """
        Plot test and validation confusion matrices side by side.
        
        Parameters
        ----------
        cm_test : np.ndarray
            Test confusion matrix.
        cm_val : np.ndarray
            Validation confusion matrix.
        model_name : str
            Model name for title.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Test confusion matrix
        sns.heatmap(
            cm_test, annot=True, fmt='d', cbar=True,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[0], cmap='Blues'
        )
        axes[0].set_title('Test Confusion Matrix', fontweight='bold')
        axes[0].set_xlabel('Predicted Labels')
        axes[0].set_ylabel('True Labels')
        
        # Highlight diagonal
        for i in range(cm_test.shape[0]):
            axes[0].add_patch(patches.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor='coral', linewidth=2
            ))
        
        # Validation confusion matrix
        sns.heatmap(
            cm_val, annot=True, fmt='d', cbar=True,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=axes[1], cmap='Blues'
        )
        axes[1].set_title('Validation Confusion Matrix', fontweight='bold')
        axes[1].set_xlabel('Predicted Labels')
        axes[1].set_ylabel('True Labels')
        
        for i in range(cm_val.shape[0]):
            axes[1].add_patch(patches.Rectangle(
                (i, i), 1, 1,
                fill=False, edgecolor='coral', linewidth=2
            ))
        
        fig.suptitle(f'{model_name} - Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
    
    # ==========================================================================
    # Embedding Visualization
    # ==========================================================================
    
    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization of Embeddings",
        random_state: int = 42
    ):
        """
        Plot t-SNE visualization of embeddings.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix of shape (n_samples, embedding_dim).
        labels : np.ndarray
            Array of integer labels.
        title : str
            Plot title.
        random_state : int
            Random seed for t-SNE.
        """
        print("Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        
        for label_idx, class_name in enumerate(self.class_names):
            mask = labels == label_idx
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=class_name,
                alpha=0.6,
                s=30
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.tight_layout()
        plt.show()
    
    def plot_pca_variance(
        self,
        embeddings: np.ndarray,
        n_components: int = 10,
        title: str = "PCA Explained Variance"
    ):
        """
        Plot PCA explained variance and cumulative variance.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix.
        n_components : int
            Number of components to show.
        title : str
            Plot title.
        """
        print("Computing PCA...")
        pca = PCA()
        pca.fit(embeddings)
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        axes[0].bar(
            range(1, n_components + 1),
            explained_variance[:n_components],
            alpha=0.7, color='steelblue'
        )
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Variance per Component', fontweight='bold')
        axes[0].set_xticks(range(1, n_components + 1))
        
        # Cumulative variance
        axes[1].plot(
            range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            'o-', color='coral'
        )
        axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
        
        # Print number of components for 95% variance
        n_95 = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"Components needed for 95% variance: {n_95}")

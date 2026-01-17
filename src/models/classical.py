"""
Classical Machine Learning Models
=================================

Provides sklearn-based pipelines for sentiment classification.
Supports TF-IDF vectorization and various classifiers.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib


class TFIDFPipeline:
    """
    TF-IDF based classification pipeline.
    
    Combines TF-IDF vectorization with a classifier in a single pipeline.
    Supports multiple classifier types with optimized default parameters.
    
    Parameters
    ----------
    classifier_type : str, default='logistic_regression'
        Type of classifier. Options: 'logistic_regression', 'naive_bayes',
        'linear_svc', 'random_forest'.
    max_features : int, default=5000
        Maximum number of TF-IDF features.
    ngram_range : tuple, default=(1, 2)
        N-gram range for TF-IDF.
    min_df : int, default=5
        Minimum document frequency for terms.
    max_df : float, default=0.95
        Maximum document frequency for terms.
        
    Example
    -------
    >>> pipeline = TFIDFPipeline(classifier_type='logistic_regression')
    >>> pipeline.fit(train_texts, train_labels)
    >>> predictions = pipeline.predict(test_texts)
    >>> accuracy = pipeline.evaluate(test_texts, test_labels)
    """
    
    # Default classifier configurations optimized for multi-class sentiment
    CLASSIFIER_CONFIGS = {
        'logistic_regression': {
            'class': LogisticRegression,
            'params': {
                'multi_class': 'multinomial',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        },
        'naive_bayes': {
            'class': MultinomialNB,
            'params': {'alpha': 0.1}
        },
        'linear_svc': {
            'class': LinearSVC,
            'params': {
                'multi_class': 'ovr',
                'max_iter': 1000,
                'class_weight': 'balanced'
            }
        },
        'random_forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': 100,
                'class_weight': 'balanced',
                'max_depth': 10,
                'random_state': 42
            }
        }
    }
    
    def __init__(
        self,
        classifier_type: str = 'logistic_regression',
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 5,
        max_df: float = 0.95,
        custom_classifier: Optional[Any] = None
    ):
        """Initialize the TF-IDF pipeline."""
        self.classifier_type = classifier_type
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        
        # Initialize classifier
        if custom_classifier is not None:
            self.classifier = custom_classifier
        else:
            config = self.CLASSIFIER_CONFIGS.get(classifier_type)
            if config is None:
                raise ValueError(
                    f"Unknown classifier type: {classifier_type}. "
                    f"Available: {list(self.CLASSIFIER_CONFIGS.keys())}"
                )
            self.classifier = config['class'](**config['params'])
        
        # Create sklearn pipeline
        self.pipeline = Pipeline([
            ('tfidf', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        self._is_fitted = False
    
    def fit(self, texts: List[str], labels: List[int]) -> "TFIDFPipeline":
        """
        Fit the pipeline on training data.
        
        Parameters
        ----------
        texts : list of str
            Training texts.
        labels : list of int
            Training labels.
            
        Returns
        -------
        self
        """
        self.pipeline.fit(texts, labels)
        self._is_fitted = True
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict labels for texts.
        
        Parameters
        ----------
        texts : list of str
            Texts to classify.
            
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        return self.pipeline.predict(texts)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities (if supported).
        
        Parameters
        ----------
        texts : list of str
            Texts to classify.
            
        Returns
        -------
        np.ndarray
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        if hasattr(self.classifier, 'predict_proba'):
            return self.pipeline.predict_proba(texts)
        raise NotImplementedError(
            f"{self.classifier_type} does not support predict_proba"
        )
    
    def evaluate(
        self,
        texts: List[str],
        labels: List[int],
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the pipeline on test data.
        
        Parameters
        ----------
        texts : list of str
            Test texts.
        labels : list of int
            True labels.
        target_names : list of str, optional
            Names for each class.
            
        Returns
        -------
        dict
            Evaluation metrics including accuracy and classification report.
        """
        predictions = self.predict(texts)
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(
            labels, predictions,
            target_names=target_names,
            output_dict=True
        )
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': predictions
        }
    
    def save(self, path: str) -> None:
        """Save the pipeline to disk."""
        joblib.dump(self.pipeline, path)
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "TFIDFPipeline":
        """Load a pipeline from disk."""
        instance = cls.__new__(cls)
        instance.pipeline = joblib.load(path)
        instance._is_fitted = True
        return instance


class ClassicalModelPipeline:
    """
    Multi-model comparison pipeline for classical ML.
    
    Trains and evaluates multiple classifiers simultaneously
    for easy comparison.
    
    Parameters
    ----------
    use_tfidf : bool, default=True
        Whether to use TF-IDF features.
    models : dict, optional
        Custom model configurations.
        
    Example
    -------
    >>> pipeline = ClassicalModelPipeline()
    >>> pipeline.fit(X_train, y_train)
    >>> results = pipeline.evaluate_all(X_test, y_test)
    >>> pipeline.compare_results()
    """
    
    def __init__(
        self,
        use_tfidf: bool = True,
        models: Optional[Dict[str, Any]] = None
    ):
        """Initialize the comparison pipeline."""
        self.use_tfidf = use_tfidf
        
        # Default model configurations
        if models is None:
            self.models = {
                'Logistic Regression': LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'Naive Bayes': MultinomialNB(alpha=0.1),
                'Linear SVM': LinearSVC(
                    multi_class='ovr',
                    max_iter=1000,
                    class_weight='balanced'
                ),
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    class_weight='balanced',
                    max_depth=10,
                    random_state=42
                ),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        else:
            self.models = models
        
        # TF-IDF vectorizer
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95
            )
        else:
            self.vectorizer = None
        
        self.results: Dict[str, Dict] = {}
        self._is_fitted = False
    
    def fit(
        self,
        texts_or_features: Any,
        labels: List[int],
        verbose: bool = True
    ) -> "ClassicalModelPipeline":
        """
        Fit all models on training data.
        
        Parameters
        ----------
        texts_or_features : list or array
            Raw texts (if use_tfidf=True) or feature matrix.
        labels : list of int
            Training labels.
        verbose : bool, default=True
            Whether to print progress.
            
        Returns
        -------
        self
        """
        # Vectorize if using TF-IDF
        if self.vectorizer is not None:
            if verbose:
                print("Fitting TF-IDF vectorizer...")
            X = self.vectorizer.fit_transform(texts_or_features)
        else:
            X = texts_or_features
        
        # Fit each model
        for name, model in self.models.items():
            if verbose:
                print(f"Training {name}...")
            model.fit(X, labels)
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        texts_or_features: Any,
        model_name: str
    ) -> np.ndarray:
        """
        Predict using a specific model.
        
        Parameters
        ----------
        texts_or_features : list or array
            Raw texts or feature matrix.
        model_name : str
            Name of the model to use.
            
        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Transform if using TF-IDF
        if self.vectorizer is not None:
            X = self.vectorizer.transform(texts_or_features)
        else:
            X = texts_or_features
        
        return self.models[model_name].predict(X)
    
    def evaluate_all(
        self,
        texts_or_features: Any,
        labels: List[int],
        target_names: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Evaluate all models on test data.
        
        Parameters
        ----------
        texts_or_features : list or array
            Raw texts or feature matrix.
        labels : list of int
            True labels.
        target_names : list of str, optional
            Names for each class.
        verbose : bool, default=True
            Whether to print results.
            
        Returns
        -------
        dict
            Results for each model.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        # Transform if using TF-IDF
        if self.vectorizer is not None:
            X = self.vectorizer.transform(texts_or_features)
        else:
            X = texts_or_features
        
        self.results = {}
        
        for name, model in self.models.items():
            predictions = model.predict(X)
            accuracy = accuracy_score(labels, predictions)
            report = classification_report(
                labels, predictions,
                target_names=target_names,
                output_dict=True
            )
            
            self.results[name] = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1-score': report['weighted avg']['f1-score'],
                'predictions': predictions,
                'classification_report': report
            }
            
            if verbose:
                print(f"\n{name}: Accuracy = {accuracy:.4f}")
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on accuracy.
        
        Returns
        -------
        tuple
            (model_name, model_instance)
        """
        if not self.results:
            raise RuntimeError("No results available. Call evaluate_all() first.")
        
        best_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        return best_name, self.models[best_name]
    
    def save_best_model(self, path: str) -> None:
        """Save the best model to disk."""
        best_name, best_model = self.get_best_model()
        
        # Save model and vectorizer
        save_obj = {
            'model': best_model,
            'vectorizer': self.vectorizer,
            'model_name': best_name
        }
        joblib.dump(save_obj, path)
        print(f"Best model ({best_name}) saved to {path}")

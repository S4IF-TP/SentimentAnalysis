"""
Text Preprocessing Module
=========================

Provides comprehensive text preprocessing pipeline for sentiment analysis.
Handles missing values, text cleaning, stopword removal, and text normalization.
"""

import re
from typing import List, Optional, Set
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    A production-ready text preprocessing pipeline.
    
    This class provides methods for:
    - Handling missing values with configurable thresholds
    - Cleaning text (removing URLs, punctuation, extra whitespace)
    - Removing stopwords while preserving sentiment-important words
    - Lowercasing and normalization
    
    Example
    -------
    >>> preprocessor = TextPreprocessor()
    >>> clean_text = preprocessor.preprocess("I really don't like this! http://example.com")
    >>> print(clean_text)
    'really dont like this!'
    """
    
    # Default stopwords to keep (important for sentiment analysis)
    DEFAULT_STOPWORDS_TO_KEEP = {
        'not', 'no', 'nor', "don't", "isn't", "aren't", "couldn't", "didn't",
        "doesn't", "hadn't", "hasn't", "haven't", "mightn't", "mustn't",
        "needn't", "shouldn't", "wasn't", "weren't", "won't", "wouldn't",
        'but', 'however', 'although', 'though'
    }
    
    # Regex patterns for text cleaning
    URL_PATTERN = re.compile(r'https?://[A-Za-z0-9./_?=#]+')
    PUNCTUATION_PATTERN = re.compile(r"[^a-zA-Z0-9\s!?*$]")
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    def __init__(
        self,
        stopwords_to_keep: Optional[Set[str]] = None,
        remove_urls: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        remove_stopwords: bool = True
    ):
        """
        Initialize the TextPreprocessor.
        
        Parameters
        ----------
        stopwords_to_keep : set, optional
            Set of stopwords to keep (e.g., negations for sentiment).
            If None, uses default set.
        remove_urls : bool, default=True
            Whether to remove URLs from text.
        remove_punctuation : bool, default=True
            Whether to remove punctuation (except !?*$).
        lowercase : bool, default=True
            Whether to convert text to lowercase.
        remove_stopwords : bool, default=True
            Whether to remove English stopwords.
        """
        self.remove_urls = remove_urls
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        
        # Build custom stopwords list
        if stopwords_to_keep is None:
            stopwords_to_keep = self.DEFAULT_STOPWORDS_TO_KEEP
        
        self.stopwords = self._build_stopwords(stopwords_to_keep)
    
    def _build_stopwords(self, stopwords_to_keep: Set[str]) -> Set[str]:
        """
        Build the stopwords set, excluding words we want to keep.
        
        Parameters
        ----------
        stopwords_to_keep : set
            Words to exclude from stopwords removal.
            
        Returns
        -------
        set
            The filtered stopwords set.
        """
        english_stopwords = set(stopwords.words('english'))
        return english_stopwords - stopwords_to_keep
    
    def preprocess(self, text: str) -> str:
        """
        Apply the full preprocessing pipeline to a single text.
        
        Parameters
        ----------
        text : str
            The input text to preprocess.
            
        Returns
        -------
        str
            The cleaned and preprocessed text.
        """
        if not isinstance(text, str):
            return ""
        
        # Step 1: Lowercase
        if self.lowercase:
            text = text.lower()
        
        # Step 2: Remove stopwords
        if self.remove_stopwords:
            text = " ".join([
                word for word in text.split() 
                if word not in self.stopwords
            ])
        
        # Step 3: Remove URLs
        if self.remove_urls:
            text = self.URL_PATTERN.sub(" ", text)
        
        # Step 4: Remove punctuation (keep !?*$)
        if self.remove_punctuation:
            text = self.PUNCTUATION_PATTERN.sub("", text)
        
        # Step 5: Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text).strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Apply preprocessing to a batch of texts.
        
        Parameters
        ----------
        texts : list of str
            List of texts to preprocess.
            
        Returns
        -------
        list of str
            List of preprocessed texts.
        """
        return [self.preprocess(text) for text in texts]
    
    def preprocess_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        output_column: str = "processed_text"
    ) -> pd.DataFrame:
        """
        Apply preprocessing to a DataFrame column.
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        text_column : str
            Name of the column containing text.
        output_column : str, default='processed_text'
            Name for the new preprocessed column.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with the new preprocessed column.
        """
        df = df.copy()
        df[output_column] = df[text_column].apply(self.preprocess)
        return df
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        threshold: float = 5.0,
        columns_to_keep: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values in a DataFrame.
        
        Strategy:
        - If missing < threshold%: drop rows with any missing values
        - If missing >= threshold%: only drop rows where specified columns are missing
        
        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame.
        threshold : float, default=5.0
            Percentage threshold for strategy selection.
        columns_to_keep : list, optional
            Columns to check for missing values when using subset strategy.
        verbose : bool, default=True
            Whether to print progress information.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with missing values handled.
        """
        missing_pct = df.isnull().mean().mean() * 100
        
        if verbose:
            print(f"Missing values: {missing_pct:.4f}%")
        
        if missing_pct < threshold:
            if verbose:
                print(f"Missing < {threshold}%. Dropping rows with any missing values...")
            df_clean = df.dropna()
        else:
            if verbose:
                print(f"Missing >= {threshold}%. Keeping rows where key columns are complete...")
            if columns_to_keep:
                df_clean = df.dropna(subset=columns_to_keep)
            else:
                df_clean = df
        
        if verbose:
            print(f"Rows removed: {len(df) - len(df_clean)}")
        
        return df_clean


class SentimentLabelEncoder:
    """
    Encoder for sentiment labels.
    
    Provides bidirectional mapping between string labels and numeric classes.
    
    Example
    -------
    >>> encoder = SentimentLabelEncoder()
    >>> encoder.encode("positive")
    2
    >>> encoder.decode(0)
    'negative'
    """
    
    DEFAULT_MAPPING = {"positive": 2, "neutral": 1, "negative": 0}
    
    def __init__(self, mapping: Optional[dict] = None):
        """
        Initialize the encoder.
        
        Parameters
        ----------
        mapping : dict, optional
            Custom label-to-integer mapping.
        """
        self.mapping = mapping or self.DEFAULT_MAPPING
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
    
    def encode(self, label: str) -> int:
        """Encode a string label to integer."""
        return self.mapping.get(label, -1)
    
    def decode(self, value: int) -> str:
        """Decode an integer to string label."""
        return self.reverse_mapping.get(value, "unknown")
    
    def encode_series(self, series: pd.Series) -> pd.Series:
        """Encode a pandas Series of labels."""
        return series.map(self.mapping)
    
    def decode_series(self, series: pd.Series) -> pd.Series:
        """Decode a pandas Series of integers."""
        return series.map(self.reverse_mapping)
    
    @property
    def classes(self) -> List[str]:
        """Return list of class names in order."""
        return [self.reverse_mapping[i] for i in sorted(self.reverse_mapping.keys())]
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.mapping)

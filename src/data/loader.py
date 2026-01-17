"""
Data Loader Module
==================

Provides high-level data loading and preparation utilities.
"""

import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import TextPreprocessor, SentimentLabelEncoder


class SentimentDataLoader:
    """
    High-level data loader for sentiment analysis.
    
    Handles the complete data loading pipeline:
    1. Load CSV files
    2. Handle missing values
    3. Preprocess text
    4. Encode labels
    5. Split into train/val/test
    
    Parameters
    ----------
    train_path : str
        Path to training CSV file.
    test_path : str
        Path to test CSV file.
    text_column : str, default='text'
        Name of the text column.
    label_column : str, default='sentiment'
        Name of the label column.
    encoding : str, default='ISO-8859-1'
        File encoding.
        
    Example
    -------
    >>> loader = SentimentDataLoader('train.csv', 'test.csv')
    >>> train_df, val_df, test_df = loader.load_and_prepare()
    """
    
    def __init__(
        self,
        train_path: str,
        test_path: str,
        text_column: str = "text",
        label_column: str = "sentiment",
        encoding: str = "ISO-8859-1"
    ):
        """Initialize the data loader."""
        self.train_path = train_path
        self.test_path = test_path
        self.text_column = text_column
        self.label_column = label_column
        self.encoding = encoding
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.label_encoder = SentimentLabelEncoder()
        
        # Data containers
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw CSV files.
        
        Returns
        -------
        tuple
            (train_df, test_df) DataFrames.
        """
        print(f"Loading training data from: {self.train_path}")
        train_df = pd.read_csv(self.train_path, encoding=self.encoding)
        
        print(f"Loading test data from: {self.test_path}")
        test_df = pd.read_csv(self.test_path, encoding=self.encoding)
        
        print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def load_and_prepare(
        self,
        val_size: float = 0.2,
        random_state: int = 1770,
        missing_threshold: float = 5.0,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete data loading and preparation pipeline.
        
        Parameters
        ----------
        val_size : float, default=0.2
            Fraction of training data for validation.
        random_state : int, default=1770
            Random seed for reproducibility.
        missing_threshold : float, default=5.0
            Threshold for missing value handling strategy.
        verbose : bool, default=True
            Whether to print progress info.
            
        Returns
        -------
        tuple
            (train_df, val_df, test_df) prepared DataFrames.
        """
        # Step 1: Load raw data
        train_df, test_df = self.load_raw_data()
        
        # Step 2: Handle missing values
        if verbose:
            print("\n--- Handling Missing Values ---")
        train_df = TextPreprocessor.handle_missing_values(
            train_df, threshold=missing_threshold, verbose=verbose
        )
        test_df = TextPreprocessor.handle_missing_values(
            test_df,
            threshold=missing_threshold,
            columns_to_keep=[self.text_column, self.label_column],
            verbose=verbose
        )
        
        # Step 3: Preprocess text
        if verbose:
            print("\n--- Preprocessing Text ---")
        train_df = self.preprocessor.preprocess_dataframe(
            train_df, self.text_column, "processed_text"
        )
        test_df = self.preprocessor.preprocess_dataframe(
            test_df, self.text_column, "processed_text"
        )
        
        # Step 4: Encode labels
        if verbose:
            print("\n--- Encoding Labels ---")
        train_df["label"] = self.label_encoder.encode_series(train_df[self.label_column])
        test_df["label"] = self.label_encoder.encode_series(test_df[self.label_column])
        
        # Step 5: Split train into train/val
        if verbose:
            print(f"\n--- Splitting Data (val_size={val_size}) ---")
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_df["label"]
        )
        
        if verbose:
            print(f"Final splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Store for later access
        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)
        
        return self.train_df, self.val_df, self.test_df
    
    def get_texts_and_labels(
        self,
        split: str = "train"
    ) -> Tuple[list, list]:
        """
        Get preprocessed texts and labels for a specific split.
        
        Parameters
        ----------
        split : str
            One of 'train', 'val', 'test'.
            
        Returns
        -------
        tuple
            (texts, labels) lists.
        """
        df_map = {
            "train": self.train_df,
            "val": self.val_df,
            "test": self.test_df
        }
        
        df = df_map.get(split)
        if df is None:
            raise ValueError(f"Split '{split}' not loaded. Call load_and_prepare() first.")
        
        return df["processed_text"].tolist(), df["label"].tolist()

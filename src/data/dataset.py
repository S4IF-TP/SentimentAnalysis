"""
Dataset Module
==============

PyTorch Dataset classes for sentiment analysis.
Supports both tokenized inputs and pre-computed embeddings.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedModel
from tqdm import tqdm


class TokenizedDataset(Dataset):
    """
    Dataset for tokenized text inputs.
    
    Tokenizes text on initialization for efficient batch loading.
    Use this with BERT-based models that need raw tokenized inputs.
    
    Parameters
    ----------
    texts : list of str
        List of text samples.
    labels : list of int
        List of integer labels.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer (e.g., BertTokenizer).
    max_length : int
        Maximum sequence length for tokenization.
        
    Example
    -------
    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> dataset = TokenizedDataset(texts, labels, tokenizer, max_length=128)
    >>> inputs, label = dataset[0]
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128
    ):
        """Initialize and tokenize all texts."""
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a single sample.
        
        Returns
        -------
        tuple
            (encoding_dict, label) where encoding_dict contains
            'input_ids', 'attention_mask', etc.
        """
        return (
            {key: val[idx] for key, val in self.encodings.items()},
            self.labels[idx]
        )


class TextDatasetWithEmbeddings(Dataset):
    """
    Dataset with pre-computed BERT embeddings.
    
    Computes embeddings at initialization for faster training.
    Use this with classical ML models or MLP classifiers.
    
    Parameters
    ----------
    texts : list of str
        List of text samples.
    labels : list of int
        List of integer labels.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer.
    max_length : int
        Maximum sequence length.
    model : PreTrainedModel
        BERT model for generating embeddings.
    device : torch.device
        Device for computation.
    batch_size : int, default=512
        Batch size for embedding generation.
        
    Attributes
    ----------
    embeddings : torch.Tensor
        Pre-computed embeddings of shape (n_samples, embedding_dim).
    labels : torch.Tensor
        Label tensor of shape (n_samples,).
        
    Example
    -------
    >>> dataset = TextDatasetWithEmbeddings(
    ...     texts, labels, tokenizer, max_length=128,
    ...     model=bert_model, device=device
    ... )
    >>> embedding, label = dataset[0]
    >>> print(embedding.shape)  # (768,) for bert-base
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        model: PreTrainedModel,
        device: torch.device,
        batch_size: int = 512
    ):
        """Initialize and compute embeddings."""
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model = model
        self.device = device
        self.batch_size = batch_size
        
        # Tokenize all texts upfront
        self.encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Precompute embeddings
        self.embeddings = self._precompute_embeddings()
    
    def _precompute_embeddings(self) -> torch.Tensor:
        """
        Generate embeddings in batches to avoid GPU memory overflow.
        
        Returns
        -------
        torch.Tensor
            Embeddings of shape (n_samples, embedding_dim).
        """
        print("Generating BERT embeddings in batches...")
        self.model.eval()
        self.model.to(self.device)
        
        num_samples = self.encodings["input_ids"].size(0)
        all_embeddings = []
        
        for start_idx in tqdm(range(0, num_samples, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_samples)
            
            # Get batch inputs
            batch_input_ids = self.encodings["input_ids"][start_idx:end_idx].to(self.device)
            batch_attention_mask = self.encodings["attention_mask"][start_idx:end_idx].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                # Use pooler_output (CLS token representation)
                batch_embeddings = outputs.pooler_output
            
            all_embeddings.append(batch_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0)
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns
        -------
        tuple
            (embedding, label) tensors.
        """
        return self.embeddings[idx], self.labels[idx]
    
    def get_numpy(self) -> Tuple:
        """
        Get embeddings and labels as numpy arrays.
        
        Useful for sklearn models.
        
        Returns
        -------
        tuple
            (embeddings_array, labels_array)
        """
        return self.embeddings.numpy(), self.labels.numpy()


class SentimentDataset(Dataset):
    """
    General-purpose sentiment dataset.
    
    A flexible dataset class that can hold preprocessed data
    in various formats.
    
    Parameters
    ----------
    texts : list of str
        List of text samples.
    labels : list of int
        List of integer labels.
    transform : callable, optional
        Optional transform to apply to each sample.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        transform: Optional[callable] = None
    ):
        """Initialize the dataset."""
        self.texts = texts
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Get a single sample.
        
        Returns
        -------
        tuple
            (text, label) pair.
        """
        text = self.texts[idx]
        label = self.labels[idx]
        
        if self.transform:
            text = self.transform(text)
        
        return text, label

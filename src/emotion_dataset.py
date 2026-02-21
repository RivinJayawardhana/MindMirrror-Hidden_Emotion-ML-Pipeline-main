"""
Dataset and data preprocessing for emotion classification.
Matches notebook implementation exactly with text augmentation.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional
import logging

# IMPORT from shared preprocessing
from src.preprocessing import build_input, EmotionHiddenDataset as BaseDataset

logger = logging.getLogger(__name__)


class EmotionHiddenDataset(BaseDataset):
    """
    PyTorch Dataset for emotion classification with text augmentation.
    Matches notebook implementation exactly - inherits from shared preprocessing.
    """
    
    def __init__(self, texts: List[str], emo_ids: List[int], hid_ids: List[int],
                 emojis: List[str], augment: bool = False, minority_classes: List[int] = None):
        """
        Args:
            texts: List of text strings
            emo_ids: List of emotion class IDs
            hid_ids: List of hidden flag IDs (0 or 1)
            emojis: List of primary emoji characters
            augment: Whether to apply text augmentation
            minority_classes: List of minority class indices for augmentation
        """
        super().__init__(texts, emo_ids, hid_ids, emojis, augment, minority_classes)
        self.class_distribution = Counter(emo_ids)
        
        logger.info(f"Created dataset with {len(self.texts)} samples")
        logger.info(f"Class distribution: {dict(self.class_distribution)}")
    
    def __getitem__(self, idx: int) -> Tuple[str, int, int]:
        """
        Get a single sample - uses shared preprocessing.
        
        Returns:
            processed_text: Preprocessed text with context tokens
            emotion_id: Emotion class ID
            hidden_id: Hidden flag ID (0 or 1)
        """
        return super().__getitem__(idx)


def collate_fn(batch, tokenizer, max_length: int = 128):
    """
    Collate function for DataLoader.
    Tokenizes texts and creates batch tensors.
    
    Args:
        batch: List of (text, emotion_id, hidden_id) tuples
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with tokenized inputs and labels
    """
    texts, emo_ids, hid_ids = zip(*batch)
    
    # Tokenize texts
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Add labels
    enc["emotion_labels"] = torch.tensor(emo_ids, dtype=torch.long)
    enc["hidden_labels"] = torch.tensor(hid_ids, dtype=torch.float)
    
    return enc
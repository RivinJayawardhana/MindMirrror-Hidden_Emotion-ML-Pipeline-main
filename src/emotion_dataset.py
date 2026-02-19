"""
Dataset and data preprocessing for emotion classification.
Matches notebook implementation exactly with text augmentation.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import emoji
from collections import Counter
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Emotion detection keywords (from notebook)
NEGATIVE_KEYWORDS = [
    "hate", "angry", "mad", "furious", "sad", "depressed", "terrible",
    "cry", "crying", "die", "dead", "kill", "killing", "awful",
    "annoying", "stupid", "idiot", "worst", "bad", "horrible"
]

POSITIVE_KEYWORDS = [
    "love", "happy", "joy", "great", "wonderful", "amazing",
    "excellent", "perfect", "best", "good", "nice", "fantastic"
]


def emoji_to_description(ch: str) -> str:
    """Convert emoji character to semantic description."""
    # Handle NaN, None, or empty values
    if ch is None or (isinstance(ch, float) and np.isnan(ch)):
        return ""
    ch = str(ch).strip()
    if not ch:
        return ""
    try:
        desc = emoji.demojize(ch).strip(":").replace("_", " ")
        return desc
    except (TypeError, AttributeError):
        return ""


def has_negative_word(text: str) -> bool:
    """Check if text contains negative keywords."""
    t = text.lower()
    return any(neg in t for neg in NEGATIVE_KEYWORDS)


def has_positive_word(text: str) -> bool:
    """Check if text contains positive keywords."""
    t = text.lower()
    return any(pos in t for pos in POSITIVE_KEYWORDS)


def build_input(text: str, emoji_char: str = "") -> str:
    """
    Enhanced preprocessing with emoji context and conflict detection.
    Matches notebook implementation exactly.
    
    Args:
        text: Input text
        emoji_char: Primary emoji character
    
    Returns:
        Preprocessed text with context tokens
    """
    text = str(text).strip()
    desc = emoji_to_description(emoji_char)
    token_prefixes = []
    
    if desc:
        token_prefixes.append(f"[EMOJI={desc}]")
    
    # Enhanced conflict detection
    if desc:
        # Positive emojis with negative text
        positive_emoji_cues = ["smile", "grin", "laugh", "heart", "joy", "relieved", "wink", "blush"]
        negative_emoji_cues = ["angry", "cry", "sad", "fear", "scared", "worried", "pouting"]
        
        is_positive_emoji = any(cue in desc for cue in positive_emoji_cues)
        is_negative_emoji = any(cue in desc for cue in negative_emoji_cues)
        
        if is_positive_emoji and has_negative_word(text):
            token_prefixes.append("[CONFLICT_POS_EMOJI_NEG_TEXT]")
        elif is_negative_emoji and has_positive_word(text):
            token_prefixes.append("[CONFLICT_NEG_EMOJI_POS_TEXT]")
        
        # Special handling for common emojis
        if "smiling" in desc or "grinning" in desc:
            token_prefixes.append("[SMILE_EMOJI]")
        elif "heart" in desc:
            token_prefixes.append("[HEART_EMOJI]")
        elif "crying" in desc or "tear" in desc:
            token_prefixes.append("[CRY_EMOJI]")
        elif "angry" in desc:
            token_prefixes.append("[ANGRY_EMOJI]")
    
    # Add length indicator for hidden emotion detection
    if len(text.split()) > 15:
        token_prefixes.append("[LONG_TEXT]")
    
    prefix = " ".join(token_prefixes)
    if prefix:
        return prefix + " " + text
    return text


class EmotionHiddenDataset(Dataset):
    """
    PyTorch Dataset for emotion classification with text augmentation.
    Matches notebook implementation exactly.
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
        self.texts = list(texts)
        self.emo_ids = list(emo_ids)
        self.hid_ids = list(hid_ids)
        self.emojis = list(emojis)
        self.augment = augment
        self.minority_classes = minority_classes or [3, 4, 5]  # fear, love, surprise
        self.class_distribution = Counter(emo_ids)
        
        logger.info(f"Created dataset with {len(self.texts)} samples")
        logger.info(f"Class distribution: {dict(self.class_distribution)}")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, int, int]:
        """
        Get a single sample.
        
        Returns:
            processed_text: Preprocessed text with context tokens
            emotion_id: Emotion class ID
            hidden_id: Hidden flag ID (0 or 1)
        """
        raw_text = self.texts[idx]
        emoji_char = self.emojis[idx]
        # Convert NaN/None to empty string
        if emoji_char is None or (isinstance(emoji_char, float) and np.isnan(emoji_char)):
            emoji_char = ""
        else:
            emoji_char = str(emoji_char)
        emotion_id = self.emo_ids[idx]
        
        # Simple text augmentation for minority classes
        if self.augment and np.random.random() < 0.3:
            if emotion_id in self.minority_classes:
                # Add minor variations
                variations = [
                    f"I feel {raw_text}",
                    f"{raw_text} honestly",
                    f"To be honest, {raw_text}",
                    f"{raw_text} right now"
                ]
                raw_text = np.random.choice(variations)
        
        proc_text = build_input(raw_text, emoji_char)
        return proc_text, emotion_id, self.hid_ids[idx]


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

"""
Shared preprocessing module matching notebook implementation exactly.
All files should import from here to ensure consistency.
"""
import emoji
import numpy as np
import pandas as pd
from typing import List, Optional

# ============================================================================
# EXACT NOTEBOOK KEYWORDS
# ============================================================================
NEGATIVE_KEYWORDS = [
    "hate", "angry", "mad", "furious", "sad", "depressed", "terrible",
    "cry", "crying", "die", "dead", "kill", "killing", "awful",
    "annoying", "stupid", "idiot", "worst", "bad", "horrible"
]

POSITIVE_KEYWORDS = [
    "love", "happy", "joy", "great", "wonderful", "amazing",
    "excellent", "perfect", "best", "good", "nice", "fantastic"
]

# ============================================================================
# EXACT NOTEBOOK FUNCTIONS
# ============================================================================

def emoji_to_description(ch):
    """Exact notebook implementation"""
    if not ch or pd.isna(ch):
        return ""
    desc = emoji.demojize(str(ch)).strip(":").replace("_", " ")
    return desc

def has_negative_word(text):
    """Exact notebook implementation"""
    t = str(text).lower()
    return any(neg in t for neg in NEGATIVE_KEYWORDS)

def has_positive_word(text):
    """Exact notebook implementation"""
    t = str(text).lower()
    return any(pos in t for pos in POSITIVE_KEYWORDS)

def build_input(text, emoji_char):
    """
    Enhanced preprocessing - EXACT NOTEBOOK IMPLEMENTATION:
    1) Convert emoji to semantic description
    2) Detect emotion-text conflicts
    3) Add context tokens
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

def normalize_emotion_label(e):
    """Exact notebook implementation"""
    valid_emotions = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    
    e = str(e).strip().lower()
    if e in valid_emotions:
        return e
    if e == "happy":
        return "joy"
    if e in ["mad", "furious", "rage"]:
        return "anger"
    return None

def extract_primary_emoji(text):
    """Extract first emoji from text - notebook style"""
    text = str(text)
    for ch in text:
        if ch in emoji.EMOJI_DATA:
            return ch
    return ""

# ============================================================================
# Dataset Class (Exact Notebook Implementation)
# ============================================================================

class EmotionHiddenDataset:
    """Exact notebook dataset class with augmentation"""
    
    def __init__(self, texts, emo_ids, hid_ids, emojis, augment=False, minority_classes=None):
        self.texts = list(texts)
        self.emo_ids = list(emo_ids)
        self.hid_ids = list(hid_ids)
        self.emojis = list(emojis)
        self.augment = augment
        self.minority_classes = minority_classes or [3, 4, 5]  # fear, love, surprise

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        raw_text = self.texts[idx]
        emoji_char = self.emojis[idx] if idx < len(self.emojis) else ""
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
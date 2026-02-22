"""
Data preprocessing pipeline matching notebook workflow exactly with 3-task support.
Returns: 27-class emotion labels, binary hidden flag, and 6-class hidden emotion labels.
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import emoji

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_data_paths, get_emotion_categories
from src.data_ingestion import DataIngestorCSV
# IMPORT from shared preprocessing
from src.preprocessing import normalize_emotion_label, extract_primary_emoji

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def map_to_6_classes(emotion_27: str) -> str:
    """
    Map 27-class emotion to 6 basic emotions.
    Customize this mapping based on your GoEmotions taxonomy.
    """
    mapping = {
        # Joy group
        'joy': 'joy',
        'amusement': 'joy',
        'excitement': 'joy',
        'happiness': 'joy',
        'love': 'love',
        'desire': 'love',
        'optimism': 'joy',
        'relief': 'joy',
        'pride': 'joy',
        'admiration': 'joy',
        'gratitude': 'joy',
        'caring': 'love',
        
        # Sadness group
        'sadness': 'sadness',
        'disappointment': 'sadness',
        'embarrassment': 'sadness',
        'grief': 'sadness',
        'remorse': 'sadness',
        
        # Anger group
        'anger': 'anger',
        'annoyance': 'anger',
        'disapproval': 'anger',
        'disgust': 'anger',
        
        # Fear group
        'fear': 'fear',
        'nervousness': 'fear',
        'anxiety': 'fear',
        
        # Surprise group
        'surprise': 'surprise',
        'realization': 'surprise',
        'confusion': 'surprise',
        'curiosity': 'surprise',
        
        # Neutral/other - map to most appropriate
        'neutral': 'joy',  # Default to joy, or you can add 'neutral' as 7th class
    }
    
    # Return mapped value, default to 'joy' if not found
    return mapping.get(emotion_27.lower(), 'joy')


def emotion_data_pipeline(data_path: str = None):
    """
    End-to-end data preprocessing pipeline with 3-task support.
    
    Returns:
        Tuple of (train_data, val_data, label_encoder_27, label_encoder_6)
        where:
        - train_data: dict with 'texts', 'emo_ids', 'hid_ids', 'hidden6_ids', 'emojis'
        - val_data: dict with 'texts', 'emo_ids', 'hid_ids', 'hidden6_ids', 'emojis'
        - label_encoder_27: sklearn LabelEncoder for 27-class emotions
        - label_encoder_6: sklearn LabelEncoder for 6-class hidden emotions
    """
    data_paths = get_data_paths()
    emotion_categories = get_emotion_categories()
    
    # Use provided path or default
    if data_path is None:
        data_path = data_paths.get('raw_data', 'merged_full_dataset.csv')
    
    logger.info("=" * 60)
    logger.info("EMOTION DATA PREPROCESSING PIPELINE (3-TASK)")
    logger.info("=" * 60)
    logger.info(f"Loading data from: {data_path}")
    
    # Step 1: Load data
    try:
        ingestor = DataIngestorCSV()
        data = ingestor.ingest(data_path)
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples")
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        raise
    
    # Step 2: Clean and normalize text
    logger.info("\nCleaning and normalizing text...")
    df["text"] = df["text"].astype(str).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    
    # Step 3: Normalize emotion labels - USING SHARED FUNCTION
    logger.info("Normalizing 27-class emotion labels...")
    df["hidden_emotion_label"] = df["hidden_emotion_label"].apply(normalize_emotion_label)
    df = df[df["hidden_emotion_label"].notna()].reset_index(drop=True)
    logger.info(f"After normalization: {len(df)} samples")
    
    # Step 4: Create 6-class hidden emotion labels (NEW)
    logger.info("Creating 6-class hidden emotion labels...")
    df["hidden_6_label"] = df["hidden_emotion_label"].apply(map_to_6_classes)
    
    # Log 6-class distribution
    logger.info("\n6-class emotion distribution:")
    logger.info(df["hidden_6_label"].value_counts().to_string())
    
    # Step 5: Extract primary emoji if not present
    if "primary_emoji" not in df.columns or df["primary_emoji"].isna().all():
        logger.info("Extracting primary emoji from text...")
        df["primary_emoji"] = df["text"].apply(extract_primary_emoji)
    
    # Step 6: Encode 27-class labels
    logger.info("Encoding 27-class emotion labels...")
    le_27 = LabelEncoder()
    df["emotion_id"] = le_27.fit_transform(df["hidden_emotion_label"])
    
    # Step 7: Encode 6-class hidden emotion labels (NEW)
    logger.info("Encoding 6-class hidden emotion labels...")
    le_6 = LabelEncoder()
    df["hidden_6_id"] = le_6.fit_transform(df["hidden_6_label"])
    
    # Step 8: Binary hidden flag
    df["hidden_flag_id"] = df["hidden_emotion_flag"].astype(int)
    
    logger.info(f"\n27-class label order: {list(le_27.classes_)}")
    logger.info(f"6-class label order: {list(le_6.classes_)}")
    
    logger.info("\n27-class distribution:")
    logger.info(df["hidden_emotion_label"].value_counts().to_string())
    
    # Step 9: Train/Val split with stratification on 27-class
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING DATA")
    logger.info("=" * 60)
    
    X_train, X_val, y_train_em, y_val_em, y_train_hid, y_val_hid, y_train_h6, y_val_h6 = train_test_split(
        df["text"],
        df["emotion_id"],
        df["hidden_flag_id"],
        df["hidden_6_id"],
        test_size=0.2,
        random_state=42,
        stratify=df["emotion_id"],  # Stratify on 27-class
    )
    
    train_emojis = df.loc[X_train.index, "primary_emoji"].values
    val_emojis = df.loc[X_val.index, "primary_emoji"].values
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Prepare return data with 3 labels (27-class, binary flag, 6-class)
    train_data = {
        'texts': X_train.values.tolist(),
        'emo_ids': y_train_em.values.tolist(),        # 27-class
        'hid_ids': y_train_hid.values.tolist(),       # binary flag
        'hidden6_ids': y_train_h6.values.tolist(),    # 6-class hidden (NEW)
        'emojis': train_emojis.tolist(),
    }
    
    val_data = {
        'texts': X_val.values.tolist(),
        'emo_ids': y_val_em.values.tolist(),          # 27-class
        'hid_ids': y_val_hid.values.tolist(),         # binary flag
        'hidden6_ids': y_val_h6.values.tolist(),      # 6-class hidden (NEW)
        'emojis': val_emojis.tolist(),
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train data keys: {list(train_data.keys())}")
    logger.info(f"Train samples: {len(train_data['texts'])}")
    logger.info(f"Val samples: {len(val_data['texts'])}")
    
    # Return 4 values: train_data, val_data, 27-class encoder, 6-class encoder
    return train_data, val_data, le_27, le_6


if __name__ == "__main__":
    train_data, val_data, le_27, le_6 = emotion_data_pipeline()
    logger.info(f"\nTrain data keys: {train_data.keys()}")
    logger.info(f"27-class encoder classes: {list(le_27.classes_)}")
    logger.info(f"6-class encoder classes: {list(le_6.classes_)}")
    
    # Quick validation
    logger.info(f"\nSample train item:")
    logger.info(f"  Text: {train_data['texts'][0][:50]}...")
    logger.info(f"  27-class ID: {train_data['emo_ids'][0]}")
    logger.info(f"  Binary flag: {train_data['hid_ids'][0]}")
    logger.info(f"  6-class ID: {train_data['hidden6_ids'][0]}")
    logger.info(f"  Emoji: {train_data['emojis'][0]}")
"""
Data preprocessing pipeline matching notebook workflow exactly.
Updated to work with relabeled dataset (true_emotion, is_hidden columns)
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
from src.preprocessing import extract_primary_emoji

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def emotion_data_pipeline(data_path: str = None):
    """
    End-to-end data preprocessing pipeline for relabeled dataset.
    
    Returns:
        Tuple of (train_data, val_data, dummy_encoder, label_encoder_6)
        where:
        - train_data: dict with 'texts', 'hid_ids', 'hidden6_ids', 'emojis'
        - val_data: dict with 'texts', 'hid_ids', 'hidden6_ids', 'emojis'
        - dummy_encoder: placeholder (not used)
        - label_encoder_6: sklearn LabelEncoder for 6-class emotions
    """
    data_paths = get_data_paths()
    
    # Use provided path or default
    if data_path is None:
        data_path = data_paths.get('raw_data', 'merged_full_dataset.csv')
    
    logger.info("=" * 60)
    logger.info("EMOTION DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Loading data from: {data_path}")
    
    # Step 1: Load data
    try:
        ingestor = DataIngestorCSV()
        data = ingestor.ingest(data_path)
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Columns available: {list(df.columns)}")
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        raise
    
    # Step 2: Clean and normalize text
    logger.info("\nCleaning and normalizing text...")
    df["text"] = df["text"].astype(str).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    
    # Step 3: Handle emotion labels - check for relabeled columns
    logger.info("Checking for emotion label columns...")
    
    if 'true_emotion' in df.columns:
        logger.info("✓ Using 'true_emotion' as primary emotion label")
        emotion_col = 'true_emotion'
        df['emotion_6class'] = df['true_emotion']
    elif 'hidden_emotion_label' in df.columns:
        logger.info("Using 'hidden_emotion_label' - will map to 6 classes")
        # This would need mapping function, but for relabeled data we don't need this path
        df['emotion_6class'] = df['hidden_emotion_label']
    else:
        raise KeyError("No emotion label column found. Expected 'true_emotion'")
    
    # Step 4: Handle hidden flag
    logger.info("Checking for hidden flag column...")
    if 'is_hidden' in df.columns:
        logger.info("✓ Using 'is_hidden' as hidden flag")
        df["hidden_flag"] = df["is_hidden"].astype(int)
    elif 'hidden_emotion_flag' in df.columns:
        logger.info("Using 'hidden_emotion_flag' as hidden flag")
        df["hidden_flag"] = df["hidden_emotion_flag"].astype(int)
    else:
        logger.warning("No hidden flag column found. Setting default to 0.")
        df["hidden_flag"] = 0
    
    # Step 5: Extract primary emoji if not present
    if "primary_emoji" not in df.columns or df["primary_emoji"].isna().all():
        logger.info("Extracting primary emoji from text...")
        df["primary_emoji"] = df["text"].apply(extract_primary_emoji)
    else:
        logger.info(f"✓ Using existing primary_emoji column")
    
    # Step 6: Encode 6-class labels
    logger.info("Encoding 6-class emotion labels...")
    le_6 = LabelEncoder()
    df["emotion_id"] = le_6.fit_transform(df["emotion_6class"])
    
    # Log 6-class distribution
    logger.info("\n📊 6-class emotion distribution:")
    dist = df["emotion_6class"].value_counts()
    for emotion, count in dist.items():
        logger.info(f"  {emotion:10s}: {count:4d} ({count/len(df)*100:5.1f}%)")
    
    logger.info(f"\n6-class label order: {list(le_6.classes_)}")
    
    # Step 7: Train/Val split with stratification on 6-class
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING DATA")
    logger.info("=" * 60)
    
    # Simple split - only what we need
    X_train, X_val, y_train_hid, y_val_hid, y_train_emo, y_val_emo = train_test_split(
        df["text"],
        df["hidden_flag"],
        df["emotion_id"],
        test_size=0.2,
        random_state=42,
        stratify=df["emotion_id"],  # Stratify on emotion
    )
    
    # Get corresponding emojis
    train_emojis = df.loc[X_train.index, "primary_emoji"].fillna('').values
    val_emojis = df.loc[X_val.index, "primary_emoji"].fillna('').values
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Prepare return data (only what the training pipeline needs)
    train_data = {
        'texts': X_train.values.tolist(),
        'hid_ids': y_train_hid.values.tolist(),        # binary flag (is_hidden)
        'hidden6_ids': y_train_emo.values.tolist(),    # 6-class emotion (true_emotion)
        'emojis': train_emojis.tolist(),
    }
    
    val_data = {
        'texts': X_val.values.tolist(),
        'hid_ids': y_val_hid.values.tolist(),          # binary flag (is_hidden)
        'hidden6_ids': y_val_emo.values.tolist(),      # 6-class emotion (true_emotion)
        'emojis': val_emojis.tolist(),
    }
    
    # Create dummy encoder for 27-class (for compatibility with training pipeline)
    dummy_encoder = LabelEncoder()
    dummy_encoder.classes_ = np.array(['dummy'])
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Train data keys: {list(train_data.keys())}")
    logger.info(f"Train samples: {len(train_data['texts'])}")
    logger.info(f"Val samples: {len(val_data['texts'])}")
    logger.info(f"Train hid_ids sample: {train_data['hid_ids'][:5]}")
    logger.info(f"Train hidden6_ids sample: {train_data['hidden6_ids'][:5]}")
    
    # Return 4 values: train_data, val_data, dummy_27_encoder, 6_encoder
    return train_data, val_data, dummy_encoder, le_6


if __name__ == "__main__":
    train_data, val_data, dummy_enc, le_6 = emotion_data_pipeline()
    logger.info(f"\n✅ Test successful!")
    logger.info(f"Train data keys: {train_data.keys()}")
    logger.info(f"6-class encoder classes: {list(le_6.classes_)}")
    
    # Quick validation
    logger.info(f"\nSample train item:")
    logger.info(f"  Text: {train_data['texts'][0][:50]}...")
    logger.info(f"  Hidden flag: {train_data['hid_ids'][0]}")
    logger.info(f"  6-class ID: {train_data['hidden6_ids'][0]}")
    logger.info(f"  Emoji: {train_data['emojis'][0]}")
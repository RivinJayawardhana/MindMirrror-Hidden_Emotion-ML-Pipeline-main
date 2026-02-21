"""
Data preprocessing pipeline matching notebook workflow exactly.
No embeddings - just preprocessing and train/val/test split.
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


def emotion_data_pipeline(data_path: str = None):
    """
    End-to-end data preprocessing pipeline.
    Matches notebook workflow: Load → Clean → Normalize → Split
    
    Returns:
        Tuple of (train_data, val_data, label_encoder)
    """
    data_paths = get_data_paths()
    emotion_categories = get_emotion_categories()
    
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
    except Exception as e:
        logger.exception(f"Data ingestion failed: {e}")
        raise
    
    # Step 2: Clean and normalize text
    logger.info("\nCleaning and normalizing text...")
    df["text"] = df["text"].astype(str).str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    
    # Step 3: Normalize emotion labels - USING SHARED FUNCTION
    logger.info("Normalizing emotion labels...")
    df["hidden_emotion_label"] = df["hidden_emotion_label"].apply(normalize_emotion_label)
    df = df[df["hidden_emotion_label"].notna()].reset_index(drop=True)
    logger.info(f"After normalization: {len(df)} samples")
    
    # Step 4: Extract primary emoji if not present - USING SHARED FUNCTION
    if "primary_emoji" not in df.columns or df["primary_emoji"].isna().all():
        logger.info("Extracting primary emoji from text...")
        df["primary_emoji"] = df["text"].apply(extract_primary_emoji)
    
    # Step 5: Encode labels
    logger.info("Encoding emotion labels...")
    le = LabelEncoder()
    df["emotion_id"] = le.fit_transform(df["hidden_emotion_label"])
    df["hidden_flag_id"] = df["hidden_emotion_flag"].astype(int)
    
    logger.info(f"Label order: {list(le.classes_)}")
    logger.info("\nClass distribution:")
    logger.info(df["hidden_emotion_label"].value_counts().to_string())
    
    # Step 6: Train/Val split with stratification
    logger.info("\n" + "=" * 60)
    logger.info("SPLITTING DATA")
    logger.info("=" * 60)
    
    X_train, X_val, y_train_em, y_val_em, y_train_hid, y_val_hid = train_test_split(
        df["text"],
        df["emotion_id"],
        df["hidden_flag_id"],
        test_size=0.2,
        random_state=42,
        stratify=df["emotion_id"],
    )
    
    train_emojis = df.loc[X_train.index, "primary_emoji"].values
    val_emojis = df.loc[X_val.index, "primary_emoji"].values
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    # Prepare return data
    train_data = {
        'texts': X_train.values.tolist(),
        'emo_ids': y_train_em.values.tolist(),
        'hid_ids': y_train_hid.values.tolist(),
        'emojis': train_emojis.tolist(),
    }
    
    val_data = {
        'texts': X_val.values.tolist(),
        'emo_ids': y_val_em.values.tolist(),
        'hid_ids': y_val_hid.values.tolist(),
        'emojis': val_emojis.tolist(),
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    
    return train_data, val_data, le


if __name__ == "__main__":
    train_data, val_data, le = emotion_data_pipeline()
    logger.info(f"\nTrain data keys: {train_data.keys()}")
    logger.info(f"Val data keys: {val_data.keys()}")
    logger.info(f"Label encoder classes: {list(le.classes_)}")
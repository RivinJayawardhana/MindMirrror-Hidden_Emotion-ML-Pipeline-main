#!/usr/bin/env python3
"""
Convenience script for training with dynamic datasets.
Wrapper around emotion_train_pipeline.py with enhanced CLI.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipelines.emotion_train_pipeline import emotion_train_pipeline
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='🚀 Train Hidden Emotion Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 Examples:
  
  # Use default dataset from config.yaml
  python train.py
  
  # Use dataset by name from config.yaml
  python train.py --dataset balanced
  
  # Use custom dataset path
  python train.py --data-path "data/my_dataset.csv"
  
  # Use custom dataset with custom experiment name
  python train.py --dataset balanced --experiment "experiment_v2"
  
  # List available datasets
  python train.py --list-datasets
  
  # Quick training with minimal epochs (for testing)
  python train.py --dataset balanced --epochs 2

💡 Tips:
  - Add your datasets to config.yaml under 'data_paths.datasets'
  - Use --list-datasets to see all available datasets
  - MLflow automatically tracks experiments (run 'mlflow ui' to view)
        """
    )
    
    parser.add_argument(
        '--data-path', '--data_path',
        type=str,
        default=None,
        help='📁 Direct path to dataset CSV file (overrides config)'
    )
    
    parser.add_argument(
        '--dataset', '--dataset-name',
        type=str,
        default=None,
        dest='dataset_name',
        help='📦 Name of dataset from config.yaml datasets section'
    )
    
    parser.add_argument(
        '--list-datasets', '--list_datasets',
        action='store_true',
        help='📋 List available datasets from config.yaml and exit'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default=None,
        help='🔬 MLflow experiment name (default: from config.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='⚙️  Number of training epochs (overrides config.yaml)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        dest='batch_size',
        help='⚙️  Training batch size (overrides config.yaml)'
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        from utils.config import load_config, get_data_paths
        config = load_config()
        data_paths = get_data_paths()
        datasets = data_paths.get('datasets', {})
        
        print("\n" + "=" * 70)
        print("📦 AVAILABLE DATASETS")
        print("=" * 70)
        if datasets:
            for name, path in datasets.items():
                exists = "✅" if os.path.exists(path) else "❌"
                print(f"  {exists} {name:20s} -> {path}")
        else:
            print("  No datasets configured in config.yaml")
        print("=" * 70)
        print(f"\n📁 Default dataset: {data_paths.get('raw_data', 'N/A')}")
        print("\n💡 To add datasets, edit config.yaml under 'data_paths.datasets'")
        print("=" * 70 + "\n")
        return
    
    # Validate arguments
    if args.data_path and args.dataset_name:
        logger.warning("⚠️  Both --data-path and --dataset provided. Using --data-path.")
        args.dataset_name = None
    
    # Override config if needed (set environment variables)
    if args.epochs:
        os.environ['OVERRIDE_EPOCHS'] = str(args.epochs)
        logger.info(f"⚙️  Overriding epochs: {args.epochs}")
    
    if args.batch_size:
        os.environ['OVERRIDE_BATCH_SIZE'] = str(args.batch_size)
        logger.info(f"⚙️  Overriding batch_size: {args.batch_size}")
    
    if args.experiment:
        os.environ['OVERRIDE_EXPERIMENT'] = args.experiment
        logger.info(f"🔬 Overriding experiment name: {args.experiment}")
    
    # Run training
    logger.info("🚀 Starting training pipeline...")
    try:
        emotion_train_pipeline(data_path=args.data_path, dataset_name=args.dataset_name)
        logger.info("✅ Training completed successfully!")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

"""
Training pipeline with transformer fine-tuning.
Matches notebook workflow exactly with all features:
- Transformer fine-tuning (not embeddings)
- Multi-task learning
- Early stopping
- MLflow logging
- Class weighting
- Gradient clipping
- Mixed precision training
- Layer-wise learning rate decay
- Cyclic learning rates
- Model checkpointing
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from functools import partial
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)
import logging
from torch.amp import autocast
from torch.cuda.amp import GradScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config, get_data_paths, get_emotion_categories
from utils.model_loader import get_pretrained_model_path, get_pretrained_tokenizer_path
from src.emotion_model import EnhancedEmotionHiddenModel, EnhancedMultitaskLoss
from src.emotion_dataset import EmotionHiddenDataset, collate_fn
from pipelines.emotion_data_pipeline import emotion_data_pipeline
from utils.mlflow_utils import init_mlflow, log_pytorch_model, log_label_encoder, log_training_config
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Enhanced early stopping with patience and min delta."""
    
    def __init__(self, patience: int = 3, monitor: str = "val_emo_accuracy", 
                 mode: str = "max", min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait
            monitor: Metric to monitor
            mode: 'max' or 'min'
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.best_epoch = 0
        self.early_stop = False
    
    def __call__(self, score: float, model_state_dict: dict, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if early stopping triggered
        """
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state_dict.items()}
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state_dict.items()}
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
                return True
            return False


def calculate_class_weights(emo_ids: list, method: str = "effective_num", 
                           beta: float = 0.999, smooth: float = 1.0) -> dict:
    """
    Calculate class weights for imbalanced data with smoothing.
    
    Args:
        emo_ids: List of emotion class IDs
        method: 'effective_num', 'inverse_freq', or 'sqrt_inverse'
        beta: Smoothing factor for effective number method
        smooth: Smoothing factor for zero counts
    
    Returns:
        Dict mapping class_idx -> weight
    """
    emo_counts = Counter(emo_ids)
    num_classes = len(emo_counts)
    
    # Add smoothing
    for i in range(num_classes):
        if i not in emo_counts:
            emo_counts[i] = smooth
    
    if method == "effective_num":
        # Effective number of samples (from notebook)
        effective_num = 1.0 - np.power(beta, list(emo_counts.values()))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        class_weights = {i: float(w) for i, w in enumerate(weights)}
    elif method == "sqrt_inverse":
        # Square root inverse frequency (less aggressive than inverse)
        total = sum(emo_counts.values())
        weights = [np.sqrt(total / (num_classes * count)) for count in emo_counts.values()]
        weights = weights / np.sum(weights) * num_classes
        class_weights = {i: float(w) for i, w in enumerate(weights)}
    else:  # inverse_freq
        # Inverse frequency weighting
        total = sum(emo_counts.values())
        class_weights = {i: total / (num_classes * count) for i, count in emo_counts.items()}
    
    return class_weights


def evaluate_model(model, dataloader, criterion=None, device="cuda", use_amp=False):
    """
    Evaluate model on validation/test set with comprehensive metrics.
    """
    model.eval()
    all_emo_preds = []
    all_hid_preds = []
    all_emo_labels = []
    all_hid_labels = []
    all_emo_probs = []
    all_hid_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            with autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                emo_logits, hid_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                if criterion:
                    loss, _, _ = criterion(
                        emo_logits, batch["emotion_labels"],
                        hid_logits, batch["hidden_labels"]
                    )
                    total_loss += loss.item()
            
            emo_probs = F.softmax(emo_logits, dim=-1)
            hid_probs = torch.sigmoid(hid_logits)
            
            emo_preds = emo_logits.argmax(dim=-1)
            hid_preds = (hid_probs > 0.5).long()
            
            all_emo_preds.extend(emo_preds.cpu().numpy())
            all_hid_preds.extend(hid_preds.cpu().numpy())
            all_emo_labels.extend(batch["emotion_labels"].cpu().numpy())
            all_hid_labels.extend(batch["hidden_labels"].cpu().numpy())
            all_emo_probs.extend(emo_probs.cpu().numpy())
            all_hid_probs.extend(hid_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_emo_labels = np.array(all_emo_labels)
    all_emo_preds = np.array(all_emo_preds)
    all_hid_labels = np.array(all_hid_labels)
    all_hid_preds = np.array(all_hid_preds)
    
    metrics = {
        "emo_accuracy": accuracy_score(all_emo_labels, all_emo_preds),
        "emo_f1_macro": f1_score(all_emo_labels, all_emo_preds, average='macro'),
        "emo_f1_weighted": f1_score(all_emo_labels, all_emo_preds, average='weighted'),
        "hid_accuracy": accuracy_score(all_hid_labels, all_hid_preds),
        "hid_precision": precision_score(all_hid_labels, all_hid_preds, zero_division=0),
        "hid_recall": recall_score(all_hid_labels, all_hid_preds, zero_division=0),
        "hid_f1": f1_score(all_hid_labels, all_hid_preds, zero_division=0),
    }
    
    try:
        metrics["hid_auc"] = roc_auc_score(all_hid_labels, all_hid_preds)
    except:
        metrics["hid_auc"] = 0.0
    
    if criterion:
        metrics["loss"] = total_loss / len(dataloader)
    
    return metrics


def train_model(config: dict, train_data: dict, val_data: dict, label_encoder, device: str):
    """
    Main training function with all accuracy-improving features.
    """
    train_cfg = config['training']
    model_cfg = config['model']
    tokenizer_cfg = config['tokenizer']
    
    # Mixed precision training
    use_amp = train_cfg.get('use_mixed_precision', True) and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    
    # Load model/tokenizer from local cache
    model_path = get_pretrained_model_path(model_cfg['base_model_name'])
    tokenizer_path = get_pretrained_tokenizer_path(model_cfg['base_model_name'])
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Create datasets
    train_ds = EmotionHiddenDataset(
        texts=train_data['texts'],
        emo_ids=train_data['emo_ids'],
        hid_ids=train_data['hid_ids'],
        emojis=train_data['emojis'],
        augment=train_cfg['augmentation']['enabled'],
        minority_classes=train_cfg['augmentation']['minority_classes']
    )
    
    val_ds = EmotionHiddenDataset(
        texts=val_data['texts'],
        emo_ids=val_data['emo_ids'],
        hid_ids=val_data['hid_ids'],
        emojis=val_data['emojis'],
        augment=False
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(
        train_data['emo_ids'],
        method=train_cfg['class_weights']['method'],
        beta=train_cfg['class_weights']['beta']
    )
    logger.info(f"Class weights: {class_weights}")
    
    # Create weighted sampler
    sample_weights = [class_weights.get(c, 1.0) for c in train_data['emo_ids']]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders
    _collate = partial(collate_fn, tokenizer=tokenizer, max_length=tokenizer_cfg['max_length'])
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        sampler=sampler,
        collate_fn=_collate,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        collate_fn=_collate,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Initialize model with gradient checkpointing
    model = EnhancedEmotionHiddenModel(
        base_model_name=model_cfg['base_model_name'],
        num_emotions=model_cfg['num_emotions'],
        dropout_p=model_cfg['dropout'],
        local_model_path=model_path,
        use_gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        hidden_size_factor=train_cfg.get('hidden_size_factor', 2)
    )
    
    # FIX: Ensure consistent dtype
    model = model.to(device)
    model = model.to_consistent_dtype(torch.float32)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Freeze encoder layers
    model.freeze_encoder_layers(num_layers=model_cfg['freeze_layers'])
    
    # Initialize loss function
    loss_cfg = train_cfg['loss']
    criterion = EnhancedMultitaskLoss(
        class_weights_dict=class_weights,
        gamma=loss_cfg['focal_gamma'],
        hidden_weight=loss_cfg['hidden_weight'],
        pos_weight=loss_cfg['pos_weight'],
        label_smoothing=loss_cfg['label_smoothing'],
        device=str(device),
        use_uncertainty_weighting=train_cfg.get('uncertainty_weighting', True)
    )
    
    # Setup optimizer with layer-wise learning rates
    if train_cfg.get('layerwise_lr_decay', False):
        # Use layer-wise decay
        param_groups = model.get_layerwise_lr_decay(
            base_lr=float(train_cfg['learning_rate_head']),
            decay_factor=train_cfg.get('lr_decay_factor', 0.95)
        )
        logger.info(f"Using layer-wise LR decay with {len(param_groups)} groups")
    else:
        # Standard parameter grouping
        encoder_params = []
        head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "encoder" in name:
                    encoder_params.append(param)
                else:
                    head_params.append(param)
        
        param_groups = [
            {"params": encoder_params, "lr": float(train_cfg['learning_rate_encoder']), 
             "weight_decay": float(train_cfg['weight_decay'])},
            {"params": head_params, "lr": float(train_cfg['learning_rate_head']), 
             "weight_decay": float(train_cfg['weight_decay'])},
        ]
    
    optimizer = torch.optim.AdamW(
        param_groups,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Setup scheduler
    num_training_steps = int(train_cfg['num_epochs'] * len(train_loader))
    w = train_cfg['warmup_steps']
    warmup_steps = int(w * num_training_steps) if isinstance(w, float) else int(w)
    num_training_steps = max(1, num_training_steps)
    warmup_steps = min(warmup_steps, num_training_steps - 1) if num_training_steps > 1 else 0
    
    scheduler_type = train_cfg.get('scheduler_type', 'cosine')
    if scheduler_type == "cosine_with_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps,
            num_cycles=train_cfg.get('num_cycles', 2)
        )
    elif scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=num_training_steps
        )
    
    # Early stopping
    early_stopping = None
    if train_cfg['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=train_cfg['early_stopping']['patience'],
            monitor=train_cfg['early_stopping']['monitor'],
            mode=train_cfg['early_stopping']['mode'],
            min_delta=train_cfg.get('min_delta', 0.001)
        )
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("STARTING TRAINING")
    logger.info("=" * 60)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(train_cfg['num_epochs']):
        logger.info(f"\n{'='*40}")
        logger.info(f"EPOCH {epoch + 1}/{train_cfg['num_epochs']}")
        logger.info(f"{'='*40}")
        
        # Training phase
        model.train()
        train_loss = 0
        emo_correct = 0
        hid_correct = 0
        total_samples = 0
        
        nan_batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast('cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp):
                emo_logits, hid_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                loss, l_emo, l_hid = criterion(
                    emo_logits, batch["emotion_labels"],
                    hid_logits, batch["hidden_labels"]
                )
            
            # Skip backward pass if loss is NaN/inf
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batch_count += 1
                if nan_batch_count > len(train_loader) * 0.5:
                    logger.error(f"Too many NaN batches ({nan_batch_count}/{batch_idx + 1})")
                    raise RuntimeError("Model training unstable")
                logger.warning(f"NaN/Inf loss at batch {batch_idx + 1}, skipping")
                continue
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                       max_norm=train_cfg['max_grad_norm'])
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                logger.warning(f"NaN/Inf gradients at batch {batch_idx + 1}, skipping")
                scaler.update()
                optimizer.zero_grad()
                continue
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Calculate batch accuracy
            emo_preds = emo_logits.argmax(dim=-1)
            hid_preds = (torch.sigmoid(hid_logits) > 0.5).float()
            
            emo_correct += (emo_preds == batch["emotion_labels"]).sum().item()
            hid_targets_float = batch["hidden_labels"].float()
            hid_correct += (hid_preds == hid_targets_float).sum().item()
            total_samples += len(batch["emotion_labels"])
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                train_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                if total_samples > 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                              f"Loss: {loss.item():.4f} | "
                              f"Emo Acc: {emo_correct/total_samples:.3f} | "
                              f"Hid Acc: {hid_correct/total_samples:.3f}")
        
        # Calculate epoch metrics
        valid_batches = len(train_loader) - nan_batch_count
        avg_train_loss = train_loss / max(1, valid_batches)
        train_emo_acc = emo_correct / total_samples if total_samples > 0 else 0.0
        train_hid_acc = hid_correct / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"\nTraining Summary:")
        logger.info(f"  Avg Loss: {avg_train_loss:.4f}")
        logger.info(f"  Emotion Accuracy: {train_emo_acc:.3f}")
        logger.info(f"  Hidden Accuracy: {train_hid_acc:.3f}")
        logger.info(f"  Current LR: {optimizer.param_groups[-1]['lr']:.2e}")
        
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_emo_accuracy": train_emo_acc,
                "train_hid_accuracy": train_hid_acc,
                "learning_rate": optimizer.param_groups[-1]['lr']
            }, step=epoch + 1)
        
        # Validation phase
        logger.info(f"\nValidation Results:")
        val_metrics = evaluate_model(model, val_loader, criterion, device, use_amp)
        
        # Log all validation metrics
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.3f}")
                if mlflow.active_run():
                    mlflow.log_metric(f"val_{key}", value, step=epoch + 1)
        
        # Early stopping check
        monitor_metric = val_metrics.get(train_cfg['early_stopping']['monitor'], 
                                        val_metrics['emo_accuracy'])
        
        if early_stopping:
            if early_stopping(monitor_metric, model.state_dict(), epoch):
                logger.info(f"Loading best model from epoch {early_stopping.best_epoch + 1}")
                model.load_state_dict(early_stopping.best_model_state)
                break
        
        # Save best model
        if val_metrics['emo_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['emo_accuracy']
            best_epoch = epoch
            data_paths = get_data_paths()
            model_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['model_artifacts_dir'])
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, 'best_emotion_model.pt')
            
            # Save with metadata
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder': label_encoder,
                'class_names': list(label_encoder.classes_),
                'tokenizer': tokenizer,
                'config': config,
                'val_metrics': val_metrics,
                'best_val_acc': best_val_acc
            }, model_path)
            logger.info(f"  ✓ New best model saved! (Acc: {best_val_acc:.3f})")
    
    # Final evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    final_metrics = evaluate_model(model, val_loader, criterion, device, use_amp)
    
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.3f}")
    
    # Log best epoch info
    logger.info(f"\nBest model from epoch {best_epoch + 1} with accuracy: {best_val_acc:.3f}")
    
    return model, final_metrics, tokenizer


def emotion_train_pipeline(data_path: str = None, dataset_name: str = None):
    """
    Main training pipeline entry point.
    """
    # Load configuration
    config = load_config()
    
    # Override config from environment variables
    import os
    if 'OVERRIDE_EPOCHS' in os.environ:
        config['training']['num_epochs'] = int(os.environ['OVERRIDE_EPOCHS'])
    if 'OVERRIDE_BATCH_SIZE' in os.environ:
        config['training']['batch_size'] = int(os.environ['OVERRIDE_BATCH_SIZE'])
    if 'OVERRIDE_EXPERIMENT' in os.environ:
        config['mlflow']['experiment_name'] = os.environ['OVERRIDE_EXPERIMENT']
    
    data_paths = get_data_paths()
    
    # Determine dataset path
    if data_path:
        dataset_path = data_path
        logger.info(f"Using provided dataset path: {dataset_path}")
    elif dataset_name:
        datasets = data_paths.get('datasets', {})
        if dataset_name in datasets:
            dataset_path = datasets[dataset_name]
            logger.info(f"Using dataset '{dataset_name}': {dataset_path}")
        else:
            available = list(datasets.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")
    else:
        dataset_path = data_paths.get('raw_data', 'merged_full_dataset.csv')
        logger.info(f"Using default dataset: {dataset_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize MLflow
    if config.get('mlflow', {}).get('tracking_enabled', True):
        experiment_name = config['mlflow']['experiment_name']
        if dataset_name:
            experiment_name = f"{experiment_name}_{dataset_name}"
        init_mlflow(
            tracking_uri=data_paths.get('mlflow_tracking_uri'),
            experiment_name=experiment_name
        )
    
    # Load and preprocess data
    train_data, val_data, label_encoder = emotion_data_pipeline(data_path=dataset_path)
    
    # Start MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "model_name": config['model']['base_model_name'],
            "num_epochs": config['training']['num_epochs'],
            "batch_size": config['training']['batch_size'],
            "lr_encoder": config['training']['learning_rate_encoder'],
            "lr_head": config['training']['learning_rate_head'],
            "max_length": config['tokenizer']['max_length'],
            "dataset_path": dataset_path,
            "dataset_name": dataset_name or "default",
            "focal_gamma": config['training']['loss']['focal_gamma'],
            "label_smoothing": config['training']['loss']['label_smoothing'],
            "hidden_weight": config['training']['loss']['hidden_weight'],
            "pos_weight": config['training']['loss']['pos_weight'],
            "scheduler_type": config['training'].get('scheduler_type', 'cosine'),
            "layerwise_lr_decay": config['training'].get('layerwise_lr_decay', False),
            "gradient_checkpointing": config['training'].get('gradient_checkpointing', False),
            "mixed_precision": config['training'].get('use_mixed_precision', True)
        })
        
        # Train model
        model, metrics, tokenizer = train_model(config, train_data, val_data, label_encoder, device)
        
        # Log final metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"final_{key}", value)
        
        # Log artifacts
        if config.get('mlflow', {}).get('log_artifacts', True):
            log_pytorch_model(model, artifact_path="model")
            log_label_encoder(label_encoder)
            log_training_config({
                "class_names": list(label_encoder.classes_),
                "model_config": config['model'],
                "training_config": config['training'],
                "final_metrics": metrics
            })
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    return model, metrics, tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train emotion detection model with dynamic dataset support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default dataset
  python pipelines/emotion_train_pipeline.py
  
  # Use dataset by name
  python pipelines/emotion_train_pipeline.py --dataset balanced
  
  # Use custom dataset path
  python pipelines/emotion_train_pipeline.py --data-path "path/to/dataset.csv"
  
  # List available datasets
  python pipelines/emotion_train_pipeline.py --list-datasets
        """
    )
    
    parser.add_argument(
        '--data-path', '--data_path',
        type=str,
        default=None,
        help='Direct path to dataset CSV file'
    )
    
    parser.add_argument(
        '--dataset', '--dataset-name',
        type=str,
        default=None,
        dest='dataset_name',
        help='Name of dataset from config.yaml datasets section'
    )
    
    parser.add_argument(
        '--list-datasets', '--list_datasets',
        action='store_true',
        help='List available datasets from config.yaml and exit'
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list_datasets:
        config = load_config()
        data_paths = get_data_paths()
        datasets = data_paths.get('datasets', {})
        print("\nAvailable datasets in config.yaml:")
        print("=" * 60)
        for name, path in datasets.items():
            print(f"  {name:20s} -> {path}")
        print("=" * 60)
        print(f"\nDefault dataset: {data_paths.get('raw_data', 'N/A')}")
        exit(0)
    
    # Validate arguments
    if args.data_path and args.dataset_name:
        logger.warning("Both --data-path and --dataset provided. Using --data-path.")
        args.dataset_name = None
    
    # Run training
    emotion_train_pipeline(data_path=args.data_path, dataset_name=args.dataset_name)
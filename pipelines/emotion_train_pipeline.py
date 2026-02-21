"""
Training pipeline with transformer fine-tuning.
Enhanced with accuracy improvements from research best practices.
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import Counter
from functools import partial
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer, 
    get_cosine_schedule_with_warmup, 
    get_linear_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support
)
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config, get_data_paths, get_emotion_categories
from utils.model_loader import get_pretrained_model_path, get_pretrained_tokenizer_path
from src.emotion_model import EnhancedEmotionHiddenModel, EnhancedMultitaskLoss
from pipelines.emotion_data_pipeline import emotion_data_pipeline
from utils.mlflow_utils import init_mlflow, log_pytorch_model, log_label_encoder, log_training_config
from src.preprocessing import EmotionHiddenDataset, build_input
import mlflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# IMPROVEMENT 1: Advanced Data Augmentation
# ============================================================================

class AdvancedAugmentation:
    """Advanced text augmentation techniques for minority classes"""
    
    def __init__(self, tokenizer, p=0.3):
        self.tokenizer = tokenizer
        self.p = p
        
        # Synonym replacement dictionary
        self.synonyms = {
            'happy': ['joyful', 'glad', 'pleased', 'delighted'],
            'sad': ['unhappy', 'down', 'depressed', 'gloomy'],
            'angry': ['mad', 'furious', 'irritated', 'annoyed'],
            'love': ['adore', 'cherish', 'treasure', 'care for'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious'],
            'surprise': ['shock', 'astonishment', 'amazement'],
            'good': ['great', 'wonderful', 'excellent', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor']
        }
    
    def random_deletion(self, text, p=0.2):
        """Randomly delete words with probability p"""
        words = text.split()
        if len(words) == 1:
            return text
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else random.choice(words)
    
    def random_swap(self, text, n=2):
        """Randomly swap two words n times"""
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(min(n, len(words))):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def synonym_replacement(self, text):
        """Replace words with synonyms"""
        words = text.split()
        new_words = []
        for word in words:
            if word.lower() in self.synonyms and random.random() < 0.3:
                new_words.append(random.choice(self.synonyms[word.lower()]))
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
    def augment(self, text, emotion_id, minority_classes):
        """Apply augmentation if sample is from minority class"""
        if emotion_id not in minority_classes or random.random() > self.p:
            return text
        
        # Choose random augmentation
        aug_type = random.choice(['delete', 'swap', 'synonym', 'prefix', 'suffix'])
        
        if aug_type == 'delete':
            return self.random_deletion(text)
        elif aug_type == 'swap':
            return self.random_swap(text)
        elif aug_type == 'synonym':
            return self.synonym_replacement(text)
        elif aug_type == 'prefix':
            prefixes = [
                "I feel", "Honestly,", "To be honest,", 
                "I think", "In my opinion,", "Personally,"
            ]
            return f"{random.choice(prefixes)} {text}"
        else:  # suffix
            suffixes = [
                " right now", " today", " honestly",
                " tbh", " tbh", " tbh"
            ]
            return f"{text}{random.choice(suffixes)}"

# ============================================================================
# IMPROVEMENT 2: MixUp for Emotion Classification
# ============================================================================

class MixUp:
    """MixUp augmentation for better generalization"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def mixup(self, x1, x2, y1, y2):
        """Apply mixup to two samples"""
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix inputs (for embeddings, not raw text)
        mixed_x = lam * x1 + (1 - lam) * x2
        
        # Mix labels
        mixed_y = lam * y1 + (1 - lam) * y2
        
        return mixed_x, mixed_y, lam

# ============================================================================
# IMPROVEMENT 3: Advanced Optimizer with Layer-wise Learning Rate Decay
# ============================================================================

def get_optimizer_with_llrd(model, base_lr=2e-5, decay_factor=0.95):
    """
    Layer-wise learning rate decay
    Lower layers get lower learning rates
    """
    optimizer_grouped_parameters = []
    
    # Get encoder layers if they exist
    if hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
        layers = list(model.encoder.encoder.layer)
        
        # Embeddings layer (lowest LR)
        embeddings_params = list(model.encoder.embeddings.parameters())
        if embeddings_params:
            optimizer_grouped_parameters.append({
                'params': embeddings_params,
                'lr': base_lr * (decay_factor ** 12),
                'name': 'embeddings'
            })
        
        # Encoder layers with decreasing LR
        for i, layer in enumerate(layers):
            layer_params = list(layer.parameters())
            if layer_params:
                lr = base_lr * (decay_factor ** (11 - i))
                optimizer_grouped_parameters.append({
                    'params': layer_params,
                    'lr': lr,
                    'name': f'encoder_layer_{i}'
                })
    else:
        # Fallback for models without standard layer structure
        logger.warning("Model does not have standard encoder layer structure, using all parameters with same LR")
        all_params = list(model.encoder.parameters())
        if all_params:
            optimizer_grouped_parameters.append({
                'params': all_params,
                'lr': base_lr,
                'name': 'encoder'
            })
    
    # Pooler parameters (if exists)
    if hasattr(model.encoder, 'pooler'):
        pooler_params = list(model.encoder.pooler.parameters())
        if pooler_params:
            optimizer_grouped_parameters.append({
                'params': pooler_params,
                'lr': base_lr,
                'name': 'pooler'
            })
    
    # Shared projection parameters
    if hasattr(model, 'shared_projection'):
        shared_params = list(model.shared_projection.parameters())
        if shared_params:
            optimizer_grouped_parameters.append({
                'params': shared_params,
                'lr': base_lr,
                'name': 'shared_projection'
            })
    
    # Emotion head parameters
    if hasattr(model, 'emotion_head'):
        emotion_params = list(model.emotion_head.parameters())
        if emotion_params:
            optimizer_grouped_parameters.append({
                'params': emotion_params,
                'lr': base_lr * 2,  # Higher LR for task-specific heads
                'name': 'emotion_head'
            })
    
    # Hidden head parameters
    if hasattr(model, 'hidden_head'):
        hidden_params = list(model.hidden_head.parameters())
        if hidden_params:
            optimizer_grouped_parameters.append({
                'params': hidden_params,
                'lr': base_lr * 2,
                'name': 'hidden_head'
            })
    
    # Temperature parameter
    if hasattr(model, 'temperature') and model.temperature is not None:
        optimizer_grouped_parameters.append({
            'params': [model.temperature],
            'lr': base_lr * 0.1,  # Very low LR for temperature
            'name': 'temperature'
        })
    
    # Ensure we have at least one parameter group
    if not optimizer_grouped_parameters:
        # Fallback: use all parameters
        all_params = list(model.parameters())
        if all_params:
            optimizer_grouped_parameters.append({
                'params': all_params,
                'lr': base_lr,
                'name': 'all_parameters'
            })
        else:
            raise ValueError("No parameters found in model")
    
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=base_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

# ============================================================================
# IMPROVEMENT 4: Advanced Evaluation with Confidence Analysis (FIXED dtype)
# ============================================================================

def evaluate_model_advanced(model, dataloader, criterion=None, device="cuda", use_amp=False):
    """
    Enhanced evaluation with confidence analysis and per-class metrics
    """
    model.eval()
    all_true_emo, all_pred_emo = [], []
    all_true_hid, all_pred_hid = [], []
    all_emo_probs = []
    all_hid_probs = []
    all_confidences = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device with correct dtypes
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).float()
            emotion_labels = batch["emotion_labels"].to(device).long()
            hidden_labels = batch["hidden_labels"].to(device).float()
            
            # Forward pass
            emo_logits, hid_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if criterion is not None:
                loss, _, _ = criterion(
                    emo_logits, emotion_labels,
                    hid_logits, hidden_labels
                )
                total_loss += loss.item()
                num_batches += 1
            
            emo_probs = F.softmax(emo_logits, dim=-1)
            hid_probs = torch.sigmoid(hid_logits)
            
            emo_preds = emo_logits.argmax(dim=-1)
            hid_preds = (hid_probs > 0.5).long()
            
            # Calculate confidence (max probability)
            confidences = emo_probs.max(dim=-1)[0]
            
            all_true_emo.extend(emotion_labels.cpu().numpy())
            all_pred_emo.extend(emo_preds.cpu().numpy())
            all_true_hid.extend(hidden_labels.cpu().numpy())
            all_pred_hid.extend(hid_preds.cpu().numpy())
            all_emo_probs.extend(emo_probs.cpu().numpy())
            all_hid_probs.extend(hid_probs.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to numpy arrays
    all_true_emo = np.array(all_true_emo)
    all_pred_emo = np.array(all_pred_emo)
    all_true_hid = np.array(all_true_hid)
    all_pred_hid = np.array(all_pred_hid)
    all_confidences = np.array(all_confidences)
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("EMOTION CLASSIFICATION REPORT")
    print("=" * 70)
    
    # Full classification report
    unique_classes = np.unique(np.concatenate([all_true_emo, all_pred_emo]))
    target_names = [f"class_{i}" for i in unique_classes]
    print(classification_report(
        all_true_emo,
        all_pred_emo,
        labels=unique_classes,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))
    
    # Per-class accuracy with confidence
    print("\n" + "-" * 70)
    print("PER-CLASS ACCURACY & CONFIDENCE ANALYSIS")
    print("-" * 70)
    cm = confusion_matrix(all_true_emo, all_pred_emo, labels=unique_classes)
    
    class_metrics = {}
    for i, class_id in enumerate(unique_classes):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        
        # Average confidence for this class
        class_mask = all_true_emo == class_id
        if class_mask.any():
            avg_conf = all_confidences[class_mask].mean()
            correct_mask = (all_true_emo == class_id) & (all_pred_emo == class_id)
            correct_conf = all_confidences[correct_mask].mean() if correct_mask.any() else 0
            incorrect_mask = (all_true_emo == class_id) & (all_pred_emo != class_id)
            incorrect_conf = all_confidences[incorrect_mask].mean() if incorrect_mask.any() else 0
        else:
            avg_conf = correct_conf = incorrect_conf = 0
        
        class_metrics[int(class_id)] = {
            'accuracy': acc,
            'avg_confidence': avg_conf,
            'correct_confidence': correct_conf,
            'incorrect_confidence': incorrect_conf
        }
        
        print(f"Class {int(class_id):2d}: Acc={acc:.4f} | AvgConf={avg_conf:.4f} | "
              f"CorrectConf={correct_conf:.4f} | IncorrectConf={incorrect_conf:.4f}")
    
    # Overall accuracy
    macro_acc = accuracy_score(all_true_emo, all_pred_emo)
    macro_f1 = f1_score(all_true_emo, all_pred_emo, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_true_emo, all_pred_emo, average='weighted', zero_division=0)
    
    print(f"\nOverall Accuracy:  {macro_acc:.4f}")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    
    # Hidden flag metrics
    print("\n" + "=" * 70)
    print("HIDDEN FLAG DETECTION")
    print("=" * 70)
    
    acc_hid = accuracy_score(all_true_hid, all_pred_hid)
    prec_hid, rec_hid, f1_hid, _ = precision_recall_fscore_support(
        all_true_hid,
        all_pred_hid,
        average="binary",
        pos_label=1,
        zero_division=0
    )
    
    print(f"Accuracy:  {acc_hid:.4f}")
    print(f"Precision: {prec_hid:.4f}")
    print(f"Recall:    {rec_hid:.4f}")
    print(f"F1-Score:  {f1_hid:.4f}")
    
    # AUC for hidden flag
    try:
        hid_auc = roc_auc_score(all_true_hid, all_hid_probs)
        print(f"AUC:       {hid_auc:.4f}")
    except:
        hid_auc = 0.0
    
    # Confusion matrix
    cm_hid = confusion_matrix(all_true_hid, all_pred_hid)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg    {cm_hid[0,0]:4d}  {cm_hid[0,1]:4d}")
    print(f"       Pos    {cm_hid[1,0]:4d}  {cm_hid[1,1]:4d}")
    
    if criterion is not None and num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.4f}")
    
    metrics = {
        "emo_accuracy": macro_acc,
        "emo_macro_f1": macro_f1,
        "emo_weighted_f1": weighted_f1,
        "hid_accuracy": acc_hid,
        "hid_precision": prec_hid,
        "hid_recall": rec_hid,
        "hid_f1": f1_hid,
        "hid_auc": hid_auc,
        "avg_confidence": np.mean(all_confidences),
        "class_metrics": class_metrics,
        "emo_probs": all_emo_probs,
        "predictions": all_pred_emo
    }
    
    if criterion is not None:
        metrics["loss"] = total_loss / max(num_batches, 1)
    
    return metrics

# ============================================================================
# IMPROVEMENT 5: Advanced Early Stopping with Plateau Detection
# ============================================================================

class AdvancedEarlyStopping:
    """Advanced early stopping with plateau detection and model checkpointing"""
    
    def __init__(self, patience: int = 5, monitor: str = "val_emo_accuracy", 
                 mode: str = "max", min_delta: float = 0.001, 
                 plateau_patience: int = 3):
        self.patience = patience
        self.plateau_patience = plateau_patience
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.plateau_counter = 0
        self.best_score = None
        self.best_model_state = None
        self.best_epoch = 0
        self.early_stop = False
        self.scores_history = []
    
    def __call__(self, score: float, model_state_dict: dict, epoch: int, lr: float = None) -> dict:
        """
        Returns: dict with 'stop' (bool) and 'reduce_lr' (bool)
        """
        self.scores_history.append(score)
        
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state_dict.items()}
            self.best_epoch = epoch
            return {'stop': False, 'reduce_lr': False}
        
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
            plateau = len(self.scores_history) > 3 and abs(score - self.scores_history[-2]) < self.min_delta
        else:
            improved = score < (self.best_score - self.min_delta)
            plateau = len(self.scores_history) > 3 and abs(score - self.scores_history[-2]) < self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = {k: v.cpu().clone() for k, v in model_state_dict.items()}
            self.best_epoch = epoch
            self.counter = 0
            self.plateau_counter = 0
            return {'stop': False, 'reduce_lr': False}
        else:
            self.counter += 1
            
            # Detect plateau
            if plateau:
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
            
            reduce_lr = self.plateau_counter >= self.plateau_patience
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                logger.info(f"Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch + 1}")
                return {'stop': True, 'reduce_lr': False}
            
            return {'stop': False, 'reduce_lr': reduce_lr}

# ============================================================================
# IMPROVEMENT 6: Label Smoothing with Confidence Penalty
# ============================================================================

class ConfidencePenaltyLoss(nn.Module):
    """Cross-entropy with label smoothing and confidence penalty"""
    
    def __init__(self, num_classes, smoothing=0.1, confidence_penalty=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence_penalty = confidence_penalty
    
    def forward(self, logits, targets):
        # Label smoothing
        smooth_targets = torch.zeros_like(logits).scatter_(
            1, targets.unsqueeze(1), 1.0 - self.smoothing
        )
        smooth_targets += self.smoothing / self.num_classes
        
        # Cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        
        # Confidence penalty (encourage diverse predictions)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        confidence_penalty = -entropy * self.confidence_penalty
        
        return ce_loss + confidence_penalty

# ============================================================================
# IMPROVEMENT 7: Focal Loss with Adaptive Gamma
# ============================================================================

class AdaptiveFocalLoss(nn.Module):
    """Focal loss with adaptive gamma based on class difficulty"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Adaptive gamma: increase for easy examples, decrease for hard ones
        gamma_t = self.gamma * (1 + pt)  # Higher gamma for easy examples
        
        focal_weight = (1 - pt) ** gamma_t
        
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

# ============================================================================
# IMPROVEMENT 8: Cosine Annealing with Warm Restarts
# ============================================================================

def get_scheduler_with_warm_restarts(optimizer, num_training_steps, num_warmup_steps=0.1, num_cycles=3):
    """Cosine annealing with warm restarts"""
    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_warmup_steps * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=num_cycles
    )

# ============================================================================
# Collate Function
# ============================================================================

def collate_fn(batch, tokenizer, max_length: int = 128):
    """
    Collate function used by DataLoader workers.
    Takes a batch of (text, emotion_id, hidden_id) and tokenizes it.
    """
    texts, emo_ids, hid_ids = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["emotion_labels"] = torch.tensor(emo_ids, dtype=torch.long)
    enc["hidden_labels"] = torch.tensor(hid_ids, dtype=torch.float)
    return enc

# ============================================================================
# MAIN TRAINING FUNCTION WITH ALL IMPROVEMENTS (FIXED dtype issues)
# ============================================================================

def train_model_advanced(config: dict, train_data: dict, val_data: dict, label_encoder, device: str):
    """
    Main training function with all accuracy improvements
    """
    train_cfg = config['training']
    model_cfg = config['model']
    tokenizer_cfg = config.get('tokenizer', {"max_length": 128})
    
    # Mixed precision training - DISABLED to avoid dtype issues
    use_amp = False
    scaler = None
    
    # Load model/tokenizer with cache
    model_path = get_pretrained_model_path(model_cfg['base_model_name'])
    tokenizer_path = get_pretrained_tokenizer_path(model_cfg['base_model_name'])
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        local_files_only=False
    )
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Add special tokens if needed
    special_tokens = ['[EMOJI=', '[CONFLICT_POS_EMOJI_NEG_TEXT]', '[CONFLICT_NEG_EMOJI_POS_TEXT]',
                      '[SMILE_EMOJI]', '[HEART_EMOJI]', '[CRY_EMOJI]', '[ANGRY_EMOJI]', '[LONG_TEXT]', ']']
    
    # Check if tokens exist, add if not
    new_tokens = []
    for token in special_tokens:
        if token not in tokenizer.vocab:
            new_tokens.append(token)
    
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} special tokens to tokenizer")
    
    # Create datasets with advanced augmentation
    train_ds = EmotionHiddenDataset(
        texts=train_data['texts'],
        emo_ids=train_data['emo_ids'],
        hid_ids=train_data['hid_ids'],
        emojis=train_data['emojis'],
        augment=True,
        minority_classes=[3, 4, 5]  # fear, love, surprise
    )
    
    val_ds = EmotionHiddenDataset(
        texts=val_data['texts'],
        emo_ids=val_data['emo_ids'],
        hid_ids=val_data['hid_ids'],
        emojis=val_data['emojis'],
        augment=False
    )
    
    # Calculate class weights with improved method
    class_weights = calculate_class_weights_advanced(
        train_data['emo_ids'],
        method="effective_num",
        beta=0.999
    )
    logger.info(f"Class weights: {class_weights}")
    
    # Create weighted sampler with replacement
    sample_weights = [class_weights.get(c, 1.0) for c in train_data['emo_ids']]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataloaders with improved collation
    _collate = partial(collate_fn, tokenizer=tokenizer, max_length=tokenizer_cfg['max_length'])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        sampler=sampler,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg['val_batch_size'],
        shuffle=False,
        collate_fn=_collate,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Initialize model
    model = EnhancedEmotionHiddenModel(
        base_model_name=model_cfg['base_model_name'],
        num_emotions=model_cfg['num_emotions'],
        dropout_p=model_cfg['dropout'],
        local_model_path=model_path,
        use_gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        hidden_size_factor=train_cfg.get('hidden_size_factor', 2)
    )
    
    # Resize token embeddings if we added new tokens
    if new_tokens:
        model.encoder.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings")
    
    # Move to device and ensure consistent dtype (float32)
    model = model.to(device)
    model = model.float()
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Selective layer freezing (keep more layers trainable)
    freeze_layers = model_cfg.get('freeze_layers', 1)
    for name, param in model.named_parameters():
        if "encoder.embeddings" in name or any(f"encoder.encoder.layer.{i}" in name for i in range(freeze_layers)):
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Initialize loss function with improvements
    loss_cfg = train_cfg['loss']
    criterion = EnhancedMultitaskLoss(
        class_weights_dict=class_weights,
        gamma=loss_cfg['focal_gamma'],
        hidden_weight=loss_cfg['hidden_weight'],
        pos_weight=loss_cfg['pos_weight'],
        label_smoothing=loss_cfg['label_smoothing'],
        device=str(device),
        use_uncertainty_weighting=True
    )
    
    # Setup optimizer with layer-wise learning rate decay
    optimizer = get_optimizer_with_llrd(
        model,
        base_lr=float(train_cfg['learning_rate_encoder']),
        decay_factor=0.95
    )
    
    # Setup scheduler with warm restarts
    num_training_steps = int(train_cfg['num_epochs'] * len(train_loader))
    scheduler = get_scheduler_with_warm_restarts(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0.1,
        num_cycles=3
    )
    
    # Stochastic Weight Averaging - disabled by default
    use_swa = False
    swa_model = None
    swa_scheduler = None
    swa_start = 0
    
    # Advanced early stopping
    early_stopping = AdvancedEarlyStopping(
        patience=train_cfg['early_stopping']['patience'],
        monitor=train_cfg['early_stopping']['monitor'],
        mode=train_cfg['early_stopping']['mode'],
        min_delta=0.0005,
        plateau_patience=2
    )
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRAINING WITH ADVANCED FEATURES")
    logger.info("=" * 70)
    
    best_val_acc = 0.0
    best_epoch = 0
    gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    
    for epoch in range(train_cfg['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{train_cfg['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Training phase
        model.train()
        train_loss = 0
        emo_correct = 0
        hid_correct = 0
        total_samples = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device with correct dtypes
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).float()
            emotion_labels = batch["emotion_labels"].to(device).long()
            hidden_labels = batch["hidden_labels"].to(device).float()
            
            # Forward pass
            emo_logits, hid_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss, l_emo, l_hid = criterion(
                emo_logits, emotion_labels,
                hid_logits, hidden_labels
            )
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Skip if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx + 1}, skipping")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Calculate batch accuracy
            emo_preds = emo_logits.argmax(dim=-1)
            hid_preds = (torch.sigmoid(hid_logits) > 0.5).long()
            
            emo_correct += (emo_preds == emotion_labels).sum().item()
            hid_correct += (hid_preds == hidden_labels.long()).sum().item()
            total_samples += len(emotion_labels)
            train_loss += loss.item() * gradient_accumulation_steps
            
            if (batch_idx + 1) % 50 == 0:
                if total_samples > 0:
                    logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                              f"Loss: {loss.item()*gradient_accumulation_steps:.4f} | "
                              f"Emo Acc: {emo_correct/total_samples:.4f} | "
                              f"Hid Acc: {hid_correct/total_samples:.4f}")
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_emo_acc = emo_correct / total_samples if total_samples > 0 else 0.0
        train_hid_acc = hid_correct / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"\nTraining Summary:")
        logger.info(f"  Avg Loss: {avg_train_loss:.4f}")
        logger.info(f"  Emotion Accuracy: {train_emo_acc:.4f}")
        logger.info(f"  Hidden Accuracy: {train_hid_acc:.4f}")
        logger.info(f"  Current LR: {optimizer.param_groups[-1]['lr']:.2e}")
        
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_emo_accuracy": train_emo_acc,
                "train_hid_accuracy": train_hid_acc,
                "learning_rate": optimizer.param_groups[-1]['lr']
            }, step=epoch + 1)
        
        # Validation phase with advanced metrics
        logger.info(f"\nValidation Results:")
        val_metrics = evaluate_model_advanced(model, val_loader, criterion, device, use_amp=False)
        
        # Log validation metrics
        if mlflow.active_run():
            mlflow.log_metrics({
                "val_emo_accuracy": val_metrics["emo_accuracy"],
                "val_emo_macro_f1": val_metrics["emo_macro_f1"],
                "val_hid_accuracy": val_metrics["hid_accuracy"],
                "val_hid_f1": val_metrics["hid_f1"],
                "val_hid_auc": val_metrics["hid_auc"],
                "val_loss": val_metrics.get("loss", 0.0)
            }, step=epoch + 1)
        
        # Early stopping check
        monitor_metric = val_metrics.get(train_cfg['early_stopping']['monitor'], 
                                        val_metrics['emo_accuracy'])
        
        early_stop_result = early_stopping(
            monitor_metric, 
            model.state_dict(), 
            epoch,
            optimizer.param_groups[-1]['lr'] if optimizer.param_groups else None
        )
        
        # Reduce learning rate on plateau
        if early_stop_result['reduce_lr']:
            logger.info(f"  ⚠️ Plateau detected, reducing learning rate")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
        
        if early_stop_result['stop']:
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
            logger.info(f"  ✓ New best model saved! (Acc: {best_val_acc:.4f})")
    
    # Load best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
        logger.info(f"\nLoaded best model from epoch {early_stopping.best_epoch + 1} with accuracy: {best_val_acc:.4f}")
    
    # Final evaluation
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION ON BEST MODEL")
    logger.info("=" * 70)
    final_metrics = evaluate_model_advanced(model, val_loader, criterion, device, use_amp=False)
    
    return model, final_metrics, tokenizer

# ============================================================================
# HELPER FUNCTION: Advanced Class Weight Calculation
# ============================================================================

def calculate_class_weights_advanced(emo_ids: list, method: str = "effective_num", 
                                     beta: float = 0.999, smooth: float = 1.0) -> dict:
    """
    Calculate class weights with advanced smoothing techniques
    """
    emo_counts = Counter(emo_ids)
    num_classes = len(emo_counts)
    
    # Add smoothing for missing classes
    for i in range(num_classes):
        if i not in emo_counts:
            emo_counts[i] = smooth
    
    if method == "effective_num":
        # Effective number of samples
        effective_num = 1.0 - np.power(beta, list(emo_counts.values()))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        
        # Apply square root scaling to prevent extreme weights
        weights = np.sqrt(weights) * 2
        
        class_weights = {i: float(w) for i, w in enumerate(weights)}
    
    elif method == "inverse_sqrt":
        # Inverse square root frequency
        total = sum(emo_counts.values())
        weights = [np.sqrt(total / (count)) for count in emo_counts.values()]
        weights = [w / sum(weights) * num_classes for w in weights]
        class_weights = {i: float(w) for i, w in enumerate(weights)}
    
    else:  # inverse_freq with smoothing
        total = sum(emo_counts.values())
        class_weights = {i: total / (num_classes * count + 1) for i, count in emo_counts.items()}
    
    return class_weights

# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def emotion_train_pipeline(data_path: str = None, dataset_name: str = None):
    """
    Main training pipeline entry point with all improvements
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
        
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
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
            "max_length": 128,
            "dataset_path": dataset_path,
            "dataset_name": dataset_name or "default",
            "focal_gamma": config['training']['loss']['focal_gamma'],
            "label_smoothing": config['training']['loss']['label_smoothing'],
            "hidden_weight": config['training']['loss']['hidden_weight'],
            "pos_weight": config['training']['loss']['pos_weight'],
            "scheduler_type": "cosine_with_restarts",
            "mixed_precision": False,
            "gradient_accumulation_steps": config['training'].get('gradient_accumulation_steps', 1),
            "layerwise_lr_decay": 0.95,
            "augmentation": "advanced",
            "weight_method": "effective_num_with_sqrt"
        })
        
        # Train model with advanced features
        model, metrics, tokenizer = train_model_advanced(config, train_data, val_data, label_encoder, device)
        
        # Log final metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key not in ['emo_probs', 'predictions', 'class_metrics']:
                mlflow.log_metric(f"final_{key}", value)
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'class_names': list(label_encoder.classes_),
            'tokenizer': tokenizer,
            'config': config,
            'final_metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }, "final_emotion_hidden_model_advanced.pt")
        logger.info("\nModel saved as 'final_emotion_hidden_model_advanced.pt'")
        
        # Log artifacts
        if config.get('mlflow', {}).get('log_artifacts', True):
            log_pytorch_model(model, artifact_path="model")
            log_label_encoder(label_encoder)
            log_training_config({
                "class_names": list(label_encoder.classes_),
                "model_config": config['model'],
                "training_config": config['training'],
                "final_metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "per_class_metrics": metrics.get('class_metrics', {})
            })
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    return model, metrics, tokenizer


# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion_advanced(text, emoji_char="", model=None, tokenizer=None, le=None, device="cuda"):
    """
    Advanced prediction function with confidence calibration
    """
    if model is None:
        # Load saved model
        checkpoint = torch.load("final_emotion_hidden_model_advanced.pt", map_location=device)
        model = EnhancedEmotionHiddenModel(
            "microsoft/deberta-v3-base", 
            len(checkpoint['class_names']),
            dropout_p=0.3
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        le = checkpoint['label_encoder']
        tokenizer = checkpoint['tokenizer']
    
    model.eval()
    
    # Preprocess using shared function
    proc_text = build_input(text, emoji_char)
    
    # Tokenize
    enc = tokenizer(
        proc_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    
    input_ids = enc["input_ids"].to(device).long()
    attention_mask = enc["attention_mask"].to(device).float()
    
    with torch.no_grad():
        emo_logits, hid_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply temperature scaling for better calibration
        emo_probs = torch.softmax(emo_logits / 1.5, dim=-1)[0]
        hid_prob = torch.sigmoid(hid_logits)[0].item()
        
        # Get predictions
        emo_id = torch.argmax(emo_probs).item()
        emo_label = le.inverse_transform([emo_id])[0]
        emo_confidence = emo_probs[emo_id].item()
        
        # Get top-3 predictions
        top_probs, top_indices = torch.topk(emo_probs, min(3, len(emo_probs)))
        top_emotions = le.inverse_transform(top_indices.cpu().numpy())
        top_confidences = top_probs.cpu().numpy()
        
        # Calculate entropy for uncertainty
        entropy = -(emo_probs * torch.log(emo_probs + 1e-9)).sum().item()
        normalized_entropy = entropy / np.log(len(le.classes_))
        
        # Calculate margin
        if len(emo_probs) > 1:
            sorted_probs = torch.sort(emo_probs, descending=True)[0]
            margin = (sorted_probs[0] - sorted_probs[1]).item()
        else:
            margin = 1.0
    
    result = {
        "predicted_emotion": emo_label,
        "emotion_confidence": emo_confidence,
        "hidden_probability": hid_prob,
        "is_hidden": hid_prob > 0.5,
        "uncertainty": normalized_entropy,
        "margin": margin,
        "top_predictions": [
            {"emotion": e, "confidence": float(c)}
            for e, c in zip(top_emotions, top_confidences)
        ],
        "processed_text": proc_text,
        "all_probabilities": {
            le.inverse_transform([i])[0]: float(emo_probs[i])
            for i in range(len(emo_probs))
        }
    }
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train advanced emotion detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        '--epochs', '--num-epochs',
        type=int,
        default=None,
        dest='num_epochs',
        help='Override number of epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size'
    )
    
    parser.add_argument(
        '--list-datasets', '--list_datasets',
        action='store_true',
        help='List available datasets from config.yaml and exit'
    )
    
    args = parser.parse_args()
    
    # Override config from command line
    if args.num_epochs:
        os.environ['OVERRIDE_EPOCHS'] = str(args.num_epochs)
    if args.batch_size:
        os.environ['OVERRIDE_BATCH_SIZE'] = str(args.batch_size)
    
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
"""
Training pipeline with transformer fine-tuning for multi-task emotion detection.
Tasks: binary hidden flag | 6-class hidden emotion
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
    AutoModel,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_fscore_support
)
import logging
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import load_config, get_data_paths
from utils.model_loader import get_pretrained_model_path, get_pretrained_tokenizer_path
from utils.mlflow_utils import init_mlflow, log_pytorch_model, log_label_encoder, log_training_config
from src.preprocessing import build_input
import mlflow
from pipelines.emotion_data_pipeline import emotion_data_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ADVANCED DATA AUGMENTATION
# ============================================================================

class AdvancedAugmentation:
    """Advanced text augmentation techniques for minority classes"""
    
    def __init__(self, tokenizer, p=0.3):
        self.tokenizer = tokenizer
        self.p = p
        
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
        words = text.split()
        if len(words) == 1:
            return text
        new_words = [word for word in words if random.random() > p]
        return ' '.join(new_words) if new_words else random.choice(words)
    
    def random_swap(self, text, n=2):
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(min(n, len(words))):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ' '.join(words)
    
    def synonym_replacement(self, text):
        words = text.split()
        new_words = []
        for word in words:
            if word.lower() in self.synonyms and random.random() < 0.3:
                new_words.append(random.choice(self.synonyms[word.lower()]))
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
    def augment(self, text, emotion_id, minority_classes):
        if emotion_id not in minority_classes or random.random() > self.p:
            return text
        
        aug_type = random.choice(['delete', 'swap', 'synonym', 'prefix', 'suffix'])
        
        if aug_type == 'delete':
            return self.random_deletion(text)
        elif aug_type == 'swap':
            return self.random_swap(text)
        elif aug_type == 'synonym':
            return self.synonym_replacement(text)
        elif aug_type == 'prefix':
            prefixes = ["I feel", "Honestly,", "To be honest,", "I think", "In my opinion,", "Personally,"]
            return f"{random.choice(prefixes)} {text}"
        else:
            suffixes = [" right now", " today", " honestly", " tbh", " tbh", " tbh"]
            return f"{text}{random.choice(suffixes)}"

# ============================================================================
# MIXUP AUGMENTATION
# ============================================================================

class MixUp:
    """MixUp augmentation for better generalization"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def mixup(self, x1, x2, y1, y2):
        """Apply mixup to two samples (2 tasks)"""
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_x = lam * x1 + (1 - lam) * x2
        mixed_y1 = lam * y1 + (1 - lam) * y2  # hidden flag
        mixed_y2 = lam * y1 + (1 - lam) * y2  # 6-class hidden
        
        return mixed_x, mixed_y1, mixed_y2, lam

# ============================================================================
# 1. MODEL UPGRADE - 2 TASKS WITH EMOJI EMBEDDING FUSION
# ============================================================================

class EnhancedEmotionHiddenModel(nn.Module):
    """
    Multi-task transformer model for:
    1. Binary hidden emotion flag detection
    2. 6-class hidden emotion classification
    """
    
    def __init__(self, base_model_name: str, num_hidden6: int = 6,
                 dropout_p: float = 0.3, local_model_path: str = None,
                 use_gradient_checkpointing: bool = False, hidden_size_factor: int = 2):
        super().__init__()
        
        load_path = local_model_path if local_model_path else base_model_name
        try:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=True)
        except OSError:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=False)
        
        if use_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        hidden_size = self.encoder.config.hidden_size
        expanded_size = hidden_size * hidden_size_factor
        
        logger.info(f"Loaded encoder: {load_path} (hidden_size={hidden_size})")
        
        # Emoji emotion embedding (6-class hidden emotion → embedding)
        self.emoji_emotion_embedding = nn.Embedding(num_hidden6, 16)
        
        # Attention-based pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
        # Shared projection with residual (now takes concatenated features)
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size + 16, expanded_size),
            nn.LayerNorm(expanded_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(expanded_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_p / 2)
        )
        
        # REMOVED: 27-class emotion head
        
        # Hidden flag head (binary)
        self.hidden_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_p / 2),
            nn.Linear(hidden_size // 2, 64),
            nn.GELU(),
            nn.Dropout(dropout_p / 4),
            nn.Linear(64, 1),
        )
        
        # 6-class hidden emotion head
        self.hidden6_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_p / 2),
            nn.Linear(hidden_size // 4, num_hidden6),
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.hidden_head, self.hidden6_head,
                       self.shared_projection, self.attention_pooling]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        nn.init.orthogonal_(self.emoji_emotion_embedding.weight)
        nn.init.constant_(self.temperature, 1.5)
    
    def forward(self, input_ids, attention_mask, emoji_emotion_ids=None):
        """
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            emoji_emotion_ids: 6-class emotion IDs for emoji [batch_size] (optional)
        Returns:
            hid_logits: Binary hidden flag logits [batch_size]
            hidden6_logits: 6-class hidden emotion logits [batch_size, 6]
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Advanced pooling: weighted average based on token importance
        attention_weights = self.attention_pooling(hidden_states).squeeze(-1)
        attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1).unsqueeze(-1)
        pooled = (hidden_states * attention_weights).sum(dim=1)
        
        # Fallback to mean pooling if attention pooling fails
        if torch.isnan(pooled).any():
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        pooled = torch.clamp(pooled, min=-10.0, max=10.0)
        
        # Emoji emotion embedding fusion (if available)
        if emoji_emotion_ids is not None:
            emoji_embed = self.emoji_emotion_embedding(emoji_emotion_ids)  # [batch_size, 16]
            # Concatenate with pooled features
            combined = torch.cat([pooled, emoji_embed], dim=-1)  # [batch_size, hidden_size + 16]
        else:
            # Pad with zeros if no emoji emotion ids
            combined = torch.cat([pooled, torch.zeros(pooled.size(0), 16).to(pooled.device)], dim=-1)
        
        # Shared features with residual
        shared_features = self.shared_projection(combined)
        shared_features = shared_features + pooled  # Residual on original pooled
        
        # Separate heads with temperature scaling
        hid_logits = self.hidden_head(shared_features).squeeze(-1)
        hidden6_logits = self.hidden6_head(shared_features)
        
        return hid_logits, hidden6_logits
    
    def freeze_encoder_layers(self, num_layers: int = 2):
        """Freeze first N layers of the encoder"""
        frozen_count = 0
        for name, param in self.encoder.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
                frozen_count += 1
            for i in range(num_layers):
                if f"encoder.layer.{i}" in name:
                    param.requires_grad = False
                    frozen_count += 1
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Frozen {frozen_count} parameter groups")
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# ============================================================================
# 2. LOSS UPGRADE - 2 TASKS WITH UNCERTAINTY WEIGHTING
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for multi-class tasks"""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits, targets):
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                torch.zeros_like(logits), logits)
        
        num_classes = logits.size(-1)
        
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits).scatter_(
                    1, targets.unsqueeze(1), 1.0
                )
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        if self.label_smoothing > 0:
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        ce_loss = torch.clamp(ce_loss, min=0.0, max=10.0)
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
        
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


class EnhancedMultitaskLoss(nn.Module):
    """
    Multi-task loss with uncertainty weighting for 2 tasks:
    1. Binary hidden flag detection
    2. 6-class hidden emotion classification
    """
    
    def __init__(self, gamma: float = 2.0, hidden_weight: float = 0.8,
                 pos_weight: float = 2.0, label_smoothing: float = 0.1,
                 device: str = "cuda", use_uncertainty_weighting: bool = True):
        super().__init__()
        
        # Task weights (can be used if uncertainty weighting is disabled)
        self.hidden_weight = hidden_weight
        
        # Binary hidden flag loss
        self.hid_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        
        # 6-class hidden emotion loss
        self.hidden6_loss = FocalLoss(
            alpha=None,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Learnable task uncertainty (log variance) for 2 tasks
        if use_uncertainty_weighting:
            self.log_hidden_var = nn.Parameter(torch.zeros(1, device=device))
            self.log_hidden6_var = nn.Parameter(torch.zeros(1, device=device))
    
    def forward(self, hid_logits, hid_targets, hidden6_logits, hidden6_targets):
        """
        Compute multi-task loss with optional uncertainty weighting.
        
        Args:
            hid_logits: Binary hidden flag logits [batch_size]
            hid_targets: Binary targets [batch_size]
            hidden6_logits: 6-class hidden emotion logits [batch_size, 6]
            hidden6_targets: 6-class targets [batch_size]
        """
        hid_targets_float = hid_targets.float() if hid_targets.dtype != torch.float32 else hid_targets
        l_hid = self.hid_loss(hid_logits, hid_targets_float)
        
        l_hidden6 = self.hidden6_loss(hidden6_logits, hidden6_targets)
        
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = sum(L_i * exp(-s_i) + s_i)
            precision_hid = torch.exp(-self.log_hidden_var)
            precision_hid6 = torch.exp(-self.log_hidden6_var)
            
            total_loss = (precision_hid * l_hid + self.log_hidden_var +
                         precision_hid6 * l_hidden6 + self.log_hidden6_var)
        else:
            total_loss = l_hid + self.hidden_weight * l_hidden6
        
        return total_loss, l_hid, l_hidden6

# ============================================================================
# 3. DATASET/COLLATE - 2 TASKS
# ============================================================================

class EmotionHiddenDataset(Dataset):
    """Dataset with binary hidden flag and 6-class hidden emotion"""
    
    def __init__(self, texts, hid_ids, hidden6_ids, emojis,
                 augment=False, minority_classes=None):
        self.texts = list(texts)
        self.hid_ids = list(hid_ids)           # binary flag
        self.hidden6_ids = list(hidden6_ids)   # 6-class hidden emotion
        self.emojis = list(emojis)
        self.augment = augment
        self.minority_classes = minority_classes or [3, 4, 5]
        
        logger.info(f"Created dataset with {len(self.texts)} samples")
        logger.info(f"6-class distribution: {dict(Counter(hidden6_ids))}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        raw_text = self.texts[idx]
        emoji_char = self.emojis[idx] if idx < len(self.emojis) else ""
        hidden6_id = self.hidden6_ids[idx]
        
        # Simple text augmentation for minority classes
        if self.augment and np.random.random() < 0.3:
            if hidden6_id in self.minority_classes:
                variations = [
                    f"I feel {raw_text}",
                    f"{raw_text} honestly",
                    f"To be honest, {raw_text}",
                    f"{raw_text} right now"
                ]
                raw_text = np.random.choice(variations)
        
        proc_text = build_input(raw_text, emoji_char)
        return proc_text, self.hid_ids[idx], hidden6_id


def collate_fn(batch, tokenizer, max_length: int = 128):
    """
    Collate function with 2 labels: hidden_flag, hidden6
    """
    texts, hid_ids, hidden6_ids = zip(*batch)
    
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    enc["hidden_labels"] = torch.tensor(hid_ids, dtype=torch.float)        # binary flag
    enc["hidden6_labels"] = torch.tensor(hidden6_ids, dtype=torch.long)    # 6-class
    
    return enc

# ============================================================================
# 4. EVALUATION - 2 TASKS
# ============================================================================

def evaluate_model_advanced(model, dataloader, criterion=None, device="cuda", use_amp=False):
    """
    Enhanced evaluation with 2-task metrics
    """
    model.eval()
    
    # Binary hidden flag metrics
    all_true_hid, all_pred_hid = [], []
    all_hid_probs = []
    
    # 6-class hidden emotion metrics
    all_true_hidden6, all_pred_hidden6 = [], []
    all_hidden6_probs = []
    
    all_confidences = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move to device with correct dtypes
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).float()
            hidden_labels = batch["hidden_labels"].to(device).float()
            hidden6_labels = batch["hidden6_labels"].to(device).long()
            
            # Forward pass - now returns 2 outputs
            hid_logits, hidden6_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            if criterion is not None:
                loss, l_hid, l_hid6 = criterion(
                    hid_logits, hidden_labels,
                    hidden6_logits, hidden6_labels
                )
                total_loss += loss.item()
                num_batches += 1
            
            # Binary flag predictions
            hid_probs = torch.sigmoid(hid_logits)
            hid_preds = (hid_probs > 0.5).long()
            
            # 6-class hidden emotion predictions
            hidden6_probs = F.softmax(hidden6_logits, dim=-1)
            hidden6_preds = hidden6_logits.argmax(dim=-1)
            
            # Confidence (max probability of 6-class)
            confidences = hidden6_probs.max(dim=-1)[0]
            
            # Store binary flag
            all_true_hid.extend(hidden_labels.cpu().numpy())
            all_pred_hid.extend(hid_preds.cpu().numpy())
            all_hid_probs.extend(hid_probs.cpu().numpy())
            
            # Store 6-class
            all_true_hidden6.extend(hidden6_labels.cpu().numpy())
            all_pred_hidden6.extend(hidden6_preds.cpu().numpy())
            all_hidden6_probs.extend(hidden6_probs.cpu().numpy())
            
            all_confidences.extend(confidences.cpu().numpy())
    
    # Convert to numpy arrays
    all_true_hid = np.array(all_true_hid)
    all_pred_hid = np.array(all_pred_hid)
    all_true_hidden6 = np.array(all_true_hidden6)
    all_pred_hidden6 = np.array(all_pred_hidden6)
    all_confidences = np.array(all_confidences)
    
    # =======================================================================
    # BINARY HIDDEN FLAG METRICS
    # =======================================================================
    print("\n" + "=" * 70)
    print("BINARY HIDDEN FLAG DETECTION")
    print("=" * 70)
    
    acc_hid = accuracy_score(all_true_hid, all_pred_hid)
    prec_hid, rec_hid, f1_hid, _ = precision_recall_fscore_support(
        all_true_hid, all_pred_hid,
        average="binary", pos_label=1, zero_division=0
    )
    
    print(f"Accuracy:  {acc_hid:.4f}")
    print(f"Precision: {prec_hid:.4f}")
    print(f"Recall:    {rec_hid:.4f}")
    print(f"F1-Score:  {f1_hid:.4f}")
    
    try:
        hid_auc = roc_auc_score(all_true_hid, all_hid_probs)
        print(f"AUC:       {hid_auc:.4f}")
    except:
        hid_auc = 0.0
    
    cm_hid = confusion_matrix(all_true_hid, all_pred_hid)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Neg    Pos")
    print(f"Actual Neg    {cm_hid[0,0]:4d}  {cm_hid[0,1]:4d}")
    print(f"       Pos    {cm_hid[1,0]:4d}  {cm_hid[1,1]:4d}")
    
    # =======================================================================
    # 6-CLASS HIDDEN EMOTION METRICS
    # =======================================================================
    print("\n" + "=" * 70)
    print("6-CLASS HIDDEN EMOTION CLASSIFICATION REPORT")
    print("=" * 70)
    
    # Define emotion names
    emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
    
    unique_h6 = np.unique(np.concatenate([all_true_hidden6, all_pred_hidden6]))
    h6_target_names = [emotion_names[i] if i < len(emotion_names) else f"class_{i}" for i in unique_h6]
    
    print(classification_report(
        all_true_hidden6, all_pred_hidden6,
        labels=unique_h6, target_names=h6_target_names,
        digits=4, zero_division=0
    ))
    
    # Per-class accuracy for 6-class
    print("\nPER-CLASS ACCURACY (6-class):")
    cm_h6 = confusion_matrix(all_true_hidden6, all_pred_hidden6, labels=unique_h6)
    for i, class_id in enumerate(unique_h6):
        total = cm_h6[i].sum()
        correct = cm_h6[i, i]
        acc = correct / total if total > 0 else 0
        emotion_name = emotion_names[class_id] if class_id < len(emotion_names) else f"class_{class_id}"
        print(f"{emotion_name:10s} (id={int(class_id)}): {acc:.4f} ({correct}/{total})")
    
    macro_acc_h6 = accuracy_score(all_true_hidden6, all_pred_hidden6)
    macro_f1_h6 = f1_score(all_true_hidden6, all_pred_hidden6, average='macro', zero_division=0)
    weighted_f1_h6 = f1_score(all_true_hidden6, all_pred_hidden6, average='weighted', zero_division=0)
    
    print(f"\nOverall Accuracy (6-class):  {macro_acc_h6:.4f}")
    print(f"Macro F1-Score (6-class):    {macro_f1_h6:.4f}")
    print(f"Weighted F1-Score (6-class): {weighted_f1_h6:.4f}")
    
    # Confusion matrix for 6-class
    print(f"\nConfusion Matrix (6-class):")
    print("Rows: True, Columns: Predicted")
    print("      " + " ".join([f"{emotion_names[i][:4]:4s}" for i in unique_h6]))
    for i in range(len(cm_h6)):
        row_str = f"{emotion_names[int(unique_h6[i])][:4]:4s} "
        for j in range(len(cm_h6[i])):
            row_str += f"{cm_h6[i,j]:4d} "
        print(row_str)
    
    if criterion is not None and num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.4f}")
    
    metrics = {
        # Binary flag metrics
        "hid_accuracy": acc_hid,
        "hid_precision": prec_hid,
        "hid_recall": rec_hid,
        "hid_f1": f1_hid,
        "hid_auc": hid_auc,
        
        # 6-class metrics
        "hidden6_accuracy": macro_acc_h6,
        "hidden6_macro_f1": macro_f1_h6,
        "hidden6_weighted_f1": weighted_f1_h6,
        
        "avg_confidence": np.mean(all_confidences),
        "hid_probs": all_hid_probs,
        "hidden6_predictions": all_pred_hidden6,
        "hidden6_probs": all_hidden6_probs
    }
    
    if criterion is not None:
        metrics["loss"] = total_loss / max(num_batches, 1)
    
    return metrics

# ============================================================================
# LAYER-WISE LEARNING RATE DECAY - UPDATED FOR 2 TASKS
# ============================================================================

def get_optimizer_with_llrd(model, base_lr=2e-5, decay_factor=0.95):
    """Layer-wise learning rate decay for 2-task model"""
    optimizer_grouped_parameters = []
    
    if hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
        layers = list(model.encoder.encoder.layer)
        
        # Embeddings layer
        embeddings_params = list(model.encoder.embeddings.parameters())
        if embeddings_params:
            optimizer_grouped_parameters.append({
                'params': embeddings_params,
                'lr': base_lr * (decay_factor ** 12),
                'name': 'embeddings'
            })
        
        # Encoder layers
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
        all_params = list(model.encoder.parameters())
        if all_params:
            optimizer_grouped_parameters.append({
                'params': all_params,
                'lr': base_lr,
                'name': 'encoder'
            })
    
    # Pooler
    if hasattr(model.encoder, 'pooler'):
        pooler_params = list(model.encoder.pooler.parameters())
        if pooler_params:
            optimizer_grouped_parameters.append({
                'params': pooler_params,
                'lr': base_lr,
                'name': 'pooler'
            })
    
    # Shared projection
    if hasattr(model, 'shared_projection'):
        shared_params = list(model.shared_projection.parameters())
        if shared_params:
            optimizer_grouped_parameters.append({
                'params': shared_params,
                'lr': base_lr,
                'name': 'shared_projection'
            })
    
    # REMOVED: 27-class emotion head
    
    # Binary flag head
    if hasattr(model, 'hidden_head'):
        hidden_params = list(model.hidden_head.parameters())
        if hidden_params:
            optimizer_grouped_parameters.append({
                'params': hidden_params,
                'lr': base_lr * 2,
                'name': 'hidden_head'
            })
    
    # 6-class head
    if hasattr(model, 'hidden6_head'):
        hidden6_params = list(model.hidden6_head.parameters())
        if hidden6_params:
            optimizer_grouped_parameters.append({
                'params': hidden6_params,
                'lr': base_lr * 2,
                'name': 'hidden6_head'
            })
    
    # Emoji embedding
    if hasattr(model, 'emoji_emotion_embedding'):
        optimizer_grouped_parameters.append({
            'params': [model.emoji_emotion_embedding.weight],
            'lr': base_lr,
            'name': 'emoji_embedding'
        })
    
    # Temperature
    if hasattr(model, 'temperature') and model.temperature is not None:
        optimizer_grouped_parameters.append({
            'params': [model.temperature],
            'lr': base_lr * 0.1,
            'name': 'temperature'
        })
    
    if not optimizer_grouped_parameters:
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
# COSINE ANNEALING WITH WARM RESTARTS
# ============================================================================

def get_scheduler_with_warm_restarts(optimizer, num_training_steps, num_warmup_steps=0.1, num_cycles=3):
    return get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_warmup_steps * num_training_steps),
        num_training_steps=num_training_steps,
        num_cycles=num_cycles
    )

# ============================================================================
# ADVANCED EARLY STOPPING
# ============================================================================

class AdvancedEarlyStopping:
    """Early stopping with plateau detection"""
    
    def __init__(self, patience: int = 5, monitor: str = "val_hidden6_accuracy",
                 mode: str = "max", min_delta: float = 0.001, plateau_patience: int = 3):
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
# CLASS WEIGHT CALCULATION (for 6-class if needed)
# ============================================================================

def calculate_class_weights_6class(hidden6_ids: list, method: str = "effective_num",
                                     beta: float = 0.999, smooth: float = 1.0) -> dict:
    emo_counts = Counter(hidden6_ids)
    num_classes = len(emo_counts)
    
    for i in range(num_classes):
        if i not in emo_counts:
            emo_counts[i] = smooth
    
    if method == "effective_num":
        effective_num = 1.0 - np.power(beta, list(emo_counts.values()))
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * num_classes
        weights = np.sqrt(weights) * 2
        class_weights = {i: float(w) for i, w in enumerate(weights)}
    else:
        total = sum(emo_counts.values())
        class_weights = {i: total / (num_classes * count + 1) for i, count in emo_counts.items()}
    
    return class_weights

# ============================================================================
# MAIN TRAINING FUNCTION - 2 TASKS
# ============================================================================

def train_model_advanced(config: dict, train_data: dict, val_data: dict,
                         label_encoder_6, device: str):
    """
    Main training function with 2-task learning
    """
    train_cfg = config['training']
    model_cfg = config['model']
    tokenizer_cfg = config.get('tokenizer', {"max_length": 128})
    
    use_amp = False
    
    # Load model/tokenizer
    model_path = get_pretrained_model_path(model_cfg['base_model_name'])
    tokenizer_path = get_pretrained_tokenizer_path(model_cfg['base_model_name'])
    
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        cache_dir=os.path.expanduser("~/.cache/huggingface"),
        local_files_only=False
    )
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Add special tokens
    special_tokens = ['[EMOJI=', '[CONFLICT_POS_EMOJI_NEG_TEXT]', '[CONFLICT_NEG_EMOJI_POS_TEXT]',
                      '[SMILE_EMOJI]', '[HEART_EMOJI]', '[CRY_EMOJI]', '[ANGRY_EMOJI]', '[LONG_TEXT]', ']']
    
    new_tokens = [t for t in special_tokens if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} special tokens")
    
    # Create datasets with 2 labels
    train_ds = EmotionHiddenDataset(
        texts=train_data['texts'],
        hid_ids=train_data['hid_ids'],           # binary flag
        hidden6_ids=train_data['hidden6_ids'],   # 6-class
        emojis=train_data['emojis'],
        augment=True,
        minority_classes=[3, 4, 5]
    )
    
    val_ds = EmotionHiddenDataset(
        texts=val_data['texts'],
        hid_ids=val_data['hid_ids'],             # binary flag
        hidden6_ids=val_data['hidden6_ids'],     # 6-class
        emojis=val_data['emojis'],
        augment=False
    )
    
    # Optional: Weighted sampler for 6-class imbalance
    h6_counts = Counter(train_data['hidden6_ids'])
    h6_weights = calculate_class_weights_6class(train_data['hidden6_ids'])
    sample_weights = [h6_weights.get(c, 1.0) for c in train_data['hidden6_ids']]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # DataLoaders
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
    
    # Initialize 2-task model
    model = EnhancedEmotionHiddenModel(
        base_model_name=model_cfg['base_model_name'],
        num_hidden6=6,
        dropout_p=model_cfg['dropout'],
        local_model_path=model_path,
        use_gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
        hidden_size_factor=train_cfg.get('hidden_size_factor', 2)
    )
    
    # Resize embeddings if new tokens added
    if new_tokens:
        model.encoder.resize_token_embeddings(len(tokenizer))
        logger.info("Resized token embeddings")
    
    model = model.to(device).float()
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Freeze layers
    freeze_layers = model_cfg.get('freeze_layers', 1)
    for name, param in model.named_parameters():
        if "encoder.embeddings" in name or any(f"encoder.encoder.layer.{i}" in name for i in range(freeze_layers)):
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    
    # Loss function with 2 tasks
    loss_cfg = train_cfg['loss']
    criterion = EnhancedMultitaskLoss(
        gamma=loss_cfg['focal_gamma'],
        hidden_weight=loss_cfg['hidden_weight'],
        pos_weight=loss_cfg['pos_weight'],
        label_smoothing=loss_cfg['label_smoothing'],
        device=str(device),
        use_uncertainty_weighting=True
    )
    
    # Optimizer with LLRD
    optimizer = get_optimizer_with_llrd(
        model,
        base_lr=float(train_cfg['learning_rate_encoder']),
        decay_factor=0.95
    )
    
    # Scheduler
    num_training_steps = int(train_cfg['num_epochs'] * len(train_loader))
    scheduler = get_scheduler_with_warm_restarts(
        optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=0.1,
        num_cycles=3
    )
    
    # Early stopping (monitor hidden6 accuracy)
    early_stopping = AdvancedEarlyStopping(
        patience=train_cfg['early_stopping']['patience'],
        monitor="val_hidden6_accuracy",
        mode="max",
        min_delta=0.0005,
        plateau_patience=2
    )
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("STARTING 2-TASK TRAINING (Hidden Flag + 6-class Hidden Emotion)")
    logger.info("=" * 70)
    
    best_val_acc = 0.0
    best_epoch = 0
    grad_accum = train_cfg.get('gradient_accumulation_steps', 1)
    
    for epoch in range(train_cfg['num_epochs']):
        logger.info(f"\n{'='*50}")
        logger.info(f"EPOCH {epoch + 1}/{train_cfg['num_epochs']}")
        logger.info(f"{'='*50}")
        
        # Training
        model.train()
        train_loss = 0
        hid_correct = h6_correct = 0
        total_samples = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(device).long()
            attention_mask = batch["attention_mask"].to(device).float()
            hidden_labels = batch["hidden_labels"].to(device).float()
            hidden6_labels = batch["hidden6_labels"].to(device).long()
            
            # Forward - returns 2 outputs
            hid_logits, hidden6_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss, l_hid, l_hid6 = criterion(
                hid_logits, hidden_labels,
                hidden6_logits, hidden6_labels
            )
            
            loss = loss / grad_accum
            
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN/Inf loss at batch {batch_idx + 1}")
                continue
            
            loss.backward()
            
            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Accuracy
            hid_preds = (torch.sigmoid(hid_logits) > 0.5).long()
            h6_preds = hidden6_logits.argmax(dim=-1)
            
            hid_correct += (hid_preds == hidden_labels.long()).sum().item()
            h6_correct += (h6_preds == hidden6_labels).sum().item()
            total_samples += len(hidden_labels)
            train_loss += loss.item() * grad_accum
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} | "
                          f"Loss: {loss.item()*grad_accum:.4f} | "
                          f"Hid: {hid_correct/total_samples:.3f} | "
                          f"H6: {h6_correct/total_samples:.3f}")
        
        # Epoch metrics
        avg_loss = train_loss / len(train_loader)
        train_hid_acc = hid_correct / total_samples
        train_h6_acc = h6_correct / total_samples
        
        logger.info(f"\nTraining Summary:")
        logger.info(f"  Loss: {avg_loss:.4f} | Hid: {train_hid_acc:.4f} | H6: {train_h6_acc:.4f}")
        
        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "train_hid_accuracy": train_hid_acc,
                "train_h6_accuracy": train_h6_acc,
            }, step=epoch + 1)
        
        # Validation
        logger.info(f"\nValidation Results:")
        val_metrics = evaluate_model_advanced(model, val_loader, criterion, device)
        
        if mlflow.active_run():
            mlflow.log_metrics({
                "val_hid_accuracy": val_metrics["hid_accuracy"],
                "val_hid_f1": val_metrics["hid_f1"],
                "val_hid_auc": val_metrics["hid_auc"],
                "val_hidden6_accuracy": val_metrics["hidden6_accuracy"],
                "val_hidden6_macro_f1": val_metrics["hidden6_macro_f1"],
            }, step=epoch + 1)
        
        # Early stopping (monitor hidden6 accuracy)
        monitor_metric = val_metrics["hidden6_accuracy"]
        result = early_stopping(monitor_metric, model.state_dict(), epoch)
        
        if result['reduce_lr']:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
        
        if result['stop']:
            model.load_state_dict(early_stopping.best_model_state)
            break
        
        # Save best model based on hidden6 accuracy
        if val_metrics['hidden6_accuracy'] > best_val_acc:
            best_val_acc = val_metrics['hidden6_accuracy']
            best_epoch = epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder_6': label_encoder_6,
                'class_names_6': list(label_encoder_6.classes_),
                'tokenizer': tokenizer,
                'config': config,
                'val_metrics': val_metrics,
            }, "best_2task_model.pt")
            logger.info(f"  ✓ New best model saved! (H6 Acc: {best_val_acc:.4f})")
    
    # Final evaluation
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    logger.info("\n" + "=" * 70)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 70)
    final_metrics = evaluate_model_advanced(model, val_loader, criterion, device)
    
    return model, final_metrics, tokenizer

# ============================================================================
# MAIN PIPELINE - UPDATED FOR 2 TASKS
# ============================================================================

def emotion_train_pipeline(data_path: str = None, dataset_name: str = None):
    """
    Main training pipeline for 2-task emotion detection
    """
    config = load_config()
    
    # Override from env
    if 'OVERRIDE_EPOCHS' in os.environ:
        config['training']['num_epochs'] = int(os.environ['OVERRIDE_EPOCHS'])
    
    data_paths = get_data_paths()
    
    # Dataset path
    if data_path:
        dataset_path = data_path
    elif dataset_name:
        datasets = data_paths.get('datasets', {})
        if dataset_name in datasets:
            dataset_path = datasets[dataset_name]
        else:
            raise ValueError(f"Dataset '{dataset_name}' not found")
    else:
        dataset_path = data_paths.get('raw_data', 'merged_full_dataset.csv')
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # MLflow
    if config.get('mlflow', {}).get('tracking_enabled', True):
        init_mlflow(
            tracking_uri=data_paths.get('mlflow_tracking_uri'),
            experiment_name=config['mlflow']['experiment_name']
        )
    
    # Load data - now returns 4 values, ignore the 27-class encoder
    train_data, val_data, _, label_encoder_6 = emotion_data_pipeline(data_path=dataset_path)
    
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config['model']['base_model_name'],
            "num_epochs": config['training']['num_epochs'],
            "batch_size": config['training']['batch_size'],
            "lr_encoder": config['training']['learning_rate_encoder'],
            "max_length": 128,
            "focal_gamma": config['training']['loss']['focal_gamma'],
            "label_smoothing": config['training']['loss']['label_smoothing'],
            "hidden_weight": config['training']['loss']['hidden_weight'],
            "scheduler_type": "cosine_with_restarts",
            "mixed_precision": False,
            "layerwise_lr_decay": 0.95,
            "augmentation": "advanced",
            "tasks": "binary_flag + 6class_hidden_emotion"
        })
        
        # Train 2-task model
        model, metrics, tokenizer = train_model_advanced(
            config, train_data, val_data,
            label_encoder_6, device
        )
        
        # Log final metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and k not in ['hid_probs', 'hidden6_predictions', 'hidden6_probs']:
                mlflow.log_metric(f"final_{k}", v)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder_6': label_encoder_6,
            'class_names_6': list(label_encoder_6.classes_),
            'tokenizer': tokenizer,
            'config': config,
            'final_metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        }, "final_2task_model.pt")
        logger.info("\nModel saved as 'final_2task_model.pt'")
        
        # Log artifacts
        log_pytorch_model(model, artifact_path="model")
        log_label_encoder(label_encoder_6, "label_encoder_6")
    
    return model, metrics, tokenizer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--list-datasets', action='store_true')
    args = parser.parse_args()
    
    if args.epochs:
        os.environ['OVERRIDE_EPOCHS'] = str(args.epochs)
    
    if args.list_datasets:
        config = load_config()
        data_paths = get_data_paths()
        datasets = data_paths.get('datasets', {})
        print("\nAvailable datasets:")
        for name, path in datasets.items():
            print(f"  {name:20s} -> {path}")
        exit(0)
    
    emotion_train_pipeline(data_path=args.data_path, dataset_name=args.dataset)
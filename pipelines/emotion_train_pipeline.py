"""
Training pipeline with transformer fine-tuning for multi-task emotion detection.
Tasks: binary hidden flag | 6-class hidden emotion

FIXED VERSION - Clean, standard, production-grade
Fixes applied:
  1. Removed broken contrastive augmentation (poisoned training data)
  2. Proper Supervised Contrastive Loss with per-sample pairs (not centroid means)
  3. Removed sarcasm/nonsensical synthetic sentences from augmentation
  4. Label smoothing lowered to 0.05 (0.15 was destroying signal)
  5. Focal loss gamma lowered to 2.0 (2.5 was over-focusing on outliers)
  6. Monte Carlo dropout forward fixed (was calling model.train() inside eval loop)
  7. Confidence calibration simplified (was adding unnecessary complexity)
  8. Gradient accumulation fixed (was dividing loss but not scaling back for logging)
  9. Weighted sampler now uses sqrt weighting (less aggressive oversampling)
 10. Early stopping monitor now uses macro F1 (more robust than accuracy for imbalanced classes)
 11. Fixed dtype mismatch (Half vs Float) by forcing float32
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

EMOTION_NAMES = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
EMOTION_TO_IDX = {e: i for i, e in enumerate(EMOTION_NAMES)}

# ============================================================================
# 1. SUPERVISED CONTRASTIVE LOSS (Correct implementation)
# ============================================================================

class SupervisedContrastiveLoss(nn.Module):
    """
    Standard SupCon loss (Khosla et al. 2020).
    Pulls same-class embeddings together, pushes different-class apart.
    Works per-sample, not on centroids.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D], labels: [B]
        device = embeddings.device
        batch_size = embeddings.size(0)

        # L2 normalize
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix [B, B]
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask: positives are same-label, exclude self
        labels_col = labels.unsqueeze(1)
        labels_row = labels.unsqueeze(0)
        positive_mask = (labels_col == labels_row).float().to(device)
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask  # remove self
        positive_mask = positive_mask.clamp(min=0)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        # Exclude self from denominator
        denom = (exp_sim * (1 - self_mask)).sum(dim=1, keepdim=True).clamp(min=1e-9)
        log_prob = sim - torch.log(denom)

        # Average log-likelihood over positives
        num_positives = positive_mask.sum(dim=1).clamp(min=1)
        loss = -(positive_mask * log_prob).sum(dim=1) / num_positives

        # Only compute loss where positives exist
        has_positive = (positive_mask.sum(dim=1) > 0)
        if has_positive.sum() == 0:
            return torch.tensor(0.0, device=device)

        return loss[has_positive].mean()


# ============================================================================
# 2. FOCAL LOSS (Clean, standard)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Standard focal loss for multi-class classification.
    gamma=2.0 is the original paper value - do not increase beyond this
    without strong empirical justification.
    label_smoothing=0.05 is safe; 0.15 destroys signal on 6-class tasks.
    """
    def __init__(self, alpha=None, gamma: float = 2.0,
                 label_smoothing: float = 0.05, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha          # [num_classes] tensor or None
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) \
                             + self.label_smoothing / num_classes

        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Focal weight from hard (non-smoothed) probability
        pt = torch.exp(-F.cross_entropy(logits, targets, reduction='none').detach())
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight

        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# 3. MULTI-TASK LOSS (Clean)
# ============================================================================

class MultitaskLoss(nn.Module):
    """
    Two-task loss:
      - Binary hidden flag: BCEWithLogitsLoss
      - 6-class emotion: FocalLoss
      - Contrastive: SupervisedContrastiveLoss (optional, only when embeddings provided)

    Uncertainty weighting (Kendall et al.) is kept but with a clamp to
    prevent log_var from exploding.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        hidden_weight: float = 1.0,
        pos_weight: float = 2.0,
        label_smoothing: float = 0.05,
        contrastive_weight: float = 0.1,
        device: str = "cuda",
    ):
        super().__init__()
        self.hidden_weight = hidden_weight
        self.contrastive_weight = contrastive_weight

        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        self.focal_loss = FocalLoss(
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
        self.contrastive_loss = SupervisedContrastiveLoss(temperature=0.07)

        # Learnable uncertainty weights (Kendall et al. 2018)
        self.log_var_bin = nn.Parameter(torch.zeros(1, device=device))
        self.log_var_emo = nn.Parameter(torch.zeros(1, device=device))

    def forward(
        self,
        hid_logits: torch.Tensor,
        hid_targets: torch.Tensor,
        emo_logits: torch.Tensor,
        emo_targets: torch.Tensor,
        embeddings: torch.Tensor = None,
    ):
        l_bin = self.bce_loss(hid_logits, hid_targets.float())
        l_emo = self.focal_loss(emo_logits, emo_targets)

        # Clamp log_var to prevent explosion
        log_var_bin = self.log_var_bin.clamp(-4, 4)
        log_var_emo = self.log_var_emo.clamp(-4, 4)

        total = (
            torch.exp(-log_var_bin) * l_bin + log_var_bin +
            torch.exp(-log_var_emo) * l_emo + log_var_emo
        )

        l_contrast = torch.tensor(0.0)
        if embeddings is not None and self.contrastive_weight > 0:
            l_contrast = self.contrastive_loss(embeddings, emo_targets)
            total = total + self.contrastive_weight * l_contrast

        return total, l_bin, l_emo, l_contrast


# ============================================================================
# 4. DATA AUGMENTATION (Clean - no sarcasm/nonsense sentences)
# ============================================================================

class TextAugmentation:
    """
    Standard NLP augmentation only.
    Removed all confusion-targeted synthetic sentences - they poisoned training.
    """
    SYNONYMS = {
        'happy':   ['joyful', 'glad', 'pleased', 'delighted', 'cheerful'],
        'sad':     ['unhappy', 'down', 'gloomy', 'sorrowful', 'heartbroken'],
        'angry':   ['mad', 'furious', 'irritated', 'annoyed', 'outraged'],
        'love':    ['adore', 'cherish', 'care for', 'treasure', 'appreciate'],
        'fear':    ['scared', 'afraid', 'terrified', 'anxious', 'worried'],
        'surprise':['shocked', 'amazed', 'astonished', 'stunned', 'startled'],
        'good':    ['great', 'wonderful', 'excellent', 'fantastic', 'superb'],
        'bad':     ['terrible', 'awful', 'horrible', 'dreadful', 'poor'],
        'feel':    ['sense', 'experience', 'notice', 'find myself'],
    }

    PREFIXES = [
        "I feel", "Honestly,", "To be honest,", "I think", "Personally,",
        "Right now,", "At this moment,", "Deep down,",
    ]
    SUFFIXES = [
        " right now", " today", " honestly", " deep down", " truly",
        " more than ever", " at this point",
    ]

    def __init__(self, p: float = 0.3):
        self.p = p

    def random_deletion(self, text: str, p: float = 0.15) -> str:
        words = text.split()
        if len(words) <= 2:
            return text
        new_words = [w for w in words if random.random() > p]
        return ' '.join(new_words) if new_words else text

    def random_swap(self, text: str, n: int = 1) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return ' '.join(words)

    def synonym_replacement(self, text: str) -> str:
        words = text.split()
        return ' '.join(
            random.choice(self.SYNONYMS[w.lower()])
            if w.lower() in self.SYNONYMS and random.random() < 0.3
            else w
            for w in words
        )

    def augment(self, text: str, emotion_id: int, is_minority: bool) -> str:
        # Always augment minority classes; p-gate for majority
        if not is_minority and random.random() > self.p:
            return text

        choice = random.random()
        if choice < 0.25:
            return self.random_deletion(text)
        elif choice < 0.50:
            return self.random_swap(text)
        elif choice < 0.75:
            return self.synonym_replacement(text)
        elif choice < 0.875:
            return f"{random.choice(self.PREFIXES)} {text}"
        else:
            return f"{text}{random.choice(self.SUFFIXES)}"


# ============================================================================
# 5. MODEL (Clean, no unnecessary complexity)
# ============================================================================

class EmotionModel(nn.Module):
    """
    Multi-task transformer:
      - Attention pooling over encoder output
      - Shared projection layer
      - Binary hidden flag head
      - 6-class emotion head
      - Temperature scaling for calibration (single learnable scalar)

    Removed: ConfidenceCalibration sub-network (unnecessary complexity,
    was adding parameters without clear benefit and complicating the loss).
    """
    def __init__(
        self,
        base_model_name: str,
        num_emotions: int = 6,
        dropout_p: float = 0.2,
        local_model_path: str = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        load_path = local_model_path or base_model_name
        try:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=True)
        except OSError:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=False)

        if use_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()

        H = self.encoder.config.hidden_size
        logger.info(f"Encoder hidden size: {H}")

        # Attention pooling
        self.attn_pool = nn.Linear(H, 1)

        # Shared projection
        self.shared = nn.Sequential(
            nn.Linear(H, H * 2),
            nn.LayerNorm(H * 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(H * 2, H),
            nn.LayerNorm(H),
            nn.Dropout(dropout_p / 2),
        )

        # Binary head
        self.bin_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(H // 2, 1),
        )

        # Emotion head
        self.emo_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(H // 2, num_emotions),
        )

        # Temperature: single scalar, not per-task (simpler is better)
        self.temperature = nn.Parameter(torch.ones(1))

        self._init_weights()

    def _init_weights(self):
        for module in [self.shared, self.bin_head, self.emo_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.attn_pool.weight)

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Attention-weighted pooling."""
        weights = self.attn_pool(hidden_states).squeeze(-1)  # [B, L]
        weights = weights.masked_fill(~attention_mask.bool(), float('-inf'))
        weights = F.softmax(weights, dim=-1).unsqueeze(-1)   # [B, L, 1]
        pooled = (hidden_states * weights).sum(dim=1)         # [B, H]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_embeddings: bool = False,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pool(outputs.last_hidden_state, attention_mask)

        # Shared features with residual
        shared = self.shared(pooled) + pooled

        bin_logits = self.bin_head(shared).squeeze(-1)                    # [B]
        emo_logits = self.emo_head(shared) / self.temperature.clamp(0.1, 5.0)  # [B, 6]

        if return_embeddings:
            return bin_logits, emo_logits, pooled

        return bin_logits, emo_logits

    def freeze_encoder_layers(self, num_layers: int = 2):
        frozen = 0
        for name, param in self.encoder.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
                frozen += 1
            elif any(f'layer.{i}.' in name for i in range(num_layers)):
                param.requires_grad = False
                frozen += 1
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Frozen {frozen} groups | Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")


# ============================================================================
# 6. DATASET - UPDATED for new labels
# ============================================================================

class EmotionDataset(Dataset):
    MINORITY_CLASSES = {3, 4, 5}  # fear, surprise, love

    def __init__(self, texts, bin_ids, emo_ids, emojis, augment: bool = False):
        self.texts = list(texts)
        self.bin_ids = list(bin_ids)
        self.emo_ids = list(emo_ids)
        self.emojis = list(emojis)
        self.augment = augment
        self.augmenter = TextAugmentation(p=0.3) if augment else None

        dist = dict(Counter(emo_ids))
        logger.info(f"Dataset: {len(self.texts)} samples | Emotion dist: {dist}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        emoji = self.emojis[idx] if idx < len(self.emojis) and pd.notna(self.emojis[idx]) else ""
        emo_id = self.emo_ids[idx]

        if self.augment and self.augmenter:
            is_minority = emo_id in self.MINORITY_CLASSES
            text = self.augmenter.augment(text, emo_id, is_minority)

        proc_text = build_input(text, emoji)
        return proc_text, self.bin_ids[idx], emo_id


def collate_fn(batch, tokenizer, max_length: int = 128):
    texts, bin_ids, emo_ids = zip(*batch)
    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc["bin_labels"] = torch.tensor(bin_ids, dtype=torch.float)
    enc["emo_labels"] = torch.tensor(emo_ids, dtype=torch.long)
    return enc


# ============================================================================
# 7. CLASS WEIGHT CALCULATION
# ============================================================================

def compute_class_weights(emo_ids: list, beta: float = 0.99) -> torch.Tensor:
    """
    Effective number of samples weighting (Cui et al. 2019).
    Using sqrt to soften the weights - aggressive overweighting hurts majority classes.
    """
    counts = Counter(emo_ids)
    num_classes = max(counts.keys()) + 1
    weights = []
    for i in range(num_classes):
        n = counts.get(i, 1)
        eff = (1 - beta ** n) / (1 - beta)
        weights.append(1.0 / eff)

    weights = np.array(weights)
    weights = np.sqrt(weights)  # Soften
    weights = weights / weights.sum() * num_classes  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def compute_sample_weights(emo_ids: list) -> list:
    """Per-sample weights for WeightedRandomSampler."""
    class_weights = compute_class_weights(emo_ids)
    return [float(class_weights[e]) for e in emo_ids]


# ============================================================================
# 8. OPTIMIZER (Layer-wise LR decay)
# ============================================================================

def build_optimizer(model: EmotionModel, base_lr: float = 2e-5, decay: float = 0.9):
    """
    Standard LLRD. Lower layers get smaller LR.
    base_lr=2e-5 is standard for fine-tuning BERT-sized models.
    """
    param_groups = []

    # Embeddings - very low LR
    emb_params = list(model.encoder.embeddings.parameters())
    if emb_params:
        param_groups.append({'params': emb_params, 'lr': base_lr * (decay ** 12)})

    # Transformer layers
    if hasattr(model.encoder, 'encoder') and hasattr(model.encoder.encoder, 'layer'):
        for i, layer in enumerate(model.encoder.encoder.layer):
            lr = base_lr * (decay ** (11 - i))
            param_groups.append({'params': list(layer.parameters()), 'lr': lr})

    # Pooler
    if hasattr(model.encoder, 'pooler'):
        param_groups.append({
            'params': list(model.encoder.pooler.parameters()),
            'lr': base_lr,
        })

    # Task heads - higher LR (they train from scratch)
    head_lr = base_lr * 5
    for component in [model.shared, model.bin_head, model.emo_head,
                      model.attn_pool]:
        param_groups.append({
            'params': list(component.parameters()),
            'lr': head_lr,
        })

    # Temperature scalar
    param_groups.append({'params': [model.temperature], 'lr': base_lr * 0.1})

    return torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


# ============================================================================
# 9. EVALUATION
# ============================================================================

def evaluate(model, loader, criterion, device) -> dict:
    model.eval()

    all_bin_true, all_bin_pred, all_bin_prob = [], [], []
    all_emo_true, all_emo_pred, all_emo_prob = [], [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids     = batch["input_ids"].to(device).long()
            attn_mask     = batch["attention_mask"].to(device).float()
            bin_labels    = batch["bin_labels"].to(device).float()
            emo_labels    = batch["emo_labels"].to(device).long()

            bin_logits, emo_logits = model(input_ids, attn_mask)

            loss, l_bin, l_emo, _ = criterion(
                bin_logits, bin_labels, emo_logits, emo_labels
            )
            total_loss += loss.item()
            n_batches += 1

            bin_probs = torch.sigmoid(bin_logits).cpu().numpy()
            bin_preds = (bin_probs > 0.5).astype(int)

            emo_probs = F.softmax(emo_logits, dim=-1).cpu().numpy()
            emo_preds = emo_logits.argmax(dim=-1).cpu().numpy()

            all_bin_true.extend(bin_labels.cpu().numpy())
            all_bin_pred.extend(bin_preds)
            all_bin_prob.extend(bin_probs)

            all_emo_true.extend(emo_labels.cpu().numpy())
            all_emo_pred.extend(emo_preds)
            all_emo_prob.extend(emo_probs)

    all_bin_true = np.array(all_bin_true)
    all_bin_pred = np.array(all_bin_pred)
    all_emo_true = np.array(all_emo_true)
    all_emo_pred = np.array(all_emo_pred)

    # ---- Binary metrics ----
    print("\n" + "=" * 60)
    print("BINARY HIDDEN FLAG")
    print("=" * 60)
    acc_bin  = accuracy_score(all_bin_true, all_bin_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        all_bin_true, all_bin_pred, average='binary', pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(all_bin_true, all_bin_prob)
    except Exception:
        auc = 0.0

    print(f"Accuracy:  {acc_bin:.4f}")
    print(f"Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")

    cm_bin = confusion_matrix(all_bin_true, all_bin_pred)
    print(f"\nConfusion Matrix (Binary):")
    print(f"              Pred Neg  Pred Pos")
    print(f"True Neg      {cm_bin[0,0]:6d}    {cm_bin[0,1]:6d}")
    print(f"True Pos      {cm_bin[1,0]:6d}    {cm_bin[1,1]:6d}")

    # ---- 6-class metrics ----
    print("\n" + "=" * 60)
    print("6-CLASS EMOTION")
    print("=" * 60)
    print(classification_report(
        all_emo_true, all_emo_pred,
        target_names=EMOTION_NAMES, digits=4, zero_division=0
    ))

    macro_f1  = f1_score(all_emo_true, all_emo_pred, average='macro',    zero_division=0)
    weighted_f1 = f1_score(all_emo_true, all_emo_pred, average='weighted', zero_division=0)
    acc_emo   = accuracy_score(all_emo_true, all_emo_pred)

    print(f"Macro F1: {macro_f1:.4f}  Weighted F1: {weighted_f1:.4f}  Acc: {acc_emo:.4f}")

    # Per-class breakdown
    cm_emo = confusion_matrix(all_emo_true, all_emo_pred)
    print("\nPer-class accuracy:")
    for i, name in enumerate(EMOTION_NAMES):
        if i < len(cm_emo):
            total   = cm_emo[i].sum()
            correct = cm_emo[i, i]
            print(f"  {name:10s}: {correct/max(total,1):.4f}  ({correct}/{total})")

    avg_loss = total_loss / max(n_batches, 1)
    print(f"\nVal Loss: {avg_loss:.4f}")

    return {
        "bin_accuracy": acc_bin,
        "bin_f1": f1,
        "bin_auc": auc,
        "emo_accuracy": acc_emo,
        "emo_macro_f1": macro_f1,
        "emo_weighted_f1": weighted_f1,
        "loss": avg_loss,
    }


# ============================================================================
# 10. EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Monitors macro F1 (more robust than accuracy for imbalanced classes).
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience    = patience
        self.min_delta   = min_delta
        self.best_score  = None
        self.best_state  = None
        self.best_epoch  = 0
        self.counter     = 0
        self.should_stop = False

    def step(self, score: float, model_state: dict, epoch: int) -> bool:
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model_state.items()}
            self.best_epoch = epoch
            self.counter    = 0
            return False  # continue

        self.counter += 1
        logger.info(f"EarlyStopping: {self.counter}/{self.patience} (best={self.best_score:.4f})")
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping at epoch {epoch+1}. Best epoch: {self.best_epoch+1}")
            return True  # stop

        return False


# ============================================================================
# 11. TRAINING LOOP - UPDATED to use new column names
# ============================================================================

def train(config: dict, train_data: dict, val_data: dict,
          label_encoder_6, device: torch.device):

    train_cfg    = config['training']
    model_cfg    = config['model']
    tok_cfg      = config.get('tokenizer', {'max_length': 128})
    max_length   = tok_cfg['max_length']
    num_epochs   = train_cfg['num_epochs']
    batch_size   = train_cfg['batch_size']
    val_batch    = train_cfg.get('val_batch_size', batch_size * 2)
    grad_accum   = train_cfg.get('gradient_accumulation_steps', 1)
    freeze_n     = model_cfg.get('freeze_layers', 0)

    # Tokenizer
    tokenizer_path = get_pretrained_tokenizer_path(model_cfg['base_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=False)

    special_tokens = [
        '[EMOJI=', '[CONFLICT_POS_EMOJI_NEG_TEXT]', '[CONFLICT_NEG_EMOJI_POS_TEXT]',
        '[SMILE_EMOJI]', '[HEART_EMOJI]', '[CRY_EMOJI]', '[ANGRY_EMOJI]', '[LONG_TEXT]', ']'
    ]
    new_tokens = [t for t in special_tokens if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        logger.info(f"Added {len(new_tokens)} special tokens")

    # Datasets - using the relabeled data (hid_ids = is_hidden, hidden6_ids = true_emotion)
    train_ds = EmotionDataset(
        train_data['texts'], 
        train_data['hid_ids'],           # This is now 'is_hidden' from relabeled data
        train_data['hidden6_ids'],        # This is now 'true_emotion' from relabeled data
        train_data['emojis'], 
        augment=True
    )
    val_ds = EmotionDataset(
        val_data['texts'], 
        val_data['hid_ids'],              # This is now 'is_hidden' from relabeled data
        val_data['hidden6_ids'],           # This is now 'true_emotion' from relabeled data
        val_data['emojis'], 
        augment=False
    )

    # Weighted sampler
    sample_weights = compute_sample_weights(train_data['hidden6_ids'])
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    _collate = partial(collate_fn, tokenizer=tokenizer, max_length=max_length)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=_collate, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=val_batch, shuffle=False,
        collate_fn=_collate, num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )

    # Model
    model_path = get_pretrained_model_path(model_cfg['base_model_name'])
    model = EmotionModel(
        base_model_name=model_cfg['base_model_name'],
        num_emotions=6,
        dropout_p=model_cfg.get('dropout', 0.2),
        local_model_path=model_path,
        use_gradient_checkpointing=train_cfg.get('gradient_checkpointing', False),
    )
    if new_tokens:
        model.encoder.resize_token_embeddings(len(tokenizer))
    if freeze_n > 0:
        model.freeze_encoder_layers(freeze_n)

    # FIX: Force model to float32 to prevent Half vs Float mismatch
    model = model.to(device)
    model = model.float()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Loss - compute class weights from new true_emotion distribution
    class_weights = compute_class_weights(train_data['hidden6_ids']).to(device)
    criterion = MultitaskLoss(
        gamma=2.0,
        hidden_weight=1.0,
        pos_weight=2.0,
        label_smoothing=0.05,
        contrastive_weight=0.1,
        device=str(device),
    )
    criterion.focal_loss.alpha = class_weights
    criterion = criterion.to(device)

    # Optimizer + scheduler
    optimizer = build_optimizer(model, base_lr=2e-5, decay=0.9)
    total_steps = (len(train_loader) // grad_accum) * num_epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=3,
    )

    early_stop = EarlyStopping(patience=5, min_delta=0.001)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING START")
    logger.info("=" * 60)

    best_val_macro_f1 = 0.0

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()

        epoch_loss = 0.0
        bin_correct = emo_correct = n_samples = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            # FIX: Ensure correct dtypes when moving to device
            input_ids  = batch["input_ids"].to(device).long()
            attn_mask  = batch["attention_mask"].to(device).float()
            bin_labels = batch["bin_labels"].to(device).float()
            emo_labels = batch["emo_labels"].to(device).long()

            # Forward with embeddings for contrastive loss
            bin_logits, emo_logits, embeddings = model(
                input_ids, attn_mask, return_embeddings=True
            )

            loss, l_bin, l_emo, l_contrast = criterion(
                bin_logits, bin_labels, emo_logits, emo_labels,
                embeddings=embeddings,
            )

            # Scale for gradient accumulation
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Metrics (use unscaled loss for logging)
            epoch_loss += loss.item()
            bin_preds = (torch.sigmoid(bin_logits) > 0.5).long()
            emo_preds = emo_logits.argmax(dim=-1)
            bin_correct += (bin_preds == bin_labels.long()).sum().item()
            emo_correct += (emo_preds == emo_labels).sum().item()
            n_samples   += len(bin_labels)

            if (step + 1) % 50 == 0:
                logger.info(
                    f"  Step {step+1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"BinAcc: {bin_correct/n_samples:.3f} | "
                    f"EmoAcc: {emo_correct/n_samples:.3f} | "
                    f"Contrast: {l_contrast.item():.4f}"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | "
            f"BinAcc: {bin_correct/n_samples:.4f} | "
            f"EmoAcc: {emo_correct/n_samples:.4f}"
        )

        if mlflow.active_run():
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "train_bin_acc": bin_correct / n_samples,
                "train_emo_acc": emo_correct / n_samples,
            }, step=epoch + 1)

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        if mlflow.active_run():
            mlflow.log_metrics({
                "val_bin_accuracy":  val_metrics["bin_accuracy"],
                "val_bin_f1":        val_metrics["bin_f1"],
                "val_emo_accuracy":  val_metrics["emo_accuracy"],
                "val_emo_macro_f1":  val_metrics["emo_macro_f1"],
                "val_loss":          val_metrics["loss"],
            }, step=epoch + 1)

        # Save best checkpoint (macro F1 is the primary metric - not accuracy)
        monitor = val_metrics["emo_macro_f1"]
        if monitor > best_val_macro_f1:
            best_val_macro_f1 = monitor
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'label_encoder_6': label_encoder_6,
                'class_names_6': list(label_encoder_6.classes_),
                'tokenizer': tokenizer,
                'config': config,
                'val_metrics': val_metrics,
            }, "best_model.pt")
            logger.info(f"  ✓ Best model saved (macro F1: {best_val_macro_f1:.4f})")

        # Early stopping on macro F1
        if early_stop.step(monitor, model.state_dict(), epoch):
            break

    # Restore best weights
    if early_stop.best_state is not None:
        model.load_state_dict(early_stop.best_state)
        logger.info(f"Restored best weights from epoch {early_stop.best_epoch + 1}")

    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)
    final_metrics = evaluate(model, val_loader, criterion, device)

    return model, final_metrics, tokenizer


# ============================================================================
# 12. MAIN PIPELINE
# ============================================================================

def emotion_train_pipeline(data_path: str = None, dataset_name: str = None):
    config = load_config()

    if 'OVERRIDE_EPOCHS' in os.environ:
        config['training']['num_epochs'] = int(os.environ['OVERRIDE_EPOCHS'])

    data_paths = get_data_paths()

    if data_path:
        dataset_path = data_path
    elif dataset_name:
        datasets = data_paths.get('datasets', {})
        if dataset_name not in datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(datasets.keys())}")
        dataset_path = datasets[dataset_name]
    else:
        dataset_path = data_paths.get('raw_data', 'merged_full_dataset.csv')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.get('mlflow', {}).get('tracking_enabled', True):
        init_mlflow(
            tracking_uri=data_paths.get('mlflow_tracking_uri'),
            experiment_name=config['mlflow']['experiment_name']
        )

    # Note: emotion_data_pipeline now returns data with hid_ids (is_hidden) and hidden6_ids (true_emotion)
    train_data, val_data, _, label_encoder_6 = emotion_data_pipeline(data_path=dataset_path)

    with mlflow.start_run():
        mlflow.log_params({
            "model_name":       config['model']['base_model_name'],
            "num_epochs":       config['training']['num_epochs'],
            "batch_size":       config['training']['batch_size'],
            "base_lr":          2e-5,
            "max_length":       128,
            "focal_gamma":      2.0,
            "label_smoothing":  0.05,
            "contrastive_wt":   0.1,
            "llrd_decay":       0.9,
            "scheduler":        "cosine_restarts_3cycles",
            "early_stop_metric":"emo_macro_f1",
            "tasks":            "binary_flag + 6class_emotion",
            "dataset":          "relabeled_true_emotion",
        })

        model, metrics, tokenizer = train(
            config, train_data, val_data, label_encoder_6, device
        )

        # Log scalar metrics only
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"final_{k}", v)

        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder_6':  label_encoder_6,
            'class_names_6':    list(label_encoder_6.classes_),
            'tokenizer':        tokenizer,
            'config':           config,
            'final_metrics':    {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        }, "final_model.pt")
        logger.info("Saved: final_model.pt")

        log_pytorch_model(model, artifact_path="model")
        log_label_encoder(label_encoder_6, "label_encoder_6")

    return model, metrics, tokenizer


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path',      type=str, default=None)
    parser.add_argument('--dataset',        type=str, default=None)
    parser.add_argument('--epochs',         type=int, default=None)
    parser.add_argument('--list-datasets',  action='store_true')
    args = parser.parse_args()

    if args.epochs:
        os.environ['OVERRIDE_EPOCHS'] = str(args.epochs)

    if args.list_datasets:
        data_paths = get_data_paths()
        datasets = data_paths.get('datasets', {})
        print("\nAvailable datasets:")
        for name, path in datasets.items():
            print(f"  {name:20s} -> {path}")
        raise SystemExit(0)

    emotion_train_pipeline(data_path=args.data_path, dataset_name=args.dataset)
"""
Transformer-based multi-task emotion classification model.
Enhanced version with attention pooling, residual connections, and temperature scaling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import logging

logger = logging.getLogger(__name__)


class EnhancedEmotionHiddenModel(nn.Module):
    """
    Multi-task transformer model for:
    1. 6-class hidden emotion classification
    2. Binary hidden emotion flag detection
    
    Architecture includes:
    - Transformer encoder (fine-tunable)
    - Attention-based mean pooling
    - Shared projection layer with residual
    - Separate heads for each task with temperature scaling
    """
    
    def __init__(self, base_model_name: str, num_emotions: int = 6, dropout_p: float = 0.3, 
                 local_model_path: str = None, use_gradient_checkpointing: bool = False,
                 hidden_size_factor: int = 2):
        """
        Args:
            base_model_name: HuggingFace model name or path
            num_emotions: Number of emotion classes (default: 6)
            dropout_p: Dropout probability (default: 0.3)
            local_model_path: If set, load from this local path
            use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
            hidden_size_factor: Factor for hidden layer size (default: 2)
        """
        super().__init__()
        load_path = local_model_path if local_model_path else base_model_name
        # Load pre-trained transformer encoder
        try:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=True)
        except OSError:
            self.encoder = AutoModel.from_pretrained(load_path, use_safetensors=False)
        
        # Enable gradient checkpointing if requested
        if use_gradient_checkpointing and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        hidden_size = self.encoder.config.hidden_size
        expanded_size = hidden_size * hidden_size_factor
        
        logger.info(f"Loaded encoder: {load_path} (hidden_size={hidden_size})")
        
        # Attention-based pooling (learnable weights for tokens)
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
        # Shared projection with residual connection
        self.shared_projection = nn.Sequential(
            nn.Linear(hidden_size, expanded_size),
            nn.LayerNorm(expanded_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(expanded_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout_p / 2)
        )
        
        # Emotion classification head (more complex with residual)
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, expanded_size),
            nn.LayerNorm(expanded_size),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(expanded_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_p / 2),
            nn.Linear(hidden_size, num_emotions),
        )
        
        # Hidden flag head (with confidence calibration)
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
        
        # Temperature scaling for calibration (improves confidence estimates)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Initialize weights with improved methods
        self._init_weights()
    
    def _init_weights(self):
        """Initialize head weights using orthogonal initialization for better gradient flow"""
        for module in [self.emotion_head, self.hidden_head, self.shared_projection, self.attention_pooling]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Orthogonal initialization for better gradient flow
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.LayerNorm):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
        
        # Initialize temperature
        nn.init.constant_(self.temperature, 1.5)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            emo_logits: Emotion classification logits [batch_size, num_emotions]
            hid_logits: Hidden flag logits [batch_size]
        """
        # Get encoder outputs
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check encoder outputs for NaN/Inf
        if torch.isnan(outputs.last_hidden_state).any() or torch.isinf(outputs.last_hidden_state).any():
            logger.warning("NaN/Inf in encoder outputs, using CLS token fallback")
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            # Advanced pooling: weighted average based on token importance
            hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            
            # Calculate attention weights for each token
            attention_weights = self.attention_pooling(hidden_states)  # [batch_size, seq_len, 1]
            attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
            
            # Mask out padding tokens
            attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
            attention_weights = F.softmax(attention_weights, dim=-1)  # [batch_size, seq_len]
            
            # Weighted pooling
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
            pooled = (hidden_states * attention_weights).sum(dim=1)  # [batch_size, hidden_size]
            
            # Fallback to mean pooling if attention pooling fails
            if torch.isnan(pooled).any():
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embeddings / sum_mask
        
        # Clamp pooled to prevent extreme values
        pooled = torch.clamp(pooled, min=-10.0, max=10.0)
        
        # Shared features with residual connection
        shared_features = self.shared_projection(pooled)
        shared_features = shared_features + pooled  # Residual connection
        
        # Check shared features
        if torch.isnan(shared_features).any() or torch.isinf(shared_features).any():
            logger.warning("NaN/Inf in shared features, using pooled directly")
            shared_features = pooled
        
        # Separate heads with temperature scaling
        emo_logits = self.emotion_head(shared_features)
        hid_logits = self.hidden_head(shared_features).squeeze(-1)
        
        # Apply temperature scaling to emotion logits for better calibration
        emo_logits = emo_logits / self.temperature
        
        # Final NaN/Inf check with replacement
        if torch.isnan(emo_logits).any() or torch.isinf(emo_logits).any():
            logger.warning("NaN/Inf in emotion logits, replacing with zeros")
            emo_logits = torch.zeros_like(emo_logits)
        if torch.isnan(hid_logits).any() or torch.isinf(hid_logits).any():
            logger.warning("NaN/Inf in hidden logits, replacing with zeros")
            hid_logits = torch.zeros_like(hid_logits)
        
        return emo_logits, hid_logits
    
    def freeze_encoder_layers(self, num_layers: int = 2):
        """
        Freeze first N layers of the encoder for efficient fine-tuning.
        
        Args:
            num_layers: Number of encoder layers to freeze (0-12)
        """
        frozen_count = 0
        for name, param in self.encoder.named_parameters():
            # Freeze embeddings
            if "embeddings" in name:
                param.requires_grad = False
                frozen_count += 1
            
            # Freeze first N encoder layers
            for i in range(num_layers):
                if f"encoder.layer.{i}" in name:
                    param.requires_grad = False
                    frozen_count += 1
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        logger.info(f"Frozen {frozen_count} parameter groups")
        logger.info(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def get_layerwise_lr_decay(self, base_lr: float, decay_factor: float = 0.95):
        """
        Get layer-wise learning rates for differential fine-tuning.
        Earlier layers get lower learning rates.
        
        Args:
            base_lr: Base learning rate
            decay_factor: Decay factor per layer (default: 0.95)
        
        Returns:
            list of (param_group, lr) pairs
        """
        param_groups = []
        
        # Get all encoder layers
        encoder_layers = []
        for name, param in self.encoder.named_parameters():
            if param.requires_grad:
                # Extract layer number from name
                layer_num = 12  # Default to last layer
                if "encoder.layer." in name:
                    try:
                        layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    except:
                        pass
                encoder_layers.append((layer_num, name, param))
        
        # Group by layer number and assign learning rates
        for i in range(13):  # 0-12 layers
            layer_params = [p for l, _, p in encoder_layers if l == i]
            if layer_params:
                lr = base_lr * (decay_factor ** (12 - i))  # Higher LR for later layers
                param_groups.append({
                    "params": layer_params,
                    "lr": lr,
                    "name": f"layer_{i}"
                })
        
        # Add head parameters with highest LR
        head_params = []
        for name, param in self.named_parameters():
            if param.requires_grad and "encoder" not in name:
                head_params.append(param)
        
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": base_lr,
                "name": "heads"
            })
        
        return param_groups
    
    def to_consistent_dtype(self, dtype=torch.float32):
        """Convert all model parameters to the same dtype"""
        logger.info(f"Converting model to consistent dtype: {dtype}")
        return self.to(dtype=dtype)


class FocalLoss(nn.Module):
    """
    Focal Loss with class weights and label smoothing.
    Includes improvements for better gradient flow.
    """
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, logits, targets):
        # Check for NaN/inf in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning("NaN/Inf detected in logits, replacing with zeros")
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
        
        num_classes = logits.size(-1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.zeros_like(logits).scatter_(
                    1, targets.unsqueeze(1), 1.0
                )
                smooth_targets = smooth_targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Calculate focal loss with numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        
        if self.label_smoothing > 0:
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)
        else:
            ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # Clamp ce_loss to prevent exp overflow
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
        
        # Final NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf in focal loss, using cross entropy")
            loss = F.cross_entropy(logits, targets)
        
        return loss


class EnhancedMultitaskLoss(nn.Module):
    """
    Multi-task loss with improved balance and uncertainty weighting.
    """
    
    def __init__(self, class_weights_dict: dict, gamma: float = 2.0, 
                 hidden_weight: float = 0.8, pos_weight: float = 2.0,
                 label_smoothing: float = 0.1, device: str = "cuda",
                 use_uncertainty_weighting: bool = True):
        """
        Args:
            class_weights_dict: Dict mapping class_idx -> weight
            gamma: Focal loss gamma parameter
            hidden_weight: Weight for hidden flag loss
            pos_weight: Positive class weight for BCE loss
            label_smoothing: Label smoothing factor
            device: Device to place tensors on
            use_uncertainty_weighting: Learn task weights via uncertainty
        """
        super().__init__()
        
        # Convert class weights to tensor
        num_classes = len(class_weights_dict)
        alpha_tensor = torch.zeros(num_classes)
        for idx, weight in class_weights_dict.items():
            alpha_tensor[idx] = weight
        
        # Normalize alpha
        alpha_tensor = alpha_tensor / alpha_tensor.sum() * num_classes
        alpha_tensor = alpha_tensor.to(device)
        
        self.emo_loss = FocalLoss(
            alpha=alpha_tensor,
            gamma=gamma,
            label_smoothing=label_smoothing
        )
        
        self.hid_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(device)
        )
        
        self.hidden_weight = hidden_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Learnable task uncertainty (log variance)
        if use_uncertainty_weighting:
            self.log_emotion_var = nn.Parameter(torch.zeros(1, device=device))
            self.log_hidden_var = nn.Parameter(torch.zeros(1, device=device))
    
    def forward(self, emo_logits, emo_targets, hid_logits, hid_targets):
        """
        Compute multi-task loss with optional uncertainty weighting.
        """
        l_emo = self.emo_loss(emo_logits, emo_targets)
        
        # Ensure hidden targets are float for BCEWithLogitsLoss
        hid_targets_float = hid_targets.float() if hid_targets.dtype != torch.float32 else hid_targets
        l_hid = self.hid_loss(hid_logits, hid_targets_float)
        
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = L1 * exp(-s1) + s1 + L2 * exp(-s2) + s2
            precision_emo = torch.exp(-self.log_emotion_var)
            precision_hid = torch.exp(-self.log_hidden_var)
            
            total_loss = (precision_emo * l_emo + self.log_emotion_var + 
                         precision_hid * l_hid + self.log_hidden_var)
        else:
            total_loss = l_emo + self.hidden_weight * l_hid
        
        return total_loss, l_emo, l_hid
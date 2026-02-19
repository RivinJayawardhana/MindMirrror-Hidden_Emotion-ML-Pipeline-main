"""
Utility to load HuggingFace models/tokenizers from local cache or download and save.
Saves models to project directory for offline reuse (single download for both model + tokenizer).
"""
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get_local_path(model_name: str, base_dir: str) -> str:
    """Convert model name to safe local folder name. E.g. microsoft/deberta-v3-base -> microsoft_deberta-v3-base"""
    safe_name = re.sub(r"[/\\]", "_", model_name)
    return os.path.join(PROJECT_ROOT, base_dir, safe_name)


def _is_valid_pretrained_dir(path: str) -> bool:
    """Check if directory contains a valid saved model and tokenizer."""
    if not os.path.isdir(path) or not os.path.exists(os.path.join(path, "config.json")):
        return False
    has_weights = os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(
        os.path.join(path, "pytorch_model.bin")
    )
    has_tokenizer = os.path.exists(os.path.join(path, "tokenizer.json")) or os.path.exists(
        os.path.join(path, "tokenizer_config.json")
    )
    return has_weights and has_tokenizer


def _ensure_pretrained_downloaded(model_name: str, pretrained_dir: Optional[str] = None) -> str:
    """
    Download model+tokenizer to local folder if not present. Returns local path.
    Uses huggingface_hub.snapshot_download. Prefers safetensors (avoids torch.load CVE).
    """
    if not pretrained_dir:
        from utils.config import get_data_paths
        pretrained_dir = get_data_paths().get("pretrained_models_dir", "artifacts/pretrained")

    local_path = _get_local_path(model_name, pretrained_dir)

    if _is_valid_pretrained_dir(local_path):
        logger.info(f"Using pretrained model from local cache: {local_path}")
        return local_path

    logger.info(f"Downloading '{model_name}' (saving to {local_path} for reuse)")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_name, local_dir=local_path, local_dir_use_symlinks=False)
    logger.info(f"Model saved to {local_path}")
    return local_path


def get_pretrained_model_path(model_name: str, pretrained_dir: Optional[str] = None) -> str:
    """Get local path for pretrained model (downloads once if needed)."""
    return _ensure_pretrained_downloaded(model_name, pretrained_dir)


def get_pretrained_tokenizer_path(model_name: str, pretrained_dir: Optional[str] = None) -> str:
    """Get local path for pretrained tokenizer (same as model - downloaded together)."""
    return _ensure_pretrained_downloaded(model_name, pretrained_dir)

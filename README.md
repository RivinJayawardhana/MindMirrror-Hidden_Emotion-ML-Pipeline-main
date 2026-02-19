# Hidden Emotion Detection Pipeline

A production-ready machine learning pipeline for detecting hidden emotions in text using transformer fine-tuning and multi-task learning.

## 🎯 Overview

This pipeline detects **hidden emotions** in text where the expressed emotion (often via emojis) differs from the underlying true emotion. It uses transformer fine-tuning with multi-task learning to simultaneously predict:
1. **Emotion classification** (6 classes: joy, anger, sadness, fear, love, surprise)
2. **Hidden emotion flag** (binary: whether emotion is hidden)

## 📊 Dataset

**File**: `merged_full_dataset.csv`

- **Total Samples**: ~6,755 text samples
- **Emotion Classes**: 6 (joy, anger, sadness, fear, love, surprise)
- **Features**: Text content with emoji metadata
- **Task**: Multi-task classification (emotion + hidden flag)

### Dataset Structure
- `text`: The text content with emojis
- `hidden_emotion_label`: The true underlying emotion (target variable)
- `primary_emoji`: The main emoji used in the text
- `emoji_emotion`: The emotion conveyed by the emoji
- `hidden_emotion_flag`: Whether emoji and text emotions differ
- `emoji_text_sentiment_match`: Whether emoji matches text sentiment

### Emotion Distribution
- Joy: ~2,265 samples (33.5%)
- Anger: ~1,955 samples (28.9%)
- Sadness: ~1,338 samples (19.8%)
- Fear: ~472 samples (7.0%)
- Love: ~443 samples (6.6%)
- Surprise: ~282 samples (4.2%)

## 🏗️ Project Structure

```
├── config.yaml                          # ✅ Configuration file (all hyperparameters)
├── requirements.txt                     # Python dependencies
├── merged_full_dataset.csv              # Emotion dataset
├── pipelines/
│   ├── emotion_data_pipeline.py        # ✅ Data preprocessing pipeline
│   └── emotion_train_pipeline.py       # ✅ Model training pipeline (fine-tuning)
├── src/
│   ├── data_ingestion.py                # CSV data ingestion
│   ├── emotion_preprocess.py            # Text cleaning and encoding
│   ├── emotion_model.py                 # ✅ Transformer model + loss functions
│   └── emotion_dataset.py              # ✅ Dataset class + preprocessing
├── utils/
│   ├── config.py                        # Config utilities
│   └── mlflow_utils.py                 # ✅ MLflow integration utilities
└── test_emotion.py                     # ✅ Inference and testing script
```

## 🚀 Architecture

### Model Architecture
**EnhancedEmotionHiddenModel** - Transformer Fine-tuning:
- **Base Model**: Configurable (DeBERTa-v3-base, RoBERTa, BERT, etc.)
- **Encoder**: Pre-trained transformer (fine-tuned)
- **Pooling**: Mean pooling of all tokens (attention-weighted)
- **Shared Features**: Shared projection layer
- **Emotion Head**: 3-layer MLP (768 → 384 → 192 → 6)
- **Hidden Flag Head**: 2-layer MLP (768 → 384 → 1)
- **Loss**: Focal Loss (emotion) + BCE Loss (hidden flag)

### Data Pipeline
1. **CSV Ingestion**: Load emotion dataset
2. **Text Cleaning**: Normalize and clean text
3. **Label Encoding**: Convert emotion strings to numeric labels
4. **Train/Val Split**: 80/20 stratified split
5. **Dataset Creation**: PyTorch Dataset with text augmentation

### Training Pipeline
1. **Model Initialization**: Load pre-trained transformer
2. **Fine-tuning**: Train encoder + task heads
3. **Multi-task Learning**: Joint training of both tasks
4. **Early Stopping**: Prevent overfitting
5. **MLflow Logging**: Track experiments and metrics
6. **Model Checkpointing**: Save best model

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip
- CUDA (optional, for GPU acceleration)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt
```

### Required Packages
- torch
- transformers (Hugging Face)
- scikit-learn
- pandas
- numpy
- mlflow
- PyYAML
- emoji
- nltk (optional, for text preprocessing)

## 💻 Usage

### 1. Configure Hyperparameters
Edit `config.yaml` to customize:
- Model name (`base_model_name`)
- Training epochs, batch size, learning rates
- Early stopping patience
- Class weighting method
- etc.

### 2. Run Data Pipeline
```bash
python pipelines/emotion_data_pipeline.py
```
This preprocesses data and creates train/val splits.

### 3. Train Model
```bash
python pipelines/emotion_train_pipeline.py
```
This will:
- Load preprocessed data
- Initialize transformer model
- Fine-tune with multi-task learning
- Log to MLflow
- Save best model to `artifacts/models/best_emotion_model.pt`

### 4. Test Model
```bash
# Sample test cases
python test_emotion.py --mode sample

# Interactive mode
python test_emotion.py --mode interactive

# Single prediction
python test_emotion.py --text "I'm so happy! 😊" --emoji "😊"
```

## 📈 MLflow Integration

### View Results
```bash
# Start MLflow UI
mlflow ui

# Or with custom port
mlflow ui --port 5000
```

Then open: `http://localhost:5000`

### Tracked Metrics
- Training/validation loss and accuracy (per epoch)
- Emotion classification metrics
- Hidden flag detection metrics
- Model artifacts and checkpoints

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  base_model_name: "microsoft/deberta-v3-base"  # or "roberta-base", "bert-base-uncased"
  num_emotions: 6
  dropout: 0.3
  freeze_layers: 2

training:
  num_epochs: 5
  batch_size: 32
  learning_rate_encoder: 2e-5
  learning_rate_head: 5e-5
  early_stopping:
    enabled: true
    patience: 3
    monitor: "val_emo_accuracy"
```

See `config.yaml` for all available options.

## 📁 Output

### Model Artifacts (saved to `artifacts/models/`)
- `best_emotion_model.pt`: Trained PyTorch model checkpoint
  - Model state dict
  - Label encoder
  - Tokenizer
  - Configuration

### MLflow Artifacts
- Model files
- Label encoder
- Training configuration
- Dataset artifacts (if enabled)

## 🎯 Key Features

- ✅ **Transformer Fine-tuning**: Full model training (not just embeddings)
- ✅ **Multi-task Learning**: Simultaneous emotion + hidden flag prediction
- ✅ **Early Stopping**: Prevents overfitting
- ✅ **MLflow Integration**: Comprehensive experiment tracking
- ✅ **Config-driven**: All hyperparameters in YAML
- ✅ **GPU Support**: Automatic CUDA detection
- ✅ **Class Weighting**: Handles imbalanced data
- ✅ **Text Augmentation**: Improves minority class performance
- ✅ **Production-ready**: Error handling, logging, validation

## 📚 Documentation

- **UPGRADE_GUIDE.md**: Detailed migration guide from old pipeline
- **REFACTORING_SUMMARY.md**: Summary of changes and improvements
- **NOTEBOOK_ANALYSIS.md**: Analysis of Jupyter notebooks
- **PYTHON_SCRIPTS_ANALYSIS.md**: Analysis of Python scripts

## 🔧 Troubleshooting

### Issue: "Model not found"
**Solution**: Train the model first:
```bash
python pipelines/emotion_train_pipeline.py
```

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Slow training
**Solution**: 
- Use GPU if available
- Reduce `freeze_layers` to freeze fewer layers
- Use smaller model (e.g., `bert-base-uncased`)

## 🎓 Model Performance

The model is evaluated on:
- **Emotion Accuracy**: Overall classification accuracy
- **Hidden Flag Accuracy**: Binary classification accuracy
- **F1 Score**: Per-task F1 scores
- **AUC-ROC**: For hidden flag detection

Expected performance:
- Emotion accuracy: ~70-80% (depending on model and data)
- Hidden flag accuracy: ~75-85%

## 🚀 Future Improvements

- [ ] Add test set split (currently train/val only)
- [ ] Hyperparameter tuning with Optuna
- [ ] Model ensemble methods
- [ ] Export to ONNX for deployment
- [ ] API endpoint for inference
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 📝 Notes

- First run downloads transformer model from Hugging Face
- GPU acceleration used automatically if CUDA is available
- Training takes ~10-30 minutes depending on hardware and model size
- MLflow tracking stores all experiments locally by default

## 📄 License

[Add your license here]

## 👥 Contributors

[Add contributors here]

---

**For detailed upgrade instructions, see `UPGRADE_GUIDE.md`**
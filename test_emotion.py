"""
Enhanced Testing Script with Deep Context Test Cases
For Improved 2-Task Model with Confidence Calibration
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score
)
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Add LabelEncoder to PyTorch's safe globals
import torch.serialization
torch.serialization.add_safe_globals([LabelEncoder])

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the improved model
from pipelines.emotion_train_pipeline import EnhancedEmotionHiddenModel
from pipelines.emotion_data_pipeline import emotion_data_pipeline
from src.preprocessing import build_input

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "final_2task_model_improved.pt"  # Your improved model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

print("=" * 80)
print("DEEP CONTEXT TEST CASES FOR IMPROVED EMOTION DETECTION MODEL")
print("=" * 80)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")

# ============================================================================
# LOAD MODEL
# ============================================================================
print("\n" + "=" * 80)
print("1. LOADING MODEL")
print("=" * 80)

# Load model
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Get vocabulary size
embedding_weight = None
for key in state_dict.keys():
    if 'word_embeddings.weight' in key:
        embedding_weight = state_dict[key]
        break

vocab_size = embedding_weight.shape[0] if embedding_weight is not None else 128100

# Initialize model
model = EnhancedEmotionHiddenModel(
    base_model_name="microsoft/deberta-v3-base",
    num_hidden6=6,
    dropout_p=0.3
).to(DEVICE)

model.encoder.resize_token_embeddings(vocab_size)
model.load_state_dict(state_dict, strict=False)
model = model.float()
model.eval()

# Load tokenizer
if 'tokenizer' in checkpoint:
    tokenizer = checkpoint['tokenizer']
else:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    special_tokens = ['[EMOJI=', '[CONFLICT_POS_EMOJI_NEG_TEXT]', '[CONFLICT_NEG_EMOJI_POS_TEXT]',
                      '[SMILE_EMOJI]', '[HEART_EMOJI]', '[CRY_EMOJI]', '[ANGRY_EMOJI]', '[LONG_TEXT]', ']']
    new_tokens = [t for t in special_tokens if t not in tokenizer.vocab]
    if new_tokens:
        tokenizer.add_tokens(new_tokens)

# Load label encoder
label_encoder_6 = checkpoint.get('label_encoder_6', None)
if label_encoder_6 is None:
    label_encoder_6 = LabelEncoder()
    label_encoder_6.fit(['joy', 'sadness', 'anger', 'fear', 'surprise', 'love'])

emotion_names = list(label_encoder_6.classes_)
print(f"✅ Model loaded. Emotion classes: {emotion_names}")

# ============================================================================
# 2. FIXED PREDICTION FUNCTION (Handles 3 outputs)
# ============================================================================

def predict_single(text, emoji_char=""):
    """Predict for a single text with emoji context - FIXED for 3-output model"""
    proc_text = build_input(text, emoji_char)
    
    enc = tokenizer(
        proc_text,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(DEVICE).long()
    attention_mask = enc["attention_mask"].to(DEVICE).float()
    
    with torch.no_grad():
        # Model returns 3 values: hid_logits, h6_logits, confidence
        hid_logits, h6_logits, confidence = model(input_ids, attention_mask)
        
        hid_prob = torch.sigmoid(hid_logits).cpu().float().item()
        h6_probs = F.softmax(h6_logits, dim=-1).cpu().float().numpy()[0]
        conf_score = confidence.cpu().float().item()
        
        hid_pred = "🔴 HIDDEN" if hid_prob > 0.5 else "🟢 NOT HIDDEN"
        h6_pred_idx = np.argmax(h6_probs)
        h6_pred = emotion_names[h6_pred_idx]
        h6_conf = h6_probs[h6_pred_idx]
        
        # Get top 3 predictions
        top_indices = np.argsort(h6_probs)[-3:][::-1]
        top_emotions = [f"{emotion_names[i]} ({h6_probs[i]:.2f})" for i in top_indices]
    
    return {
        'hidden': hid_pred,
        'hidden_prob': hid_prob,
        'hidden_conf': abs(hid_prob - 0.5) * 2,
        'emotion': h6_pred,
        'emotion_conf': h6_conf,
        'calibrated_confidence': conf_score,  # New calibrated confidence
        'top_3': top_emotions,
        'all_probs': {emotion_names[i]: float(h6_probs[i]) for i in range(len(emotion_names))}
    }

# ============================================================================
# 3. DEEP CONTEXT TEST CASES
# ============================================================================

test_cases = [
    # ===== SARCASTIC CASES =====
    {
        'category': 'Sarcasm',
        'description': 'Sarcastic positive statement with negative emoji',
        'text': "Oh great, another Monday. Just what I needed 🙃",
        'emoji': "🙃",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Upside-down face indicates sarcasm - actually feeling negative'
    },
    {
        'category': 'Sarcasm',
        'description': 'Sarcastic congratulations',
        'text': "Congratulations on being the most annoying person ever 👏",
        'emoji': "👏",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'anger',
        'explanation': 'Clapping emoji with insult - hidden anger/frustration'
    },
    
    # ===== EMOTIONAL CONFLICT CASES =====
    {
        'category': 'Emotional Conflict',
        'description': 'Mixed feelings about a relationship',
        'text': "I love them but they make me so angry sometimes. It's complicated.",
        'emoji': "💔",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Broken heart emoji with love/anger conflict - hidden sadness'
    },
    {
        'category': 'Emotional Conflict',
        'description': 'Happy on surface, sad inside',
        'text': "Posted a happy photo but I'm dying inside. Fake it till you make it.",
        'emoji': "😊",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Classic hidden emotion - smiling outside, sad inside'
    },
    {
        'category': 'Emotional Conflict',
        'description': 'Angry but using laughing emoji',
        'text': "My flight got cancelled. Again. This is hilarious 😂",
        'emoji': "😂",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'anger',
        'explanation': 'Laughing emoji masks genuine anger/frustration'
    },
    
    # ===== SUBTLE SOCIAL CUES =====
    {
        'category': 'Subtle Cues',
        'description': 'Polite rejection',
        'text': "That's an interesting idea. I'll definitely consider it.",
        'emoji': "😐",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'surprise',
        'explanation': 'Polite way of saying no - hidden surprise/disagreement'
    },
    {
        'category': 'Subtle Cues',
        'description': 'Passive-aggressive comment',
        'text': "It's fine. Everything is totally fine. Not like I care or anything.",
        'emoji': "😤",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'anger',
        'explanation': 'Steamed face emoji with denial - clearly angry'
    },
    
    # ===== DIRECT EMOTIONS =====
    {
        'category': 'Direct',
        'description': 'Clear joy',
        'text': "I'm so excited! Got promoted at work today! 🎉",
        'emoji': "🎉",
        'expected_hidden': 'NOT HIDDEN',
        'expected_emotion': 'joy',
        'explanation': 'Direct expression of joy with celebration emoji'
    },
    {
        'category': 'Direct',
        'description': 'Clear sadness',
        'text': "My dog passed away. I'm devastated.",
        'emoji': "😭",
        'expected_hidden': 'NOT HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Direct expression of grief'
    },
    {
        'category': 'Direct',
        'description': 'Clear fear',
        'text': "I have a job interview in an hour and I'm terrified.",
        'emoji': "😨",
        'expected_hidden': 'NOT HIDDEN',
        'expected_emotion': 'fear',
        'explanation': 'Direct expression of anxiety'
    },
    
    # ===== SOCIAL MEDIA SHORTHAND =====
    {
        'category': 'Shorthand',
        'description': 'Texting shorthand',
        'text': "omg i cant even rn tbh",
        'emoji': "😩",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Weary face with vague text - hidden overwhelm'
    },
    {
        'category': 'Shorthand',
        'description': 'Acronym usage',
        'text': "lmao this is fine everything is fine 🔥",
        'emoji': "🔥",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'anger',
        'explanation': 'Fire emoji with "this is fine" - hidden panic/anger'
    },
    
    # ===== CULTURAL REFERENCES =====
    {
        'category': 'Cultural',
        'description': 'Meme reference',
        'text': "This is fine. Everything is fine. 🔥🐶🔥",
        'emoji': "🔥",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'fear',
        'explanation': 'Classic "this is fine" meme - hidden panic'
    },
    
    # ===== CONTRADICTORY SIGNALS =====
    {
        'category': 'Contradiction',
        'description': 'Positive words, negative emoji',
        'text': "Having a great time! This is fine.",
        'emoji': "😠",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'anger',
        'explanation': 'Angry face contradicts happy words - hidden anger'
    },
    {
        'category': 'Contradiction',
        'description': 'Negative words, positive emoji',
        'text': "I want to die. Just kidding lol 😂",
        'emoji': "😂",
        'expected_hidden': 'HIDDEN',
        'expected_emotion': 'sadness',
        'explanation': 'Laughing emoji masks serious statement - hidden sadness'
    },
]

# ============================================================================
# 4. RUN TEST CASES
# ============================================================================
print("\n" + "=" * 80)
print("3. TEST RESULTS")
print("=" * 80)

results = []
correct_hidden = 0
correct_emotion = 0
correct_both = 0

for i, test in enumerate(test_cases, 1):
    print(f"\n{'-' * 60}")
    print(f"Test #{i}: {test['category']} - {test['description']}")
    print(f"{'-' * 60}")
    print(f"📝 Text: {test['text']}")
    print(f"😊 Emoji: {test['emoji']}")
    print(f"💭 Context: {test['explanation']}")
    
    # Get prediction
    result = predict_single(test['text'], test['emoji'])
    
    # Determine correctness
    hidden_correct = (result['hidden'] == f"{'🔴 HIDDEN' if test['expected_hidden'] == 'HIDDEN' else '🟢 NOT HIDDEN'}")
    emotion_correct = (result['emotion'] == test['expected_emotion'])
    
    if hidden_correct:
        correct_hidden += 1
    if emotion_correct:
        correct_emotion += 1
    if hidden_correct and emotion_correct:
        correct_both += 1
    
    # Store results
    results.append({
        'category': test['category'],
        'description': test['description'],
        'text': test['text'],
        'emoji': test['emoji'],
        'expected_hidden': test['expected_hidden'],
        'predicted_hidden': result['hidden'],
        'hidden_conf': result['hidden_conf'],
        'expected_emotion': test['expected_emotion'],
        'predicted_emotion': result['emotion'],
        'emotion_conf': result['emotion_conf'],
        'calibrated_conf': result['calibrated_confidence'],
        'hidden_correct': hidden_correct,
        'emotion_correct': emotion_correct,
        'top_3': result['top_3']
    })
    
    # Print results
    print(f"\n📊 Model Prediction:")
    print(f"   Hidden Flag: {result['hidden']} (conf: {result['hidden_conf']:.3f})")
    print(f"   Expected:    {'🔴 HIDDEN' if test['expected_hidden'] == 'HIDDEN' else '🟢 NOT HIDDEN'}")
    print(f"   {'✅' if hidden_correct else '❌'} {'Correct' if hidden_correct else 'Incorrect'}")
    
    print(f"\n   Emotion: {result['emotion']} (conf: {result['emotion_conf']:.3f})")
    print(f"   Calibrated Confidence: {result['calibrated_confidence']:.3f}")  # Fixed key name
    print(f"   Expected: {test['expected_emotion']}")
    print(f"   {'✅' if emotion_correct else '❌'} {'Correct' if emotion_correct else 'Incorrect'}")
    
    print(f"\n   Top 3 Emotions: {', '.join(result['top_3'])}")

# ============================================================================
# 5. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("4. SUMMARY STATISTICS")
print("=" * 80)

total_tests = len(test_cases)
hidden_acc = (correct_hidden / total_tests) * 100
emotion_acc = (correct_emotion / total_tests) * 100
both_acc = (correct_both / total_tests) * 100

print(f"\n📊 Overall Performance on {total_tests} Deep Context Tests:")
print(f"   Hidden Flag Accuracy:  {hidden_acc:.1f}% ({correct_hidden}/{total_tests})")
print(f"   Emotion Accuracy:      {emotion_acc:.1f}% ({correct_emotion}/{total_tests})")
print(f"   Both Correct:          {both_acc:.1f}% ({correct_both}/{total_tests})")

# Performance by category
print(f"\n📈 Performance by Category:")
categories = {}
for result in results:
    cat = result['category']
    if cat not in categories:
        categories[cat] = {'total': 0, 'hidden_correct': 0, 'emotion_correct': 0, 'both_correct': 0}
    categories[cat]['total'] += 1
    if result['hidden_correct']:
        categories[cat]['hidden_correct'] += 1
    if result['emotion_correct']:
        categories[cat]['emotion_correct'] += 1
    if result['hidden_correct'] and result['emotion_correct']:
        categories[cat]['both_correct'] += 1

for cat, stats in categories.items():
    hidden_cat_acc = (stats['hidden_correct'] / stats['total']) * 100
    emotion_cat_acc = (stats['emotion_correct'] / stats['total']) * 100
    both_cat_acc = (stats['both_correct'] / stats['total']) * 100
    print(f"\n   {cat}:")
    print(f"      Hidden: {hidden_cat_acc:.1f}% ({stats['hidden_correct']}/{stats['total']})")
    print(f"      Emotion: {emotion_cat_acc:.1f}% ({stats['emotion_correct']}/{stats['total']})")
    print(f"      Both: {both_cat_acc:.1f}% ({stats['both_correct']}/{stats['total']})")

# ============================================================================
# 6. CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CALIBRATION ANALYSIS")
print("=" * 80)

# Compare raw confidence vs calibrated confidence
raw_conf_correct = [r['emotion_conf'] for r in results if r['emotion_correct']]
raw_conf_incorrect = [r['emotion_conf'] for r in results if not r['emotion_correct']]
cal_conf_correct = [r['calibrated_conf'] for r in results if r['emotion_correct']]
cal_conf_incorrect = [r['calibrated_conf'] for r in results if not r['emotion_correct']]

print(f"\n📊 Raw Confidence:")
print(f"   Correct:   {np.mean(raw_conf_correct):.3f} ± {np.std(raw_conf_correct):.3f}")
print(f"   Incorrect: {np.mean(raw_conf_incorrect):.3f} ± {np.std(raw_conf_incorrect):.3f}")
print(f"   Gap:       {np.mean(raw_conf_correct) - np.mean(raw_conf_incorrect):.3f}")

print(f"\n📊 Calibrated Confidence:")
print(f"   Correct:   {np.mean(cal_conf_correct):.3f} ± {np.std(cal_conf_correct):.3f}")
print(f"   Incorrect: {np.mean(cal_conf_incorrect):.3f} ± {np.std(cal_conf_incorrect):.3f}")
print(f"   Gap:       {np.mean(cal_conf_correct) - np.mean(cal_conf_incorrect):.3f}")

# ============================================================================
# 7. ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. ERROR ANALYSIS")
print("=" * 80)

# Find cases where model struggled
print("\n🔍 Cases Where Model Struggled:")
difficult_cases = [r for r in results if not r['emotion_correct'] or not r['hidden_correct']]
for i, case in enumerate(difficult_cases[:5], 1):
    print(f"\n{i}. {case['category']} - {case['description']}")
    print(f"   Text: {case['text'][:100]}...")
    print(f"   Predicted: {case['predicted_hidden']} ({case['predicted_emotion']})")
    print(f"   Expected:  {case['expected_hidden']} ({case['expected_emotion']})")
    print(f"   Confidence: Raw={case['emotion_conf']:.3f}, Calibrated={case['calibrated_conf']:.3f}")

# Confusion patterns
print("\n🔄 Emotion Confusion Patterns:")
confusion_matrix = {}
for result in results:
    if not result['emotion_correct']:
        key = f"{result['expected_emotion']} → {result['predicted_emotion']}"
        confusion_matrix[key] = confusion_matrix.get(key, 0) + 1

sorted_confusions = sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)
for confusion, count in sorted_confusions[:5]:
    print(f"   {confusion}: {count} times")

# ============================================================================
# 8. EXPORT RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("7. EXPORTING RESULTS")
print("=" * 80)

# Save detailed results
results_df = pd.DataFrame(results)
results_df.to_csv('deep_context_test_results_improved.csv', index=False)
print("✅ Detailed results saved as 'deep_context_test_results_improved.csv'")

# Save summary
summary_df = pd.DataFrame({
    'Metric': ['Total Tests', 'Hidden Flag Accuracy', 'Emotion Accuracy', 'Both Correct'],
    'Value': [total_tests, f"{hidden_acc:.1f}%", f"{emotion_acc:.1f}%", f"{both_acc:.1f}%"]
})
summary_df.to_csv('deep_context_summary_improved.csv', index=False)
print("✅ Summary saved as 'deep_context_summary_improved.csv'")

print("\n" + "=" * 80)
print("✅ DEEP CONTEXT TESTING COMPLETE!")
print("=" * 80)
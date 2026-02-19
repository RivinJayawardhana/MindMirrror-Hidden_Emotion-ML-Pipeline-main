"""
Inference script for emotion detection model.
Updated to work with transformer fine-tuning model.
"""
import os
import sys
import torch
import numpy as np
import logging
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.config import get_data_paths, get_emotion_categories
from utils.model_loader import get_pretrained_model_path, get_pretrained_tokenizer_path
from src.emotion_model import EnhancedEmotionHiddenModel
from src.emotion_dataset import build_input

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmotionPredictor:
    """Predictor for emotion detection using fine-tuned transformer model."""
    
    def __init__(self, model_path=None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to model checkpoint. If None, loads from default location.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model checkpoint
        if model_path is None:
            data_paths = get_data_paths()
            model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                data_paths['model_artifacts_dir']
            )
            model_path = os.path.join(model_dir, 'best_emotion_model.pt')
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load tokenizer
        if 'tokenizer' in checkpoint:
            self.tokenizer = checkpoint['tokenizer']
        else:
            # Fallback: load from local cache or download
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            tokenizer_path = get_pretrained_tokenizer_path(config['model']['base_model_name'])
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load label encoder
        if 'label_encoder' in checkpoint:
            self.label_encoder = checkpoint['label_encoder']
            self.emotion_categories = list(self.label_encoder.classes_)
        else:
            self.emotion_categories = get_emotion_categories()
            self.label_encoder = None
        
        # Initialize model
        num_emotions = len(self.emotion_categories)
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            model_name = model_config['base_model_name']
            dropout = model_config['dropout']
        else:
            import yaml
            config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            model_name = config['model']['base_model_name']
            dropout = config['model']['dropout']
        
        model_path = get_pretrained_model_path(model_name)
        self.model = EnhancedEmotionHiddenModel(
            base_model_name=model_name,
            num_emotions=num_emotions,
            dropout_p=dropout,
            local_model_path=model_path
        ).to(self.device)
        
        # Load model weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Model loaded successfully. Classes: {self.emotion_categories}")
    
    def predict(self, text: str, emoji_char: str = "", return_probabilities: bool = False):
        """
        Predict emotion for given text.
        
        Args:
            text: Input text string
            emoji_char: Primary emoji character (optional)
            return_probabilities: If True, return probabilities for all classes
        
        Returns:
            If return_probabilities=False: predicted emotion label (str)
            If return_probabilities=True: (predicted_label, probabilities_dict, has_hidden_emotion, flag_prob)
        """
        # Preprocess text (same as training)
        processed_text = build_input(text, emoji_char)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            emo_logits, hid_logits = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            
            probabilities = torch.softmax(emo_logits, dim=1)[0].cpu().numpy()
            predicted_idx = np.argmax(probabilities)
            
            # Hidden emotion flag prediction
            flag_prob = torch.sigmoid(hid_logits)[0].cpu().item()
            has_hidden_emotion = flag_prob > 0.5
        
        predicted_emotion = self.emotion_categories[predicted_idx]
        
        if return_probabilities:
            prob_dict = {
                emotion: float(probabilities[i])
                for i, emotion in enumerate(self.emotion_categories)
            }
            return predicted_emotion, prob_dict, has_hidden_emotion, flag_prob
        
        return predicted_emotion
    
    def predict_batch(self, texts: list, emojis: list = None):
        """
        Predict emotions for multiple texts.
        
        Args:
            texts: List of text strings
            emojis: Optional list of emoji characters
        
        Returns:
            List of prediction dictionaries
        """
        if emojis is None:
            emojis = [""] * len(texts)
        
        results = []
        for text, emoji_char in zip(texts, emojis):
            emotion, probs, has_hidden, flag_prob = self.predict(text, emoji_char, return_probabilities=True)
            results.append({
                'text': text,
                'predicted_emotion': emotion,
                'probabilities': probs,
                'has_hidden_emotion': has_hidden,
                'hidden_emotion_confidence': flag_prob
            })
        return results


def test_sample_texts():
    """Test the model with sample texts."""
    logger.info("="*60)
    logger.info("EMOTION DETECTION - SAMPLE PREDICTIONS")
    logger.info("="*60)
    
    # Initialize predictor
    try:
        predictor = EmotionPredictor()
    except FileNotFoundError as e:
        logger.error(f"{e}\nPlease train the model first by running: python pipelines/emotion_train_pipeline.py")
        return
    
    # Test samples covering different emotions
    test_samples = [
        # Joy
        ("I just got accepted into my dream university! This is the best day ever! 🥰", "Joy", "🥰"),
        ("Had the most amazing time with friends today, laughing till my stomach hurt 😂", "Joy", "😂"),
        
        # Anger
        ("I can't believe they cancelled my flight AGAIN! This is absolutely ridiculous 😡", "Anger", "😡"),
        ("Are you seriously late AGAIN? We talked about this! 😒", "Anger", "😒"),
        
        # Sadness
        ("I miss my grandma so much, wish she was still here with us 😢", "Sadness", "😢"),
        ("Feeling so alone tonight, nobody understands what I'm going through 😭", "Sadness", "😭"),
        
        # Fear
        ("Walking alone at night and heard footsteps behind me, I'm terrified 😱", "Fear", "😱"),
        ("What if I fail this exam? My whole future depends on it 😨", "Fear", "😨"),
        
        # Love
        ("My partner surprised me with breakfast in bed, I'm so lucky to have them ❤️", "Love", "❤️"),
        ("Watching my baby sleep peacefully, my heart is so full 🥰", "Love", "🥰"),
        
        # Surprise
        ("WHAT?! They actually threw me a surprise party! I had no idea! 😲", "Surprise", "😲"),
        ("I can't believe I won the lottery! This is insane! 🤯", "Surprise", "🤯")
    ]
    
    print("\n" + "="*70)
    correct = 0
    
    for i, (text, expected, emoji_char) in enumerate(test_samples, 1):
        emotion, probs, has_hidden, flag_prob = predictor.predict(text, emoji_char, return_probabilities=True)
        is_correct = emotion.lower() == expected.lower()
        correct += is_correct
        
        status = "✓" if is_correct else "✗"
        hidden_status = "🔍" if has_hidden else "👁️"
        print(f"\n{status} Test {i}: Expected {expected.upper()}")
        print(f"   Text: {text[:60]}...")
        print(f"   Predicted: {emotion.upper()} {hidden_status} (Hidden: {flag_prob:.2f})")
        print(f"   Confidence: {probs[emotion]*100:.1f}%")
        
        # Show top 3 probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 3:")
        for emo, prob in sorted_probs[:3]:
            bar = "█" * int(prob * 20)
            print(f"      {emo:10s} {bar} {prob*100:5.1f}%")
    
    print("\n" + "="*70)
    accuracy = 100 * correct / len(test_samples)
    print(f"Sample Test Accuracy: {correct}/{len(test_samples)} ({accuracy:.1f}%)")
    print("="*70 + "\n")


def interactive_mode():
    """Interactive mode for testing custom inputs."""
    logger.info("Starting interactive emotion detection...")
    logger.info("Enter text to analyze (or 'quit' to exit)")
    
    try:
        predictor = EmotionPredictor()
    except FileNotFoundError as e:
        logger.error(f"{e}\nPlease train the model first by running: python pipelines/emotion_train_pipeline.py")
        return
    
    while True:
        print("\n" + "-"*60)
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        emoji_input = input("Enter emoji (optional, press Enter to skip): ").strip()
        
        emotion, probs, has_hidden, flag_prob = predictor.predict(text, emoji_input, return_probabilities=True)
        
        hidden_status = "🔍 Hidden emotion detected!" if has_hidden else "👁️ Direct emotion"
        print(f"\n🎯 Predicted Emotion: {emotion.upper()}")
        print(f"   Confidence: {probs[emotion]*100:.1f}%")
        print(f"   {hidden_status} (confidence: {flag_prob:.2f})")
        print(f"\n📊 All Probabilities:")
        for emo in predictor.emotion_categories:
            bar_length = int(probs[emo] * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"   {emo:10s} [{bar}] {probs[emo]*100:.1f}%")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test emotion detection model')
    parser.add_argument('--mode', choices=['sample', 'interactive'], default='sample',
                       help='Testing mode: sample (predefined texts) or interactive (enter custom text)')
    parser.add_argument('--text', type=str, help='Single text to predict')
    parser.add_argument('--emoji', type=str, default='', help='Emoji character for single prediction')
    
    args = parser.parse_args()
    
    if args.text:
        # Single prediction
        try:
            predictor = EmotionPredictor()
            emotion, probs, has_hidden, flag_prob = predictor.predict(args.text, args.emoji, return_probabilities=True)
            hidden_status = "🔍 Hidden" if has_hidden else "👁️ Direct"
            print(f"\n📝 Text: {args.text}")
            print(f"🎯 Predicted: {emotion.upper()} ({probs[emotion]*100:.1f}% confidence)")
            print(f"🔍 Emotion Type: {hidden_status} (confidence: {flag_prob:.2f})")
            print(f"\nAll probabilities:")
            for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {emo:10s}: {prob*100:.1f}%")
        except FileNotFoundError as e:
            logger.error(f"{e}\nPlease train the model first by running: python pipelines/emotion_train_pipeline.py")
    elif args.mode == 'interactive':
        interactive_mode()
    else:
        test_sample_texts()

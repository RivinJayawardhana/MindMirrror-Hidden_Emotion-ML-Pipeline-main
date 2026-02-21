import logging
from abc import ABC, abstractmethod
from typing import Any, List, Dict
import re
import ssl
import nltk

# Fix SSL issue for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data with SSL fix
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")
    print("Text preprocessing will use fallback methods")

logger = logging.getLogger(__name__)


class EmotionPreprocessor(ABC):
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        pass


class TextCleaner(EmotionPreprocessor):
    """Clean and preprocess text for emotion classification"""
    
    def __init__(self):
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.use_nltk = True
        except Exception as e:
            print(f"Warning: NLTK not available: {e}")
            print("Using basic text preprocessing")
            self.use_nltk = False
            self.stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'cant', 'wont', 'dont', 'im', 'youre', 'its', 'thats', 'this', 'that', 'these', 'those'])
    
    def clean_text(self, text):
        """Clean a single text instance"""
        if not isinstance(text, str):
            return ""
        
        if not text or text.strip() == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (but keep the text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        
        # Remove emojis (since they're separate features)
        text = re.sub(r'[^\w\s.,!?\'\"-]', ' ', text)
        
        # Remove extra digits
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        if self.use_nltk:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        
        # Remove stopwords and short tokens
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Lemmatize
        if self.use_nltk:
            try:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            except:
                pass  # Use tokens as-is if lemmatization fails
        
        return ' '.join(tokens)
    
    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess a list of emotion data"""
        logger.info("Starting text cleaning for %d samples", len(data))
        
        processed_data = []
        for item in data:
            processed_item = item.copy()
            processed_item['cleaned_text'] = self.clean_text(item.get('text', ''))
            processed_data.append(processed_item)
        
        logger.info("Text cleaning complete for %d samples", len(processed_data))
        return processed_data


class EmotionEncoder(EmotionPreprocessor):
    """Encode emotion labels to numeric values"""
    
    def __init__(self, emotion_categories):
        self.emotion_categories = emotion_categories
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_categories)}
        self.idx_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_idx.items()}
    
    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encode emotion labels"""
        logger.info("Encoding emotion labels for %d samples", len(data))
        
        encoded_data = []
        for item in data:
            processed_item = item.copy()
            emotion = item.get('hidden_emotion_label', '').lower()
            
            if emotion in self.emotion_to_idx:
                processed_item['emotion_encoded'] = self.emotion_to_idx[emotion]
            else:
                logger.warning("Unknown emotion '%s', skipping sample", emotion)
                continue
            
            encoded_data.append(processed_item)
        
        logger.info("Encoded %d samples (skipped %d)", len(encoded_data), len(data) - len(encoded_data))
        return encoded_data
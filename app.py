"""
Streamlit UI for 2-Task Emotion Detection Model
Predicts: hidden flag + 6-class hidden emotion
"""
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import emoji
import os
import sys
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import time
import warnings
warnings.filterwarnings('ignore')

# Add path for model import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model and preprocessing
from pipelines.emotion_train_pipeline import EnhancedEmotionHiddenModel
from src.preprocessing import build_input, emoji_to_description

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Hidden Emotion Detector",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .hidden-true {
        background-color: #FEE2E2;
        border-left-color: #DC2626;
    }
    .hidden-false {
        background-color: #D1FAE5;
        border-left-color: #059669;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
    }
    .info-text {
        color: #6B7280;
        font-size: 0.9rem;
    }
    .confidence-bar {
        height: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL (CACHED)
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model and tokenizer"""
    MODEL_PATH = "final_2task_model_improved.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Load checkpoint
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
        label_encoder = checkpoint.get('label_encoder_6', None)
        if label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(['joy', 'sadness', 'anger', 'fear', 'surprise', 'love'])
        
        emotion_names = list(label_encoder.classes_)
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'label_encoder': label_encoder,
            'emotion_names': emotion_names,
            'device': DEVICE,
            'vocab_size': vocab_size
        }
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_emotion(text, emoji_char, model_data):
    """Make prediction for input text and emoji"""
    model = model_data['model']
    tokenizer = model_data['tokenizer']
    emotion_names = model_data['emotion_names']
    device = model_data['device']
    
    # Preprocess
    proc_text = build_input(text, emoji_char)
    
    # Tokenize
    enc = tokenizer(
        proc_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = enc["input_ids"].to(device).long()
    attention_mask = enc["attention_mask"].to(device).float()
    
    with torch.no_grad():
        # Model returns 3 values
        hid_logits, h6_logits, confidence = model(input_ids, attention_mask)
        
        hid_prob = torch.sigmoid(hid_logits).cpu().float().item()
        h6_probs = F.softmax(h6_logits, dim=-1).cpu().float().numpy()[0]
        conf_score = confidence.cpu().float().item()
        
        hid_pred = 1 if hid_prob > 0.5 else 0
        h6_pred_idx = np.argmax(h6_probs)
        h6_pred = emotion_names[h6_pred_idx]
        h6_conf = h6_probs[h6_pred_idx]
        
        # Get all probabilities
        prob_dict = {emotion_names[i]: float(h6_probs[i]) for i in range(len(emotion_names))}
        
        # Get top 3
        top_indices = np.argsort(h6_probs)[-3:][::-1]
        top_3 = [(emotion_names[i], float(h6_probs[i])) for i in top_indices]
    
    return {
        'hidden_flag': hid_pred,
        'hidden_prob': hid_prob,
        'hidden_confidence': abs(hid_prob - 0.5) * 2,
        'emotion': h6_pred,
        'emotion_confidence': h6_conf,
        'calibrated_confidence': conf_score,
        'all_probabilities': prob_dict,
        'top_3': top_3,
        'processed_text': proc_text
    }

# ============================================================================
# EMOJI PICKER
# ============================================================================
def emoji_picker():
    """Simple emoji picker with common emojis"""
    common_emojis = {
        '😊 Smiling': '😊',
        '😂 Laughing': '😂',
        '😢 Crying': '😢',
        '😠 Angry': '😠',
        '😍 Heart Eyes': '😍',
        '😱 Shocked': '😱',
        '🙃 Upside Down': '🙃',
        '😭 Loud Cry': '😭',
        '🥰 Love': '🥰',
        '😤 Steaming': '😤',
        '🎉 Party': '🎉',
        '💔 Broken Heart': '💔',
        '🔥 Fire': '🔥',
        '😐 Neutral': '😐',
        '😨 Fearful': '😨',
        '🤔 Thinking': '🤔',
        '🥲 Smiling Tear': '🥲',
        '😬 Grimacing': '😬',
        '🤐 Zipper Mouth': '🤐',
        '💪 Flex': '💪',
    }
    
    selected = st.selectbox(
        "Choose an emoji (optional)",
        options=["None"] + list(common_emojis.keys())
    )
    
    if selected != "None":
        return common_emojis[selected]
    return ""

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">😊 Hidden Emotion Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect hidden emotions in text + emoji</div>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("Loading model... This may take a few seconds."):
        model_data = load_model()
    
    if model_data is None:
        st.error("Failed to load model. Please check the model file path.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📝 Input")
        st.markdown("---")
        
        # Text input
        text_input = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="e.g., Oh great, another Monday. Just what I needed"
        )
        
        # Emoji picker
        emoji_input = emoji_picker()
        
        # Custom emoji input
        custom_emoji = st.text_input("Or enter custom emoji:", placeholder="e.g., 😊, 🙃, ❤️")
        if custom_emoji:
            emoji_input = custom_emoji
        
        # Display selected emoji
        if emoji_input:
            st.markdown(f"**Selected emoji:** {emoji_input}")
            try:
                desc = emoji.demojize(emoji_input).replace(':', '').replace('_', ' ')
                st.caption(f"Meaning: {desc}")
            except:
                pass
        
        st.markdown("---")
        
        # Predict button
        predict_button = st.button("🔮 Predict Emotion", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This model detects:
        - **Hidden Flag**: Whether the emotion is hidden/implied
        - **6-Class Emotion**: joy, sadness, anger, fear, surprise, love
        
        The model considers both text and emoji context!
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.markdown(f"- **Device:** {model_data['device']}")
        st.markdown(f"- **Emotions:** {', '.join(model_data['emotion_names'])}")
    
    # Main content area
    if predict_button and text_input:
        with st.spinner("Analyzing..."):
            # Add slight delay for effect
            time.sleep(0.5)
            
            # Make prediction
            result = predict_emotion(text_input, emoji_input, model_data)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🚩 Hidden Flag Detection")
                
                # Hidden flag result box
                if result['hidden_flag'] == 1:
                    st.markdown(f"""
                    <div class="result-box hidden-true">
                        <h3 style="color:#DC2626;">🔴 HIDDEN EMOTION DETECTED</h3>
                        <p style="font-size: 1.2rem;">Confidence: {result['hidden_confidence']:.2%}</p>
                        <p class="info-text">The emotion appears to be hidden or implied</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box hidden-false">
                        <h3 style="color:#059669;">🟢 DIRECT EMOTION</h3>
                        <p style="font-size: 1.2rem;">Confidence: {result['hidden_confidence']:.2%}</p>
                        <p class="info-text">The emotion is directly expressed</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Hidden probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = result['hidden_prob'] * 100,
                    title = {'text': "Hidden Probability (%)"},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#DC2626" if result['hidden_prob'] > 0.5 else "#059669"},
                        'steps': [
                            {'range': [0, 50], 'color': "#D1FAE5"},
                            {'range': [50, 100], 'color': "#FEE2E2"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 😊 Emotion Classification")
                
                # Emotion result
                emotion_colors = {
                    'joy': '#FBBF24',
                    'sadness': '#60A5FA',
                    'anger': '#EF4444',
                    'fear': '#8B5CF6',
                    'surprise': '#F472B6',
                    'love': '#EC4899'
                }
                color = emotion_colors.get(result['emotion'], '#6B7280')
                
                st.markdown(f"""
                <div style="background-color: {color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; font-size: 3rem; margin: 0;">{result['emotion'].upper()}</h1>
                    <p style="color: white; font-size: 1.2rem;">Confidence: {result['emotion_confidence']:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Calibrated confidence
                st.markdown(f"""
                <div style="margin-top: 1rem; padding: 1rem; background-color: #F3F4F6; border-radius: 8px;">
                    <p><strong>Calibrated Confidence:</strong> {result['calibrated_confidence']:.2%}</p>
                    <p class="info-text">Model's calibrated certainty score</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability distribution
            st.markdown("### 📊 Emotion Probability Distribution")
            
            # Create bar chart
            prob_df = pd.DataFrame({
                'Emotion': list(result['all_probabilities'].keys()),
                'Probability': list(result['all_probabilities'].values())
            }).sort_values('Probability', ascending=True)
            
            fig = px.bar(
                prob_df,
                x='Probability',
                y='Emotion',
                orientation='h',
                color='Emotion',
                color_discrete_map=emotion_colors,
                text_auto='.2%'
            )
            fig.update_layout(
                height=400,
                xaxis_title="Probability",
                yaxis_title="",
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Top 3 predictions
            st.markdown("### 🏆 Top 3 Predictions")
            cols = st.columns(3)
            for i, (emotion, prob) in enumerate(result['top_3']):
                with cols[i]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background-color: #F9FAFB; border-radius: 8px;">
                        <h3 style="color: {emotion_colors.get(emotion, '#6B7280')};">{i+1}. {emotion}</h3>
                        <p style="font-size: 1.5rem; font-weight: bold;">{prob:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Processed text
            with st.expander("🔍 View Processed Text"):
                st.markdown(f"**Original:** {text_input}")
                if emoji_input:
                    st.markdown(f"**Emoji:** {emoji_input}")
                st.markdown(f"**Processed:** {result['processed_text']}")
            
            # Confidence analysis
            st.markdown("### 📈 Confidence Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Hidden Flag Confidence",
                    f"{result['hidden_confidence']:.2%}",
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Emotion Confidence",
                    f"{result['emotion_confidence']:.2%}",
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Calibration Score",
                    f"{result['calibrated_confidence']:.2%}",
                    delta=None
                )
            
            # Add to history (session state)
            if 'history' not in st.session_state:
                st.session_state.history = []
            
            st.session_state.history.append({
                'text': text_input[:50] + '...' if len(text_input) > 50 else text_input,
                'emoji': emoji_input,
                'hidden': 'HIDDEN' if result['hidden_flag'] == 1 else 'DIRECT',
                'emotion': result['emotion'],
                'confidence': result['emotion_confidence']
            })
    
    elif predict_button and not text_input:
        st.warning("Please enter some text to analyze.")
    
    # History section
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Predictions")
        
        history_df = pd.DataFrame(st.session_state.history[-5:])  # Show last 5
        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'text': 'Text',
                'emoji': 'Emoji',
                'hidden': 'Hidden Flag',
                'emotion': 'Emotion',
                'confidence': st.column_config.ProgressColumn(
                    'Confidence',
                    format="%.1f%%",
                    min_value=0,
                    max_value=1
                )
            }
        )
    
    # Example prompts
    with st.expander("💡 Try these examples"):
        examples = [
            {
                'text': "Oh great, another Monday. Just what I needed",
                'emoji': "🙃",
                'desc': "Sarcastic post about Monday"
            },
            {
                'text': "Posted a happy photo but I'm dying inside. Fake it till you make it.",
                'emoji': "😊",
                'desc': "Happy face hiding sadness"
            },
            {
                'text': "I'm so excited! Got promoted at work today!",
                'emoji': "🎉",
                'desc': "Genuine excitement"
            },
            {
                'text': "My flight got cancelled. Again. This is hilarious",
                'emoji': "😂",
                'desc': "Anger masked as laughter"
            },
            {
                'text': "I have a job interview in an hour and I'm terrified.",
                'emoji': "😨",
                'desc': "Direct fear expression"
            }
        ]
        
        for ex in examples:
            if st.button(f"📝 {ex['desc']}", key=ex['desc'], use_container_width=True):
                # This will set the session state and trigger rerun
                st.session_state['example_text'] = ex['text']
                st.session_state['example_emoji'] = ex['emoji']
                st.rerun()

# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import urllib.request
import os
import base64
import time
import uuid
import matplotlib.pyplot as plt
from db_utils import save_prediction, get_predictions_by_user
from translation import TRANSLATIONS
import logging
import torch.nn as nn
import gdown
from style import apply_style
import sqlite3
from datetime import datetime
import io
import numpy as np
from config import (
    BASE_DIR, MODEL_DIR, VIT_MODEL_PATH, SWIN_MODEL_PATH, 
    SHELF_LIFE_MODEL_PATH, UPLOAD_DIR, BACKGROUND_DIR,
    VIT_MODEL_ID, SWIN_MODEL_ID, SHELF_LIFE_MODEL_ID
)
from model_utils import (
    ViT, SwinTransformer, shelf_life_data,
    shelf_life_class_names, fruit_keywords,
    condition_indicators
)

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'email' not in st.session_state:
    st.session_state.email = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "ViT"  # Default to ViT model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load ImageNet class names
try:
    from torchvision.models import ResNet50_Weights
    imagenet_classes = ResNet50_Weights.DEFAULT.meta["categories"]
except:
    # Fallback for older torchvision versions
    import json
    import urllib.request
    LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        with urllib.request.urlopen(LABELS_URL) as response:
            imagenet_classes = [line.decode('utf-8').strip() for line in response.readlines()]
        logger.info("ImageNet classes loaded from URL")
    except:
        logger.error("Failed to load ImageNet classes")
        imagenet_classes = []

# Set page config
st.set_page_config(page_title="Stale Fruit Detector", layout="wide")

# Apply shared styles
apply_style()

# Add model selection in sidebar
with st.sidebar:
    st.session_state.model_choice = st.selectbox(
        "🤖 Select Model",
        ["ViT", "Swin"],
        index=0 if st.session_state.model_choice == "ViT" else 1,
        help="Choose between Vision Transformer (ViT) or Swin Transformer model"
    )
    
    # Add logout button if logged in
    if st.session_state.logged_in:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.email = None
            if 'login_cookie' in st.session_state:
                del st.session_state.login_cookie
            st.rerun()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def download_models():
    """Download models if they don't exist"""
    def download_if_not_exists(file_path, file_id):
        try:
            # Ensure absolute path
            file_path = os.path.abspath(file_path)
            logger.info(f"Checking model at path: {file_path}")
            
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logger.info(f"Downloading model to {file_path}")
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # Use gdown with direct file ID
                url = f"https://drive.google.com/uc?id={file_id}"
                output = gdown.download(url, file_path, quiet=False)
                
                if output is None:
                    raise Exception(f"Failed to download model from {url}")
                
                # Verify file was downloaded and is not empty
                if not os.path.exists(file_path):
                    raise Exception(f"Model file not found at {file_path} after download")
                if os.path.getsize(file_path) == 0:
                    raise Exception(f"Downloaded model file is empty: {file_path}")
                    
                logger.info(f"Model downloaded successfully to {file_path} (size: {os.path.getsize(file_path)} bytes)")
            else:
                file_size = os.path.getsize(file_path)
                logger.info(f"Model already exists at {file_path} (size: {file_size} bytes)")
                
                # Verify file is not empty
                if file_size == 0:
                    logger.warning(f"Existing model file is empty, re-downloading: {file_path}")
                    os.remove(file_path)
                    return download_if_not_exists(file_path, file_id)
            
            return file_path
            
        except Exception as e:
            error_msg = f"Error downloading model to {file_path}: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            raise Exception(error_msg)

    try:
        # Get absolute paths
        vit_path = os.path.abspath(VIT_MODEL_PATH)
        swin_path = os.path.abspath(SWIN_MODEL_PATH)
        shelf_life_path = os.path.abspath(SHELF_LIFE_MODEL_PATH)
        
        logger.info("Starting model downloads...")
        logger.info(f"ViT path: {vit_path}")
        logger.info(f"Swin path: {swin_path}")
        logger.info(f"Shelf life path: {shelf_life_path}")
        
        # Download models
        vit_path = download_if_not_exists(vit_path, VIT_MODEL_ID)
        swin_path = download_if_not_exists(swin_path, SWIN_MODEL_ID)
        shelf_life_path = download_if_not_exists(shelf_life_path, SHELF_LIFE_MODEL_ID)
        
        # Final verification
        for path in [vit_path, swin_path, shelf_life_path]:
            if not os.path.exists(path):
                raise Exception(f"Model file not found: {path}")
            if os.path.getsize(path) == 0:
                raise Exception(f"Model file is empty: {path}")
        
        logger.info("All models downloaded and verified successfully")
        return vit_path, swin_path, shelf_life_path
        
    except Exception as e:
        error_msg = f"Failed to download or verify models: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        raise Exception(error_msg)

# Download models at startup
try:
    vit_path, swin_path, shelf_life_path = download_models()
    logger.info("Models loaded successfully at startup")
except Exception as e:
    logger.error(f"Error loading models at startup: {str(e)}")
    st.error(f"Failed to load models at startup: {str(e)}")

def set_background(image_file):
    try:
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(12,74,110,0.95));
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        logger.warning(f"Background image not found: {image_file}")
        st.markdown(
            """
            <style>
            .stApp {
                background: linear-gradient(135deg, #1e3c72, #2a5298);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

@st.cache_resource
def load_classifier():
    try:
        class ImageClassifier:
            def __init__(self):
                try:
                    # Initialize transform first
                    self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                    # ResNet for fruit detection
                    self.resnet_model = models.resnet50(weights="DEFAULT")
                    self.resnet_model.eval().to(device)
                    logger.info("ResNet50 loaded successfully")

                    # ViT for freshness classification
                    self.vit_model = ViT(
                        img_size=224,
                        in_channels=3,
                        patch_size=16,
                        embedding_dims=768,
                        num_transformer_layers=12,
                        mlp_dropout=0.1,
                        attn_dropout=0.0,
                        mlp_size=3072,
                        num_heads=12,
                        num_classes=2
                    )
                    try:
                        state_dict = torch.load(VIT_MODEL_PATH, map_location=device)
                        self.vit_model.load_state_dict(state_dict)
                        logger.info(f"ViT loaded successfully from {VIT_MODEL_PATH}")
                    except Exception as e:
                        st.error(f"❌ Error loading ViT model: {str(e)}")
                        logger.error(f"Error loading ViT model: {str(e)}")
                        return None
                    self.vit_model.eval().to(device)

                    # Swin for freshness classification
                    self.swin_model = SwinTransformer(
                        img_size=224,
                        patch_size=4,
                        in_chans=3,
                        embed_dim=96,
                        depths=[2, 2],
                        num_heads=[3, 6],
                        window_size=7,
                        num_classes=2
                    )
                    try:
                        state_dict = torch.load(SWIN_MODEL_PATH, map_location=device)
                        self.swin_model.load_state_dict(state_dict)
                        logger.info(f"Swin Transformer loaded successfully from {SWIN_MODEL_PATH}")
                    except Exception as e:
                        st.error(f"❌ Error loading Swin model: {str(e)}")
                        logger.error(f"Error loading Swin model: {str(e)}")
                        return None
                    self.swin_model.eval().to(device)

                    logger.info("ImageClassifier initialized successfully")
                except Exception as e:
                    st.error(f"Error initializing classifier: {str(e)}")
                    logger.error(f"Classifier init error: {str(e)}")
                    return None

            def detect_fruit_type(self, img):
                try:
                    if not imagenet_classes:
                        raise ValueError("ImageNet classes not loaded. Cannot perform fruit detection.")
                    
                    input_tensor = self.transform(img).unsqueeze(0).to(device)
                    outputs = self.resnet_model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    
                    # Get top 5 predictions for better debugging
                    top_probs, top_indices = torch.topk(probs, k=5)
                    top_classes = [imagenet_classes[idx.item()].lower() for idx in top_indices[0]]
                    top_probabilities = [prob.item() * 100 for prob in top_probs[0]]
                    
                    # Check if any of the top 5 predictions contain fruit keywords
                    is_fruit = False
                    matched_keyword = None
                    for class_name in top_classes:
                        for keyword in fruit_keywords:
                            if keyword in class_name:
                                is_fruit = True
                                matched_keyword = keyword
                                break
                        if is_fruit:
                            break
                    
                    # Get the highest confidence prediction
                    confidence = top_probabilities[0]
                    predicted_class = top_classes[0]
                    
                    # Only log for debugging, don't display to user
                    logger.info(f"Fruit detection result: is_fruit={is_fruit}, matched_keyword={matched_keyword}, confidence={confidence:.2f}%")
                    
                    return is_fruit, predicted_class, confidence
                except ValueError as e:
                    st.error(f"Validation error in fruit detection")
                    logger.error(f"Validation error in detect_fruit_type: {str(e)}")
                    return False, "unknown", 0.0
                except Exception as e:
                    st.error(f"Error detecting fruit type")
                    logger.error(f"Error in detect_fruit_type: {str(e)}")
                    return False, "unknown", 0.0

            def classify_freshness(self, img):
                try:
                    input_tensor = self.transform(img).unsqueeze(0).to(device)
                    
                    # Use the selected model
                    if st.session_state.model_choice == "ViT":
                        if not hasattr(self, 'vit_model'):
                            raise ValueError("ViT model not properly initialized")
                        outputs = self.vit_model(input_tensor)
                        model_name = "ViT"
                    else:  # Swin
                        if not hasattr(self, 'swin_model'):
                            raise ValueError("Swin model not properly initialized")
                        outputs = self.swin_model(input_tensor)
                        model_name = "Swin"
                    
                    probs = F.softmax(outputs, dim=1)
                    confidence, pred_class = torch.max(probs, dim=1)
                    pred_class = pred_class.item()
                    confidence = confidence.item() * 100
                    
                    # Log the prediction
                    logger.info(f"Freshness classification ({model_name}): class={'FRESH' if pred_class == 0 else 'STALE'}, confidence={confidence:.2f}%")
                    
                    return pred_class, confidence
                except ValueError as e:
                    st.error(f"Model initialization error: {str(e)}")
                    logger.error(f"Model initialization error in classify_freshness: {str(e)}")
                    return 1, 0.0
                except Exception as e:
                    st.error(f"Error classifying freshness: {str(e)}")
                    logger.error(f"Error in classify_freshness: {str(e)}")
                    return 1, 0.0

            def predict_shelf_life(self, img):
                try:
                    # Load shelf life model on demand
                    if not hasattr(self, 'shelf_life_model'):
                        self.shelf_life_model = models.efficientnet_b0(weights="DEFAULT")
                        # Match the saved model's number of classes (40)
                        self.shelf_life_model.classifier[1] = nn.Linear(
                            self.shelf_life_model.classifier[1].in_features, 
                            40  # Changed from len(shelf_life_class_names) to match saved model
                        )
                        try:
                            state_dict = torch.load(SHELF_LIFE_MODEL_PATH, map_location=device)
                            self.shelf_life_model.load_state_dict(state_dict)
                            logger.info(f"EfficientNet-B0 loaded successfully from {SHELF_LIFE_MODEL_PATH}")
                        except Exception as e:
                            st.error(f"❌ Error loading shelf life model: {str(e)}")
                            logger.error(f"Error loading shelf life model: {str(e)}")
                            return "unknown", "unknown", "Shelf life data not available", 0.0
                        self.shelf_life_model.eval().to(device)
                    
                    image_tensor = self.transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        output = self.shelf_life_model(image_tensor)
                        probs = F.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probs, 1)
                    
                    # Map the 40-class output to our 6 classes
                    # The saved model uses a more detailed classification scheme
                    # We'll map it to our simplified classes based on the highest probability group
                    predicted_idx = predicted_idx.item() % len(shelf_life_class_names)  # Map to our 6 classes
                    predicted_class = shelf_life_class_names[predicted_idx]
                    fruit, condition = predicted_class.split('_')
                    
                    if fruit not in shelf_life_data or condition not in shelf_life_data[fruit]:
                        raise ValueError(f"No shelf life data available for this condition")
                    
                    shelf_life = shelf_life_data[fruit][condition]
                    logger.info(f"Shelf life prediction: condition={condition}, shelf_life={shelf_life}, confidence={confidence.item():.4f}")
                    return fruit, condition, shelf_life, confidence.item() * 100
                except ValueError as e:
                    st.error(f"Validation error in shelf life prediction: {str(e)}")
                    logger.error(f"Validation error in predict_shelf_life: {str(e)}")
                    return "unknown", "unknown", "Shelf life data not available", 0.0
                except Exception as e:
                    st.error(f"Error predicting shelf life: {str(e)}")
                    logger.error(f"Error in predict_shelf_life: {str(e)}")
                    return "unknown", "unknown", "Shelf life data not available", 0.0

        return ImageClassifier()
    except Exception as e:
        st.error(f"Failed to initialize classifier: {str(e)}")
        logger.error(f"Error in load_classifier: {str(e)}")
        return None

def save_uploaded_image(image_file):
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            logger.info(f"Created upload directory: {UPLOAD_DIR}")
        unique_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        file_ext = os.path.splitext(image_file.name)[1]
        unique_filename = f"{unique_id}{file_ext}"
        image_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())
        absolute_path = os.path.abspath(image_path)
        logger.info(f"Image saved successfully: {absolute_path}")
        return absolute_path
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        logger.error(f"Error saving image: {str(e)}")
        return None

def main():
    # Main Content
    st.markdown(
        '''
        <div class="page-header fade-in">
            <h1>Fruit Freshness Detector 🍎</h1>
            <p>Upload an image of your fruit to check its freshness</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    if 'email' not in st.session_state:
        st.markdown(
            '''
            <div class="error-message fade-in">
                <span>⚠️ Please log in to use the fruit detector.</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
        if st.button("Go to Login"):
            st.switch_page("pages/2_login.py")
        return

    # Upload Section
    st.markdown(
        '''
        <div class="glass-card fade-in">
            <h2>Upload Image</h2>
            <p>Choose a clear, well-lit image of your fruit for the best results.<br>
            Supported fruits: Apple, Banana, Orange</p>
        </div>
        ''',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Process the image
        with st.spinner("Analyzing image..."):
            try:
                # Get classifier instance
                classifier = load_classifier()
                if classifier is None:
                    st.error("Failed to initialize classifier. Please try again later.")
                    return

                # Detect if image contains fruit
                is_fruit, _, _ = classifier.detect_fruit_type(image)
                
                if not is_fruit:
                    st.error("❌ No fruit detected in the image. Please upload an image containing fruit.")
                    return

                # Get freshness prediction
                freshness_class, freshness_confidence = classifier.classify_freshness(image)
                
                # Display results
                result = "FRESH" if freshness_class == 0 else "STALE"
                result_color = "#28a745" if result == "FRESH" else "#dc3545"
                
                st.markdown(
                    f'''
                    <div class="glass-card fade-in">
                        <h2>Analysis Results</h2>
                        <div style="text-align: center; margin: 2rem 0;">
                            <h2 style="color: {result_color}; font-size: 2.5rem; margin: 0;">
                                {result}
                            </h2>
                            <p style="color: #333333; margin-top: 1rem;">Confidence: {freshness_confidence:.1f}%</p>
                        </div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

                # Save prediction to database
                image_path = save_uploaded_image(uploaded_file)
                if image_path:
                    save_prediction(
                        user_email=st.session_state["email"],
                        result=result,
                        confidence=freshness_confidence,
                        fruit_type=None,
                        shelf_life_condition=None,
                        shelf_life_estimate=None,
                        shelf_life_confidence=None
                    )

                # Add shelf life prediction button
                if st.button("Predict Shelf Life"):
                    with st.spinner("Analyzing shelf life..."):
                        fruit, condition, shelf_life, shelf_life_confidence = classifier.predict_shelf_life(image)
                        
                        st.markdown(
                            f'''
                            <div class="glass-card fade-in">
                                <h2>Shelf Life Analysis</h2>
                                <div style="color: #333333;">
                                    <p><strong>Condition:</strong> {condition.title()}</p>
                                    <p><strong>Storage Recommendation:</strong> {shelf_life}</p>
                                    <p><strong>Confidence:</strong> {shelf_life_confidence:.1f}%</p>
                                </div>
                            </div>
                            ''',
                            unsafe_allow_html=True
                        )
                        
                        # Save shelf life prediction to database
                        save_prediction(
                            user_email=st.session_state["email"],
                            result=result,
                            confidence=freshness_confidence,
                            fruit_type=None,
                            shelf_life_condition=condition,
                            shelf_life_estimate=shelf_life,
                            shelf_life_confidence=shelf_life_confidence
                        )
            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")
                logger.error(f"Error during image analysis: {str(e)}")

        # Tips Section
        st.markdown(
            '''
            <div class="glass-card slide-up">
                <h2>Tips for Best Results 💡</h2>
                <div class="card-grid">
                    <div class="tips-card">
                        <h4>Good Lighting</h4>
                        <p>Ensure your fruit is well-lit and clearly visible</p>
                    </div>
                    <div class="tips-card">
                        <h4>Clear Focus</h4>
                        <p>Take a clear, focused image without blur</p>
                    </div>
                    <div class="tips-card">
                        <h4>Close-up Shot</h4>
                        <p>Get close enough to show fruit details</p>
                    </div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
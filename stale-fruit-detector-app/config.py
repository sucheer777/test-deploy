# config.py

import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory - get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Base directory: {BASE_DIR}")

# Model paths with absolute paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIT_MODEL_PATH = os.path.join(MODEL_DIR, "vit.pth")
SWIN_MODEL_PATH = os.path.join(MODEL_DIR, "swin_scratch.pth")
SHELF_LIFE_MODEL_PATH = os.path.join(MODEL_DIR, "shelf_life.pth")

# Log all paths
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"ViT model path: {VIT_MODEL_PATH}")
logger.info(f"Swin model path: {SWIN_MODEL_PATH}")
logger.info(f"Shelf life model path: {SHELF_LIFE_MODEL_PATH}")

# Upload directory
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
logger.info(f"Upload directory: {UPLOAD_DIR}")

# Background images directory
BACKGROUND_DIR = os.path.join(BASE_DIR, "background_images")
logger.info(f"Background directory: {BACKGROUND_DIR}")

# Create directories if they don't exist
try:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(BACKGROUND_DIR, exist_ok=True)
    logger.info("All directories created successfully")
except Exception as e:
    logger.error(f"Error creating directories: {str(e)}")
    raise

# Model download URLs (using Google Drive file IDs)
VIT_MODEL_ID = "1FfrbjXwOx4ECahY5RhxHC3hUwl_oWwfm"
SWIN_MODEL_ID = "1_lfgSyOnP9SD-XIyK1Wd8SueHw42k_Ah"
SHELF_LIFE_MODEL_ID = "1yMcLUWjNDqrMkv-GA3JDGtbktq9hyXoc"

# Database configuration
MONGO_URI = "mongodb+srv://vivek:02Le7q3HeDgUFTCg@cluster0.7kjaiu5.mongodb.net/"
DB_NAME = "stale_fruit_db"

# Database file
DB_FILE = os.path.join(BASE_DIR, "users.db")
logger.info(f"Database file: {DB_FILE}")

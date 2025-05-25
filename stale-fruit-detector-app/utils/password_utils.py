import hashlib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hash_password(password):
    """Hash a password using SHA-256."""
    try:
        return hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        return None

def check_password(password, hashed_password):
    """Check if a password matches its hash."""
    try:
        return hash_password(password) == hashed_password
    except Exception as e:
        logger.error(f"Error checking password: {str(e)}")
        return False

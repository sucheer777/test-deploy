from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB connection URI
MONGO_URI = "mongodb+srv://chinna4812:chinna1234@my-first-cluster.m1waocd.mongodb.net/?retryWrites=true&w=majority&authSource=admin"

# Create MongoDB client and access the database
client = MongoClient(MONGO_URI)
db = client["stale_fruit_app"]

# Define collections for users and predictions
users_collection = db["users"]
predictions_collection = db["predictions"]

# Ensure the email field is unique
users_collection.create_index("email", unique=True)

def init_db():
    """Initialize the database with necessary indexes."""
    try:
        # Ensure indexes exist
        users_collection.create_index("email", unique=True)
        predictions_collection.create_index([("user_email", 1), ("timestamp", -1)])
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def add_user(email, password):
    """Add a new user to the database."""
    try:
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Try to insert the new user
        users_collection.insert_one({
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now()
        })
        logger.info(f"Successfully added new user: {email}")
        return True
    except DuplicateKeyError:
        logger.warning(f"User already exists: {email}")
        return False
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        return False

def verify_user(email, password):
    """Verify user credentials."""
    try:
        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        
        # Find user with matching email and password
        user = users_collection.find_one({
            "email": email,
            "password": hashed_password
        })
        
        if user:
            logger.info(f"Successfully verified user: {email}")
            return True
        else:
            logger.warning(f"Failed login attempt for user: {email}")
            return False
    except Exception as e:
        logger.error(f"Error verifying user: {str(e)}")
        return False

def save_prediction(user_email, result, confidence, fruit_type, shelf_life_condition=None, shelf_life_estimate=None, shelf_life_confidence=None):
    """Save prediction to database or update existing one"""
    try:
        # Check if a prediction exists within the last minute
        existing_prediction = predictions_collection.find_one({
            "user_email": user_email,
            "result": result,
            "fruit_type": fruit_type,
            "timestamp": {"$gt": datetime.now().replace(second=0, microsecond=0)}
        })
        
        if existing_prediction and (shelf_life_condition or shelf_life_estimate or shelf_life_confidence):
            # Update existing prediction with shelf life data
            predictions_collection.update_one(
                {"_id": existing_prediction["_id"]},
                {
                    "$set": {
                        "shelf_life_condition": shelf_life_condition,
                        "shelf_life_estimate": shelf_life_estimate,
                        "shelf_life_confidence": shelf_life_confidence
                    }
                }
            )
            logger.info(f"Updated existing prediction with shelf life data for user {user_email}")
        else:
            # Insert new prediction
            prediction_data = {
                "user_email": user_email,
                "result": result,
                "confidence": confidence,
                "fruit_type": fruit_type,
                "shelf_life_condition": shelf_life_condition,
                "shelf_life_estimate": shelf_life_estimate,
                "shelf_life_confidence": shelf_life_confidence,
                "timestamp": datetime.now()
            }
            predictions_collection.insert_one(prediction_data)
            logger.info(f"Inserted new prediction for user {user_email}")
            
    except Exception as e:
        logger.error(f"Error saving prediction: {str(e)}")
        raise

def get_predictions_by_user(user_email):
    """Get all predictions for a user"""
    try:
        predictions = list(predictions_collection.find(
            {"user_email": user_email},
            {
                "result": 1,
                "confidence": 1,
                "fruit_type": 1,
                "timestamp": 1,
                "shelf_life_condition": 1,
                "shelf_life_estimate": 1,
                "shelf_life_confidence": 1,
                "_id": 0
            }
        ).sort("timestamp", -1))
        
        # Convert predictions to the format expected by the history page
        formatted_predictions = []
        for pred in predictions:
            formatted_predictions.append((
                pred.get("result", ""),
                pred.get("confidence", 0.0),
                pred.get("fruit_type", "unknown"),
                pred.get("timestamp", datetime.now()),
                pred.get("shelf_life_condition"),
                pred.get("shelf_life_estimate"),
                pred.get("shelf_life_confidence")
            ))
        
        return formatted_predictions
    except Exception as e:
        logger.error(f"Error getting predictions: {str(e)}")
        raise

def find_user(email):
    """Find a user by email."""
    try:
        return users_collection.find_one({"email": email})
    except Exception as e:
        logger.error(f"Error finding user: {str(e)}")
        return None

# Initialize database when module is imported
init_db()

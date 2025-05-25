from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, ConnectionFailure
import hashlib
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None
    
    # MongoDB connection URI
    MONGO_URI = "mongodb+srv://chinna4812:chinna1234@my-first-cluster.m1waocd.mongodb.net/?retryWrites=true&w=majority&authSource=admin"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance.initialize_db()
        return cls._instance

    def initialize_db(self):
        """Initialize MongoDB connection and collections."""
        try:
            # Initialize MongoDB connection
            self.client = MongoClient(self.MONGO_URI, serverSelectionTimeoutMS=5000)
            
            # Test the connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Set up database and collections
            self.db = self.client["stale_fruit_app"]
            self.users = self.db["users"]
            self.predictions = self.db["predictions"]
            
            # Create indexes
            self.users.create_index("email", unique=True)
            self.predictions.create_index([("user_email", 1), ("timestamp", -1)])
            
            logger.info("Database initialized successfully")
        except ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {str(e)}")
            raise Exception("Failed to connect to MongoDB. Please check your connection and try again.")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    def add_user(self, email, password):
        """Add a new user to the database."""
        try:
            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            # Try to insert the new user
            self.users.insert_one({
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

    def verify_user(self, email, password):
        """Verify user credentials."""
        try:
            # Hash the password
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            # Find user with matching email and password
            user = self.users.find_one({
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

    def find_user(self, email):
        """Find a user by email."""
        try:
            return self.users.find_one({"email": email})
        except Exception as e:
            logger.error(f"Error finding user: {str(e)}")
            return None

    def save_prediction(self, user_email, result, confidence, fruit_type=None, 
                       shelf_life_condition=None, shelf_life_estimate=None, 
                       shelf_life_confidence=None):
        """Save prediction to database."""
        try:
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
            self.predictions.insert_one(prediction_data)
            logger.info(f"Successfully saved prediction for user: {user_email}")
            return True
        except Exception as e:
            logger.error(f"Error saving prediction: {str(e)}")
            return False

    def get_predictions_by_user(self, user_email):
        """Get all predictions for a user."""
        try:
            predictions = list(self.predictions.find(
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
            
            # Convert predictions to the format expected by the app
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
            return []

    def __del__(self):
        """Close the MongoDB connection when the object is destroyed."""
        try:
            if hasattr(self, 'client'):
                self.client.close()
                logger.info("MongoDB connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")

# Create a singleton instance
db = DatabaseManager() 
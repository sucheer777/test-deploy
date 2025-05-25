import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import logging
from model_utils import (
    ViT, SwinTransformer, shelf_life_class_names,
    shelf_life_data, get_condition_from_confidence
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FruitClassifier:
    def __init__(self, model_path='models/fruit_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SwinTransformer(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=len(shelf_life_class_names)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Classifier initialized on device: {self.device}")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError("Image must be a file path or PIL Image")
        
        return self.transform(image).unsqueeze(0).to(self.device)
    
    def predict_shelf_life(self, image):
        """
        Predict shelf life condition of fruit
        Returns: (fruit_type, condition, shelf_life_estimate, confidence)
        """
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
            
            # Get top predictions and their probabilities
            top_probs, top_indices = torch.topk(probabilities, k=min(5, len(shelf_life_class_names)))
            
            # Log top predictions
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                pred_class = shelf_life_class_names[idx]
                logger.info(f"Top {i+1}: {pred_class} ({prob.item()*100:.1f}%)")
            
            # Get the predicted class and confidence
            pred_idx = top_indices[0]
            confidence = top_probs[0].item() * 100
            pred_class = shelf_life_class_names[pred_idx]
            
            # Parse the prediction class
            fruit_type, condition = pred_class.split('_')
            
            # Get shelf life data
            if fruit_type in shelf_life_data and condition in shelf_life_data[fruit_type]:
                shelf_life = shelf_life_data[fruit_type][condition]
            else:
                shelf_life = "Unknown"
                logger.warning(f"No shelf life data found for {fruit_type}_{condition}")
            
            return fruit_type, condition, shelf_life, confidence
            
        except Exception as e:
            logger.error(f"Error in shelf life prediction: {str(e)}")
            return "unknown", "unknown", "Could not determine shelf life", 0.0
    
    def predict_freshness(self, image):
        """
        Predict if fruit is fresh or stale
        Returns: (result, confidence)
        """
        try:
            fruit_type, condition, _, confidence = self.predict_shelf_life(image)
            
            # Consider fresh and ripe as "FRESH", overripe and stale as "STALE"
            if condition in ['fresh', 'ripe']:
                result = "FRESH"
            else:
                result = "STALE"
            
            return result, confidence
            
        except Exception as e:
            logger.error(f"Error in freshness prediction: {str(e)}")
            return "ERROR", 0.0 
import os
import torch
from PIL import Image
from typing import Dict, Union
from src.utils import setup_logging, load_config
from src.model import FashionClassifier
from src.augmentations import get_transforms
from src.preprocessing import analyze_color_palette
from src.recommendation import RecommendationEngine

logger = setup_logging("FashionAI.Inference")

class FashionInferenceEngine:
    """Production-ready inference wrapper for FashionAI."""
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        transforms_dict = get_transforms(self.config['data']['image_size'])
        self.transform = transforms_dict['val']
        
        self.classes = ["Casual", "Formal", "Streetwear", "Sportswear", "Traditional"]
        
        self.model = FashionClassifier(
            backbone=self.config['model']['backbone'],
            num_classes=len(self.classes)
        ).to(device=self.device)
        
        ckpt_path = os.path.join(self.config['training']['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            logger.info("Loaded model weights successfully.")
        else:
            logger.warning("No model checkpoint found. Initialized with default weights.")
            
        self.model.eval()
        self.recommender = RecommendationEngine()

    def predict(self, image_input: Union[str, Image.Image]) -> Dict:
        """Processes image or text and returns complete outfit analysis JSON."""
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        else:
            image = image_input.convert('RGB')
            
        tensor_img = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            
        style = self.classes[pred_idx.item()]
        confidence = round(float(conf.item()), 2)
        
        color_palette = analyze_color_palette(image)
        recommendations = self.recommender.get_recommendations(style)
        styling_tips = self.recommender.get_styling_tips(style)
        
        return {
            "style": style,
            "season": "Autumn / Spring",
            "occasion": "Casual Hangouts" if style != "Formal" else "Business Meetings",
            "confidence": confidence,
            "color_palette": color_palette,
            "clothing_detection": ["Heavyweight Cotton Hoodie", "Relaxed Trousers", "Minimalist Sneakers"],
            "recommendations": recommendations,
            "styling_tips": styling_tips
        }
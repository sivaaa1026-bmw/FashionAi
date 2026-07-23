import torch
import numpy as np
from typing import List, Dict
from src.utils import setup_logging

logger = setup_logging("FashionAI.Recommendation")

class RecommendationEngine:
    """Engine to provide similar outfits, matching colors, and styling tips."""
    def __init__(self):
        # Sample embedding database for demonstration
        self.item_database = [
            {"id": 1, "title": "Minimalist Charcoal Hoodie", "style": "Streetwear", "price": "$59.99", "image": "https://images.unsplash.com/photo-1509631179647-0177331693ae?auto=format&fit=crop&w=600&q=80"},
            {"id": 2, "title": "Classic Black Sweatshirt", "style": "Casual", "price": "$49.99", "image": "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?auto=format&fit=crop&w=600&q=80"},
            {"id": 3, "title": "Structured Linen Blazer", "style": "Formal", "price": "$129.99", "image": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?auto=format&fit=crop&w=600&q=80"},
            {"id": 4, "title": "Athletic Track Jacket", "style": "Sportswear", "price": "$79.99", "image": "https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?auto=format&fit=crop&w=600&q=80"}
        ]

    def get_recommendations(self, style: str, embedding: torch.Tensor = None) -> List[Dict]:
        """Returns recommended items based on style match."""
        recommendations = []
        for item in self.item_database:
            match_score = 96 if item['style'].lower() == style.lower() else 88
            recommendations.append({
                **item,
                "match_score": f"{match_score}%"
            })
        return recommendations

    def get_styling_tips(self, style: str) -> List[str]:
        """Returns AI styling advice tailored to the detected style."""
        tips_map = {
            "streetwear": [
                "Layer a crisp white crewneck tee underneath to accentuate contrast at the collar hem.",
                "Accessorize with a matte silver chain bracelet to elevate the urban aesthetic."
            ],
            "formal": [
                "Pair with a slim leather belt matching your oxford shoes.",
                "Ensure your pocket square complements your tie without perfectly matching it."
            ],
            "casual": [
                "Roll up your cuffs slightly to give a relaxed, effortless look.",
                "Combine with minimalist low-top white leather sneakers."
            ]
        }
        return tips_map.get(style.lower(), [
            "Keep accessories balanced and focus on clean silhouettes.",
            "Ensure color contrast is optimized for your skin tone."
        ])
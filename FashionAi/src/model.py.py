import torch
import torch.nn as nn
import timm
from src.utils import setup_logging

logger = setup_logging("FashionAI.Model")

class FashionClassifier(nn.Module):
    """Modular Fashion Classification Model using timm backbones."""
    def __init__(self, backbone: str = 'efficientnet_b3', num_classes: int = 5, pretrained: bool = True, dropout_rate: int = 0.3):
        super(FashionClassifier, self).__init__()
        self.backbone_name = backbone
        logger.info(f"Initializing backbone: {backbone}, pretrained={pretrained}")
        
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='avg')
        in_features = self.encoder.num_features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate/2),
            nn.Linear(512, num_classes)
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts normalized embedding vectors for similarity search."""
        features = self.encoder(x)
        return torch.nn.functional.normalize(features, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits
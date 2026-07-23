import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
io = __import__('io')

from src.inference import FashionInferenceEngine
from src.utils import setup_logging

logger = setup_logging("FashionAI.API")

app = FastAPI(title="FashionAI API", version="1.0.0", description="Production-ready API for AI Fashion Stylist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = FashionInferenceEngine()

class TextPrompt(BaseModel):
    description: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "FashionAI API"}

@app.get("/metrics")
def metrics():
    return {"model": inference_engine.config['model']['backbone'], "status": "active"}

@app.post("/predict")
async def predict_outfit(file: Optional[UploadFile] = File(None), description: Optional[str] = Form(None)):
    if not file and not description:
        raise HTTPException(status_code=400, detail="Provide either an image file or a text description.")
        
    try:
        if file:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            result = inference_engine.predict(image)
        else:
            # Fallback mock/heuristic for text description
            result = {
                "style": "Streetwear" if "hoodie" in description.lower() else "Casual",
                "season": "All Season",
                "occasion": "Daily Wear",
                "confidence": 0.95,
                "color_palette": [{"hex": "#18181B", "rgb": "rgb(24, 24, 27)", "percentage": 60}],
                "clothing_detection": [description],
                "recommendations": inference_engine.recommender.get_recommendations("Streetwear"),
                "styling_tips": inference_engine.recommender.get_styling_tips("Streetwear")
            }
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend_outfits(style: str = Form("Streetwear")):
    recs = inference_engine.recommender.get_recommendations(style)
    return {"recommendations": recs}
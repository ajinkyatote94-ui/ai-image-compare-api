import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import requests
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import numpy as np

app = FastAPI()

# -------------------------------------------------
# LOAD CLIP MODEL (Best for image similarity)
# -------------------------------------------------
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.eval()

class ImageRequest(BaseModel):
    imageUrl1: str
    imageUrl2: str


# -------------------------------------------------
# GET IMAGE EMBEDDING
# -------------------------------------------------
def get_embedding(image_url: str):
    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    # Normalize to unit vector
    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

    return embedding[0].cpu().numpy()


# -------------------------------------------------
# COSINE SIMILARITY (since normalized → dot = cosine)
# -------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b))


# -------------------------------------------------
# IMAGE COMPARISON API
# -------------------------------------------------
@app.post("/compare-images/")
async def compare_images(data: ImageRequest):
    try:
        emb1 = get_embedding(data.imageUrl1)
        emb2 = get_embedding(data.imageUrl2)

        similarity = cosine_similarity(emb1, emb2)

        # -----------------------------
        # STRICT SIMILARITY CLEANING
        # -----------------------------

        # Remove negative similarities
        similarity = max(0.0, similarity)

        # HARD THRESHOLD
        if similarity < 0.20:
            image_points = 0.0
        else:
            # Square to penalize weak matches
            similarity = similarity ** 2

            # Scale to 30 points max
            image_points = similarity * 30

        # Final clamp
        image_points = min(image_points, 30.0)

        return {
            "similarity": round(similarity, 4),
            "imagePoints": round(image_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
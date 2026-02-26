import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np

app = FastAPI()

device = "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class ImageRequest(BaseModel):
    imageUrl1: str
    imageUrl2: str

def get_embedding(image_url):
    response = requests.get(image_url, timeout=10)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    return embedding[0].cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b)

@app.post("/compare-images/")
async def compare_images(data: ImageRequest):
    try:
        emb1 = get_embedding(data.imageUrl1)
        emb2 = get_embedding(data.imageUrl2)

        similarity = cosine_similarity(emb1, emb2)

        # Make scoring stricter
        if similarity < 0.3:
            image_points = 0
        else:
            image_points = float((similarity - 0.3) / 0.7 * 30)

        image_points = max(0, min(30, image_points))

        return {
            "similarity": float(similarity),
            "imagePoints": round(image_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
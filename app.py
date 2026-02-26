import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf

app = FastAPI()

# EfficientNetB0 (good balance for free tier)
model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

class ImageRequest(BaseModel):
    imageUrl1: str
    imageUrl2: str


# --------------------------------------------------
# Image Embedding
# --------------------------------------------------
def get_embedding(image_url):
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # Center crop square
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))

    img = img.resize((224, 224))

    img = np.array(img).astype("float32")
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    embedding = model.predict(img, verbose=0)[0]

    # Normalize safely
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------
def cosine_similarity(a, b):
    return float(np.dot(a, b))


# --------------------------------------------------
# Compare Endpoint
# --------------------------------------------------
@app.post("/compare-images/")
async def compare_images(data: ImageRequest):
    try:
        # ---------- Load and resize raw images ----------
        def load_image(url):
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            img = img.resize((224, 224))
            return np.array(img).astype("float32")

        img1 = load_image(data.imageUrl1)
        img2 = load_image(data.imageUrl2)

        # ---------- Quick pixel similarity check ----------
        pixel_diff = np.mean(np.abs(img1 - img2))

        if pixel_diff < 1:  
            # Almost identical image
            similarity = 1.0
        else:
            # ---------- EfficientNet embedding ----------
            def get_embedding(img_array):
                img = tf.keras.applications.efficientnet.preprocess_input(img_array)
                img = np.expand_dims(img, axis=0)
                emb = model.predict(img, verbose=0)[0]
                norm = np.linalg.norm(emb)
                return emb if norm == 0 else emb / norm

            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)

            similarity = float(np.dot(emb1, emb2))

        # ---------- Smooth scoring (MAX 25) ----------
        image_points = (similarity - 0.30) / 0.60 * 25
        image_points = float(np.clip(image_points, 0, 25))

        return {
            "similarity": round(similarity, 4),
            "imagePoints": round(image_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
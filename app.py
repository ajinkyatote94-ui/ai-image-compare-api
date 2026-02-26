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

# Lightweight & stable model for 512MB
model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

class ImageRequest(BaseModel):
    imageUrl1: str
    imageUrl2: str


# --------------------------------------------------
# Load and preprocess image
# --------------------------------------------------
def load_image(url):
    response = requests.get(url, timeout=10)
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
    return np.array(img).astype("float32")


# --------------------------------------------------
# Create embedding
# --------------------------------------------------
def get_embedding(img_array):
    img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img = np.expand_dims(img, axis=0)

    emb = model.predict(img, verbose=0)[0]
    norm = np.linalg.norm(emb)

    return emb if norm == 0 else emb / norm


# --------------------------------------------------
# Compare Endpoint
# --------------------------------------------------
@app.post("/compare-images/")
async def compare_images(data: ImageRequest):
    try:
        # Load images
        img1 = load_image(data.imageUrl1)
        img2 = load_image(data.imageUrl2)

        # Quick pixel-level identical check
        pixel_diff = np.mean(np.abs(img1 - img2))

        if pixel_diff < 1.0:
            similarity = 1.0
        else:
            emb1 = get_embedding(img1)
            emb2 = get_embedding(img2)
            similarity = float(np.dot(emb1, emb2))

        # --------------------------------------------------
        # Boosted smooth scoring (MAX 25)
        # --------------------------------------------------
        image_points = (similarity ** 0.7) * 25
        image_points = float(np.clip(image_points, 0, 25))

        return {
            "similarity": round(similarity, 4),
            "imagePoints": round(image_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
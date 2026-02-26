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

# 🔥 Better than MobileNetV2
model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

class ImageRequest(BaseModel):
    imageUrl1: str
    imageUrl2: str

def get_embedding(image_url):
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    embedding = model.predict(img, verbose=0)
    embedding = embedding / np.linalg.norm(embedding)

    return embedding[0]

def cosine_similarity(a, b):
    return np.dot(a, b)

@app.post("/compare-images/")
async def compare_images(data: ImageRequest):
    try:
        emb1 = get_embedding(data.imageUrl1)
        emb2 = get_embedding(data.imageUrl2)

        similarity = cosine_similarity(emb1, emb2)

        # 🔥 Improved scoring curve
        if similarity < 0.4:
            image_points = 0
        else:
            image_points = ((similarity - 0.4) / 0.6) * 30

        image_points = float(np.clip(image_points, 0, 30))

        return {
            "similarity": float(similarity),
            "imagePoints": round(image_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
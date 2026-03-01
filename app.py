import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
from sentence_transformers import SentenceTransformer

app = FastAPI()

# --------------------------------------------------
# LOAD MODELS (ONCE AT STARTUP)
# --------------------------------------------------

# Image Model
image_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Text Model
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# Warm-up image model
dummy = np.zeros((1, 224, 224, 3), dtype="float32")
image_model.predict(dummy, verbose=0)

# --------------------------------------------------
# REQUEST MODEL
# --------------------------------------------------

class MatchRequest(BaseModel):
    title1: str
    description1: str
    title2: str
    description2: str
    imageUrls1: list
    imageUrls2: list

# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def load_image(url):
    response = requests.get(url, timeout=8)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")

    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    img = img.crop((left, top, right, bottom))

    img = img.resize((224, 224))

    return np.array(img).astype("float32")

def get_image_embedding(img_array):
    img = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img = np.expand_dims(img, axis=0)

    emb = image_model.predict(img, verbose=0)[0]
    norm = np.linalg.norm(emb)

    return emb if norm == 0 else emb / norm

def get_best_image_similarity(images1, images2):
    best_similarity = 0

    for url1 in images1:
        for url2 in images2:
            try:
                img1 = load_image(url1)
                img2 = load_image(url2)

                emb1 = get_image_embedding(img1)
                emb2 = get_image_embedding(img2)

                similarity = float(np.dot(emb1, emb2))

                if similarity > best_similarity:
                    best_similarity = similarity

            except:
                continue

    return best_similarity

# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

def get_text_similarity(title1, desc1, title2, desc2):
    combined1 = f"{title1}. {desc1}"
    combined2 = f"{title2}. {desc2}"

    embeddings = text_model.encode(
        [combined1, combined2],
        normalize_embeddings=True
    )

    similarity = float(np.dot(embeddings[0], embeddings[1]))
    return similarity

# --------------------------------------------------
# MAIN MATCH ENDPOINT
# --------------------------------------------------

@app.post("/compare-match/")
async def compare_match(data: MatchRequest):
    try:
        # 1️⃣ TEXT SIMILARITY
        text_similarity = get_text_similarity(
            data.title1,
            data.description1,
            data.title2,
            data.description2
        )

        text_points = float(np.clip(text_similarity * 30, 0, 30))

        # 2️⃣ IMAGE SIMILARITY (ONLY IF TEXT IS STRONG ENOUGH)
        image_points = 0
        image_similarity = 0

        if text_similarity > 0.35 and data.imageUrls1 and data.imageUrls2:
            image_similarity = get_best_image_similarity(
                data.imageUrls1,
                data.imageUrls2
            )

            image_points = float(
                np.clip((image_similarity ** 0.65) * 30, 0, 30)
            )

        total_points = text_points + image_points

        return {
            "textSimilarity": round(text_similarity, 4),
            "imageSimilarity": round(image_similarity, 4),
            "textPoints": round(text_points, 2),
            "imagePoints": round(image_points, 2),
            "totalAIpoints": round(total_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
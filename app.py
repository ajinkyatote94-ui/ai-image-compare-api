import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import imagehash
from sentence_transformers import SentenceTransformer

app = FastAPI()

# -------------------------------------------
# LOAD TEXT MODEL (Free-tier safe)
# -------------------------------------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------
# REQUEST MODEL
# -------------------------------------------
class MatchRequest(BaseModel):
    title1: str
    description1: str
    title2: str
    description2: str
    imageUrls1: list = []
    imageUrls2: list = []

# -------------------------------------------
# TEXT SIMILARITY
# -------------------------------------------
def text_similarity(text1, text2):
    embeddings = text_model.encode(
        [text1, text2],
        normalize_embeddings=True
    )
    return float(np.dot(embeddings[0], embeddings[1]))

# -------------------------------------------
# IMAGE HASH SIMILARITY (All vs All)
# -------------------------------------------
def get_image_hash(url):
    response = requests.get(url, timeout=6)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return imagehash.phash(img)

def get_best_image_similarity(images1, images2):
    best_score = 0

    for url1 in images1:
        for url2 in images2:
            try:
                hash1 = get_image_hash(url1)
                hash2 = get_image_hash(url2)

                distance = hash1 - hash2
                similarity = 1 - (distance / 64)  # normalize 0–1

                if similarity > best_score:
                    best_score = similarity

            except:
                continue

    return max(0, best_score)

# -------------------------------------------
# MAIN ENDPOINT
# -------------------------------------------
@app.post("/compare-match/")
async def compare_match(data: MatchRequest):
    try:

        # 1️⃣ TITLE (Max 15)
        title_sim = text_similarity(data.title1, data.title2)
        title_points = float(np.clip(title_sim * 15, 0, 15))

        # 2️⃣ DESCRIPTION (Max 15)
        desc_sim = text_similarity(
            data.description1,
            data.description2
        )
        description_points = float(np.clip(desc_sim * 15, 0, 15))

        # 3️⃣ IMAGES (Max 20)
        image_sim = 0
        image_points = 0

        if data.imageUrls1 and data.imageUrls2:
            image_sim = get_best_image_similarity(
                data.imageUrls1,
                data.imageUrls2
            )
            image_points = float(np.clip(image_sim * 20, 0, 20))

        total_points = title_points + description_points + image_points

        return {
            "titleSimilarity": round(title_sim, 4),
            "descriptionSimilarity": round(desc_sim, 4),
            "imageSimilarity": round(image_sim, 4),

            "titlePoints": round(title_points, 2),
            "descriptionPoints": round(description_points, 2),
            "imagePoints": round(image_points, 2),

            "totalAIpoints": round(total_points, 2)
        }

    except Exception as e:
        return {"error": str(e)}
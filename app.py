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
# LOAD TEXT MODEL (LIGHT + FAST)
# -------------------------------------------
text_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------------------
# REQUEST MODEL
# -------------------------------------------
class MatchRequest(BaseModel):
    title1: str = ""
    description1: str = ""
    title2: str = ""
    description2: str = ""
    imageUrls1: list = []
    imageUrls2: list = []

# -------------------------------------------
# SAFE TEXT SIMILARITY
# Converts cosine [-1,1] → [0,1]
# -------------------------------------------
def text_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return 0.0

    embeddings = text_model.encode(
        [text1, text2],
        normalize_embeddings=True
    )

    cosine = float(np.dot(embeddings[0], embeddings[1]))

    # Convert -1→1 range into 0→1 range
    normalized_score = (cosine + 1) / 2

    return max(0.0, min(1.0, normalized_score))

# -------------------------------------------
# SAFE IMAGE HASH
# -------------------------------------------
def get_image_hash(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)

        if response.status_code != 200:
            return None

        img = Image.open(BytesIO(response.content)).convert("RGB")

        return imagehash.phash(img)

    except Exception:
        return None

# -------------------------------------------
# IMPROVED IMAGE SIMILARITY
# Less strict than before
# -------------------------------------------
def get_best_image_similarity(images1, images2):
    best_score = 0.0

    for url1 in images1:
        for url2 in images2:

            hash1 = get_image_hash(url1)
            hash2 = get_image_hash(url2)

            if hash1 is None or hash2 is None:
                continue

            distance = hash1 - hash2

            # More tolerant scaling
            similarity = max(0.0, 1 - (distance / 32))

            best_score = max(best_score, similarity)

    return best_score

# -------------------------------------------
# MAIN ENDPOINT
# -------------------------------------------
@app.post("/compare-match/")
async def compare_match(data: MatchRequest):

    try:

        # ---------------------------
        # 1️⃣ TITLE (MAX 15)
        # ---------------------------
        title_sim = text_similarity(
            data.title1 or "",
            data.title2 or ""
        )

        title_points = float(np.clip(title_sim * 15, 0, 15))

        # ---------------------------
        # 2️⃣ DESCRIPTION (MAX 15)
        # ---------------------------
        desc_sim = text_similarity(
            data.description1 or "",
            data.description2 or ""
        )

        description_points = float(np.clip(desc_sim * 15, 0, 15))

        # ---------------------------
        # 3️⃣ IMAGES (MAX 25)
        # ---------------------------
        image_sim = 0.0
        image_points = 0.0

        if data.imageUrls1 and data.imageUrls2:
            image_sim = get_best_image_similarity(
                data.imageUrls1,
                data.imageUrls2
            )

            image_points = float(np.clip(image_sim * 25, 0, 25))

        # ---------------------------
        # TOTAL AI SCORE (MAX 55)
        # ---------------------------
        total_points = (
            title_points +
            description_points +
            image_points
        )

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
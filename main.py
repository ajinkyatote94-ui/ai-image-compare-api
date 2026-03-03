# main.py

from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
import os
from ai_model import compute_image_similarity

app = FastAPI()


@app.get("/")
def root():
    return {"status": "AI server running"}


@app.post("/compare-images/")
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    file1_path = f"temp_{uuid.uuid4()}.jpg"
    file2_path = f"temp_{uuid.uuid4()}.jpg"

    try:
        # Save temp images
        with open(file1_path, "wb") as buffer:
            shutil.copyfileobj(image1.file, buffer)

        with open(file2_path, "wb") as buffer:
            shutil.copyfileobj(image2.file, buffer)

        # Compute similarity
        similarity = compute_image_similarity(file1_path, file2_path)

        return {
            "imageSimilarity": similarity,
            "imagePoints": round(similarity * 25, 2)
        }

    finally:
        # Clean up temp files
        if os.path.exists(file1_path):
            os.remove(file1_path)
        if os.path.exists(file2_path):
            os.remove(file2_path)
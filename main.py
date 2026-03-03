from fastapi import FastAPI, UploadFile, File
import shutil
import uuid
from app import compute_image_similarity

app = FastAPI()


@app.post("/compare-images/")
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    file1_path = f"temp_{uuid.uuid4()}.jpg"
    file2_path = f"temp_{uuid.uuid4()}.jpg"

    with open(file1_path, "wb") as buffer:
        shutil.copyfileobj(image1.file, buffer)

    with open(file2_path, "wb") as buffer:
        shutil.copyfileobj(image2.file, buffer)

    similarity = compute_image_similarity(file1_path, file2_path)

    return {
        "imageSimilarity": similarity,
        "imagePoints": round(similarity * 25, 2)
    }
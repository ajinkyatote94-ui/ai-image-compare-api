from fastapi import FastAPI, UploadFile, File, Form
import shutil
import uuid
import os

from ai_model import compute_match

app = FastAPI()


@app.get("/")
def root():
    return {"status": "AI running"}


@app.post("/compare-match/")
async def compare_match(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    title1: str = Form(""),
    title2: str = Form(""),
    description1: str = Form(""),
    description2: str = Form("")
):

    file1 = f"/tmp/temp_{uuid.uuid4()}.jpg"
    file2 = f"/tmp/temp_{uuid.uuid4()}.jpg"

    try:
        # Save uploaded images
        with open(file1, "wb") as f:
            shutil.copyfileobj(image1.file, f)

        with open(file2, "wb") as f:
            shutil.copyfileobj(image2.file, f)

        # Run AI match
        result = compute_match(
            file1,
            file2,
            title1,
            title2,
            description1,
            description2
        )

        return result

    finally:
        # Clean temporary files
        if os.path.exists(file1):
            os.remove(file1)

        if os.path.exists(file2):
            os.remove(file2)
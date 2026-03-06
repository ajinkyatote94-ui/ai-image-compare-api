from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import shutil
import uuid
import os

from ai_model import compute_match

app = FastAPI()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def root():
    return {"status": "AI running"}


# =========================
# MATCH API
# =========================
@app.post("/compare-match/")
async def compare_match(

    images1: List[UploadFile] = File(...),
    images2: List[UploadFile] = File(...),

    title1: str = Form(""),
    title2: str = Form(""),

    description1: str = Form(""),
    description2: str = Form(""),

    category1: str = Form(""),
    category2: str = Form(""),

    lat1: float = Form(...),
    lon1: float = Form(...),

    lat2: float = Form(...),
    lon2: float = Form(...)
):

    paths1 = []
    paths2 = []

    try:

        # limit images
        if len(images1) > 5 or len(images2) > 5:
            return {"error": "Maximum 5 images allowed"}

        # save item1 images
        for img in images1:
            path = f"/tmp/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
                shutil.copyfileobj(img.file, f)

            paths1.append(path)

            await img.close()

        # save item2 images
        for img in images2:
            path = f"/tmp/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
             content = await img.read()

             if len(content) > 5 * 1024 * 1024:  # 5MB limit
                  return {"error": "Image too large"}

            f.write(content)

            paths2.append(path)

            await img.close()

        # run AI
        result = compute_match(
            paths1,
            paths2,
            title1,
            title2,
            description1,
            description2,
            category1,
            category2,
            lat1,
            lon1,
            lat2,
            lon2
        )

        return result

    finally:

        # cleanup
        for p in paths1:
            if os.path.exists(p):
                os.remove(p)

        for p in paths2:
            if os.path.exists(p):
                os.remove(p)
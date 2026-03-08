from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ai_model import compute_match


# ================================
# APP
# ================================

app = FastAPI()


# ================================
# CORS
# ================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================
# TEMP FOLDER
# ================================

TMP_DIR = "/tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# ================================
# ROOT
# ================================

@app.get("/")
def root():
    return {"status": "AI running"}


# ================================
# MATCH API
# ================================

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

        # ================================
        # LIMIT IMAGE COUNT
        # ================================

        if len(images1) > 5 or len(images2) > 5:
            return {"error": "Maximum 5 images allowed"}

        # ================================
        # SAVE ITEM 1 IMAGES
        # ================================

        for img in images1:

            content = await img.read()

            if len(content) > 5 * 1024 * 1024:
                return {"error": "Image too large"}

            path = f"{TMP_DIR}/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
                f.write(content)

            # Validate image
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                return {"error": "Invalid image"}

            paths1.append(path)

            await img.close()

        # ================================
        # SAVE ITEM 2 IMAGES
        # ================================

        for img in images2:

            content = await img.read()

            if len(content) > 5 * 1024 * 1024:
                return {"error": "Image too large"}

            path = f"{TMP_DIR}/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
                f.write(content)

            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception:
                return {"error": "Invalid image"}

            paths2.append(path)

            await img.close()

        # ================================
        # RUN AI MATCH
        # ================================

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

    except Exception as e:

        print("AI ERROR:", e)

        return {"error": str(e)}

    finally:

        # ================================
        # CLEAN TEMP FILES
        # ================================

        for p in paths1:
            if os.path.exists(p):
                os.remove(p)

        for p in paths2:
            if os.path.exists(p):
                os.remove(p)
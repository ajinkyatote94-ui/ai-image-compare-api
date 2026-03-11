from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import os
from PIL import Image

from ai_model import compute_match


app = FastAPI()


# ============================
# CORS (Allow Flutter requests)
# ============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
# TEMP DIRECTORY (Render safe)
# ============================

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)


# ============================
# ROOT
# ============================

@app.get("/")
def root():
    return {"status": "FindIt AI running"}


# ============================
# IMAGE SAVE + VALIDATION
# ============================

async def save_upload_image(img: UploadFile):

    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
        "image/gif",
        "application/octet-stream"
    ]

    if img.content_type not in allowed_types:
        raise Exception("Unsupported image format")

    content = await img.read()

    if len(content) == 0:
        raise Exception("Empty image")

    if len(content) > 5 * 1024 * 1024:
        raise Exception("Image too large (max 5MB)")

    path = f"{TEMP_DIR}/{uuid.uuid4()}.jpg"

    with open(path, "wb") as f:
        f.write(content)

    # verify and normalize image
    try:
        with Image.open(path) as im:

            im.verify()

        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((512, 512))
            im.save(path, "JPEG")

    except:
        os.remove(path)
        raise Exception("Invalid image")

    await img.close()

    return path


# ============================
# MATCH API
# ============================

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
    lon2: float = Form(...),
):

    paths1 = []
    paths2 = []

    try:

        # limit images
        if len(images1) > 2 or len(images2) > 2:
            return {
                "success": False,
                "error": "Maximum 2 images per item"
            }

        # save images set 1
        for img in images1:
            path = await save_upload_image(img)
            paths1.append(path)

        # save images set 2
        for img in images2:
            path = await save_upload_image(img)
            paths2.append(path)

        # run AI model
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

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }

    finally:

        # cleanup temp files
        for p in paths1:
            if os.path.exists(p):
                os.remove(p)

        for p in paths2:
            if os.path.exists(p):
                os.remove(p)
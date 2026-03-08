from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from ai_model import compute_match

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


# temp folder
os.makedirs("/tmp", exist_ok=True)# ================================
# ROOT
# ================================
@app.get("/")
def root():
    return {"status": "AI running"}# ================================
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
    lon2: float = Form(...),
):

    paths1 = []
    paths2 = []

    try:

        # limit images (keep small for Render RAM)
        images1 = images1[:2]
        images2 = images2[:2]        # ================================
        # SAVE IMAGES 1
        # ================================
        for img in images1:

            content = await img.read()

            if len(content) > 5 * 1024 * 1024:
                return {"error": "Image too large"}

            path = f"/tmp/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
                f.write(content)

            # verify image
            try:
                Image.open(path).verify()
            except:
                return {"error": "Invalid image"}

            paths1.append(path)
            await img.close()

        # ================================
        # SAVE IMAGES 2
        # ================================
        for img in images2:

            content = await img.read()

            if len(content) > 5 * 1024 * 1024:
                return {"error": "Image too large"}

            path = f"/tmp/{uuid.uuid4()}.jpg"

            with open(path, "wb") as f:
                f.write(content)

            try:
                Image.open(path).verify()
            except:
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
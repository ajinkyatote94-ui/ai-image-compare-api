from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import uuid
import os
from fastapi.middleware.cors import CORSMiddleware

from ai_model import compute_match

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("/tmp", exist_ok=True)


@app.get("/")
def root():
    return {"status": "AI running"}


@app.post("/compare-match/")
async def compare_match(

    images1: UploadFile = File(...),
    images2: UploadFile = File(...),

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

        # -------- save image1 --------
        if not images1.content_type.startswith("image/"):
            return {"error": "Only image files allowed"}

        content1 = await images1.read()

        if len(content1) > 5 * 1024 * 1024:
            return {"error": "Image too large"}

        path1 = f"/tmp/{uuid.uuid4()}.jpg"

        with open(path1, "wb") as f:
            f.write(content1)

        paths1.append(path1)

        await images1.close()

        # -------- save image2 --------
        if not images2.content_type.startswith("image/"):
            return {"error": "Only image files allowed"}

        content2 = await images2.read()

        if len(content2) > 5 * 1024 * 1024:
            return {"error": "Image too large"}

        path2 = f"/tmp/{uuid.uuid4()}.jpg"

        with open(path2, "wb") as f:
            f.write(content2)

        paths2.append(path2)

        await images2.close()
        print("TITLE1:", title1)
        print("TITLE2:", title2)
        print("DESC1:", description1)
        print("DESC2:", description2)
        print("CATEGORY1:", category1)
        print("CATEGORY2:", category2)
        print("LAT1:", lat1, "LON1:", lon1)
        print("LAT2:", lat2, "LON2:", lon2)
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

        for p in paths1:
            if os.path.exists(p):
                os.remove(p)

        for p in paths2:
            if os.path.exists(p):
                os.remove(p)
from PIL import Image
import numpy as np
import math
import torch
import torchvision.transforms as T

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD DINOv2 MODEL (ONCE)
# =========================

model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

model.eval()

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
])


# =========================
# IMAGE EMBEDDING
# =========================

def get_embedding(path):

    img = Image.open(path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        emb = model(img)

    return emb.numpy()


# =========================
# IMAGE SIMILARITY
# =========================

def compute_image_similarity(imagesA, imagesB):

    if not imagesA or not imagesB:
        return 0

    best = 0

    for a in imagesA:
        for b in imagesB:

            try:
                emb1 = get_embedding(a)
                emb2 = get_embedding(b)

                sim = cosine_similarity(emb1, emb2)[0][0]

                if sim > best:
                    best = sim

            except Exception:
                continue

    return best


# =========================
# TEXT SIMILARITY
# =========================

def text_similarity(t1, t2):

    t1 = (t1 or "").strip().lower()
    t2 = (t2 or "").strip().lower()

    if not t1 and not t2:
        return 0

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2,4)
    )

    tfidf = vectorizer.fit_transform([t1, t2])

    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return sim


# =========================
# CATEGORY SCORE
# =========================

def category_score(cat1, cat2):

    cat1 = (cat1 or "").lower()
    cat2 = (cat2 or "").lower()

    if cat1 == cat2:
        return 50

    return 0


# =========================
# LOCATION DISTANCE
# =========================

def distance(lat1, lon1, lat2, lon2):

    R = 6371

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)

    a = (
        math.sin(dlat/2)**2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon/2)**2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R*c


def location_score(lat1, lon1, lat2, lon2):

    d = distance(lat1, lon1, lat2, lon2)

    if d < 1:
        return 50
    if d < 5:
        return 40
    if d < 10:
        return 30
    if d < 20:
        return 20

    return 10


# =========================
# FINAL MATCH
# =========================

def compute_match(
    imagesA,
    imagesB,
    title1,
    title2,
    desc1,
    desc2,
    category1,
    category2,
    lat1,
    lon1,
    lat2,
    lon2
):

    lat1 = float(lat1)
    lon1 = float(lon1)
    lat2 = float(lat2)
    lon2 = float(lon2)

    img_sim = compute_image_similarity(imagesA, imagesB)

    title_sim = text_similarity(title1, title2)
    desc_sim = text_similarity(desc1, desc2)

    image_points = round(img_sim * 50, 2)
    title_points = round(title_sim * 50, 2)
    desc_points = round(desc_sim * 50, 2)

    cat_points = category_score(category1, category2)

    loc_points = location_score(
        lat1,
        lon1,
        lat2,
        lon2
    )

    total = (
        image_points +
        title_points +
        desc_points +
        cat_points +
        loc_points
    )

    return {
        "image": image_points,
        "title": title_points,
        "description": desc_points,
        "category": cat_points,
        "location": loc_points,
        "total": total
    }
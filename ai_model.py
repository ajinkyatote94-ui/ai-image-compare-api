from PIL import Image
import numpy as np
import math
import torch
import torchvision.transforms as T

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# LOAD MODEL
# =========================

device = "cpu"

weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)

# remove classifier (we only want embeddings)
model.classifier = torch.nn.Identity()

model = model.to(device)
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
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(img)

    emb = emb.cpu().numpy()

    # normalize
    emb = emb / np.linalg.norm(emb)

    return emb


# =========================
# IMAGE SIMILARITY
# =========================

def compute_image_similarity(imagesA, imagesB):

    if not imagesA or not imagesB:
        return 0

    embA = [get_embedding(a) for a in imagesA]
    embB = [get_embedding(b) for b in imagesB]

    best = 0

    for e1 in embA:
        for e2 in embB:

            sim = cosine_similarity(
                e1.reshape(1,-1),
                e2.reshape(1,-1)
            )[0][0]

            if sim > best:
                best = sim

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

    sim = cosine_similarity(
        tfidf[0:1],
        tfidf[1:2]
    )[0][0]

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
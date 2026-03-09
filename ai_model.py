from PIL import Image
import numpy as np
import math
import imagehash

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# IMAGE SIMILARITY (LIGHTWEIGHT)
# =========================



def compute_image_similarity(imagesA, imagesB):

    if not imagesA or not imagesB:
        return 0

    best = 0

    for a in imagesA:
        for b in imagesB:

            try:
                with Image.open(a) as img1, Image.open(b) as img2:

                    img1 = img1.resize((256,256)).convert("RGB")
                    img2 = img2.resize((256,256)).convert("RGB")

                    # pHash similarity
                    hash1 = imagehash.phash(img1, hash_size=16)
                    hash2 = imagehash.phash(img2, hash_size=16)

                    phash_sim = 1 - ((hash1 - hash2) / 256)

                    # color histogram similarity
                    hist1 = np.array(img1.histogram())
                    hist2 = np.array(img2.histogram())

                    hist1 = hist1 / (hist1.sum() + 1e-10)
                    hist2 = hist2 / (hist2.sum() + 1e-10)
                    hist_sim = np.dot(hist1, hist2)

                    # combine scores
                    sim = (0.7 * phash_sim) + (0.3 * hist_sim)
                    sim = max(0, min(sim, 1))
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
# CATEGORY DETECTION
# =========================

def detect_category(text):

    text = text.lower()

    if "phone" in text or "laptop" in text:
        return "electronics"

    if "wallet" in text or "purse" in text:
        return "wallet"

    if "key" in text:
        return "keys"

    if "shirt" in text or "jacket" in text:
        return "clothing"

    if "watch" in text or "glasses" in text:
        return "accessories"

    if "bag" in text or "backpack" in text:
        return "bags"

    if "ring" in text or "necklace" in text:
        return "jewelry"

    if "passport" in text or "license" in text:
        return "documents"

    if "football" in text or "cricket" in text:
        return "sports"

    return "other"


def category_score(cat1, cat2, title1, title2, desc1, desc2):

    cat1 = (cat1 or "").lower()
    cat2 = (cat2 or "").lower()

    if cat1 != "other" and cat2 != "other":
        return 50 if cat1 == cat2 else 0

    text1 = (title1 + " " + desc1).lower()
    text2 = (title2 + " " + desc2).lower()

    detected1 = detect_category(text1)
    detected2 = detect_category(text2)

    if detected1 == detected2:
        return 50

    return 0# =========================
# LOCATION DISTANCE
# =========================

def distance(lat1,lon1,lat2,lon2):

    R = 6371

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)

    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))

    return R*c


def location_score(lat1,lon1,lat2,lon2):

    d = distance(lat1,lon1,lat2,lon2)

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

    img_sim = compute_image_similarity(imagesA,imagesB)

    title_sim = text_similarity(title1,title2)
    desc_sim = text_similarity(desc1,desc2)

    image_points = round(img_sim*50,2)
    title_points = round(title_sim*50,2)
    desc_points = round(desc_sim*50,2)

    cat_points = category_score(
        category1,
        category2,
        title1,
        title2,
        desc1,
        desc2
    )

    loc_points = location_score(
        lat1,
        lon1,
        lat2,
        lon2
    )

    total = image_points + title_points + desc_points + cat_points + loc_points

    return {
        "image": image_points,
        "title": title_points,
        "description": desc_points,
        "category": cat_points,
        "location": loc_points,
        "total": total
    }
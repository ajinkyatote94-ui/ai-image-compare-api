import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = "cpu"

# =========================
# IMAGE MODEL
# =========================

weights = models.MobileNet_V3_Small_Weights.DEFAULT
image_model = models.mobilenet_v3_small(weights=weights)
image_model.classifier = torch.nn.Identity()
image_model.eval()
image_model.to(device)

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================
# IMAGE EMBEDDING
# =========================

def get_embedding(path):

    try:
        image = Image.open(path).convert("RGB")
    except:
        return None

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = image_model(image)

    emb = F.normalize(emb,dim=1)

    return emb


# =========================
# IMAGE SIMILARITY (MULTI IMAGE)
# =========================

def compute_image_similarity(imagesA,imagesB):

    if not imagesA or not imagesB:
        return 0
    best = 0

    embeddingsA = [e for e in [get_embedding(p) for p in imagesA] if e is not None]
    embeddingsB = [e for e in [get_embedding(p) for p in imagesB] if e is not None]

    if not embeddingsA or not embeddingsB:
       return 0
    for a in embeddingsA:
        for b in embeddingsB:

            sim = F.cosine_similarity(a,b).item()
            sim = (sim+1)/2

            if sim > best:
                best = sim

    return best


# =========================
# TEXT SIMILARITY
# =========================


def text_similarity(t1, t2):

    if not t1 or not t2:
        return 0

    vectorizer = TfidfVectorizer()

    tfidf = vectorizer.fit_transform([t1, t2])

    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    return sim

# =========================
# CATEGORY AI
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

    # If both categories are proper
    if cat1 != "other" and cat2 != "other":
        return 50 if cat1 == cat2 else 0

    # If any category is "other" → detect using text
    text1 = (title1 + " " + desc1).lower()
    text2 = (title2 + " " + desc2).lower()

    detected1 = detect_category(text1)
    detected2 = detect_category(text2)

    if detected1 == detected2:
        return 50

    return 0
# =========================
# LOCATION
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

   

    return {
        "image": image_points,
        "title": title_points,
        "description": desc_points,
        "category": cat_points,
        "location": loc_points
    }
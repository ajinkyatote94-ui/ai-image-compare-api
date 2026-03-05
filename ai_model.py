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

image_model = models.mobilenet_v3_small(weights="DEFAULT")
image_model.classifier = torch.nn.Identity()
image_model.eval()
image_model.to(device)

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# =========================
# IMAGE EMBEDDING
# =========================

def get_embedding(path):

    image = Image.open(path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = image_model(image)

    emb = F.normalize(emb,dim=1)

    return emb


# =========================
# IMAGE SIMILARITY (MULTI IMAGE)
# =========================

def compute_image_similarity(imagesA,imagesB):

    best = 0

    embeddingsA = [get_embedding(p) for p in imagesA]
    embeddingsB = [get_embedding(p) for p in imagesB]

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

    texts = [t1, t2]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

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


def category_score(cat1,cat2,title1,title2,desc1,desc2):

    if cat1 != "other" and cat2 != "other":

        if cat1 == cat2:
            return 50

        return 0

    text1 = title1 + " " + desc1
    text2 = title2 + " " + desc2

    c1 = detect_category(text1)
    c2 = detect_category(text2)

    if c1 == c2:
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

    total = round(
        image_points +
        title_points +
        desc_points +
        cat_points +
        loc_points
    ,2)

    return {
        "imagePoints":image_points,
        "titlePoints":title_points,
        "descriptionPoints":desc_points,
        "categoryPoints":cat_points,
        "locationPoints":loc_points,
        "totalAIpoints":total
    }
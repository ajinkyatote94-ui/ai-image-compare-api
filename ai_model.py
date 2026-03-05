import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# DEVICE
# =========================

device = "cpu"
torch.set_grad_enabled(False)

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
# CENTER CROP
# =========================

def center_crop(image):

    w,h = image.size
    size = int(min(w,h)*0.8)

    left = (w-size)//2
    top = (h-size)//2

    right = left+size
    bottom = top+size

    return image.crop((left,top,right,bottom))


# =========================
# IMAGE EMBEDDING
# =========================

def get_image_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    image = center_crop(image)

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = image_model(image)

    emb = F.normalize(emb,dim=1)

    return emb


# =========================
# IMAGE SIMILARITY
# =========================

def compute_image_similarity(img1,img2):

    emb1 = get_image_embedding(img1)
    emb2 = get_image_embedding(img2)

    sim = F.cosine_similarity(emb1,emb2).item()

    sim = (sim+1)/2

    return sim


# =========================
# TEXT SIMILARITY (LIGHTWEIGHT)
# =========================

vectorizer = TfidfVectorizer()

def compute_text_similarity(text1,text2):

    if not text1 or not text2:
        return 0

    vectors = vectorizer.fit_transform([text1,text2])

    sim = cosine_similarity(vectors[0:1],vectors[1:2])[0][0]

    return float(sim)


# =========================
# FULL MATCH CALCULATION
# =========================

def compute_match(
    img1,
    img2,
    title1,
    title2,
    desc1,
    desc2
):

    image_sim = compute_image_similarity(img1,img2)

    title_sim = compute_text_similarity(title1,title2)

    desc_sim = compute_text_similarity(desc1,desc2)

    image_points = round(image_sim*50,2)
    title_points = round(title_sim*50,2)
    desc_points = round(desc_sim*50,2)

    total_ai = round(image_points+title_points+desc_points,2)

    return {
        "imageSimilarity":round(image_sim,4),
        "titleSimilarity":round(title_sim,4),
        "descriptionSimilarity":round(desc_sim,4),

        "imagePoints":image_points,
        "titlePoints":title_points,
        "descriptionPoints":desc_points,

        "totalAIpoints":total_ai
    }
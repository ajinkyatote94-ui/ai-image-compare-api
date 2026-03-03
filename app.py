import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# LOAD MODELS (ON STARTUP)
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO small model (lightweight)
yolo_model = YOLO("yolov8n.pt")

# DINOv2 base model
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
dinov2_model.eval()


# =========================
# OBJECT DETECTION + CROP
# =========================

def detect_and_crop(image_path):
    image = Image.open(image_path).convert("RGB")

    results = yolo_model(image_path)

    if len(results[0].boxes) == 0:
        # No object detected → return full image
        return image

    # Take highest confidence detection
    boxes = results[0].boxes
    confidences = boxes.conf.cpu().numpy()
    best_index = np.argmax(confidences)

    box = boxes.xyxy[best_index].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box

    cropped = image.crop((x1, y1, x2, y2))
    return cropped


# =========================
# EXTRACT DINO EMBEDDING
# =========================

def extract_embedding(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = dinov2_model(**inputs)

    # Mean pooling
    embedding = outputs.last_hidden_state.mean(dim=1)

    # Normalize vector
    embedding = F.normalize(embedding, p=2, dim=1)

    return embedding


# =========================
# IMAGE SIMILARITY FUNCTION
# =========================

def compute_image_similarity(image_path_1, image_path_2):
    # Detect + crop
    image1 = detect_and_crop(image_path_1)
    image2 = detect_and_crop(image_path_2)

    # Extract embeddings
    emb1 = extract_embedding(image1)
    emb2 = extract_embedding(image2)

    # Cosine similarity
    similarity = F.cosine_similarity(emb1, emb2).item()

    # Normalize from [-1,1] → [0,1]
    similarity = (similarity + 1) / 2

    return round(similarity, 4)
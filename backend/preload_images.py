
import os
import base64
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone
from openai import OpenAI
from pinecone import ServerlessSpec
import json

# --- Load Config.json and set globals ---
CONFIG_PATH = "config.json"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError("Missing config.json file")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# -------- Config --------
PINECONE_API_KEY = config["PINECONE_API_KEY"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]
INDEX_NAME = config["INDEX_NAME"]
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")
DIMENSION = config["DIMENSION"]
# ------------------------

# Init Pinecone
# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = INDEX_NAME
dimension = DIMENSION

# Check if index exists
if not pc.has_index(index_name):
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine", # or "dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index {index_name} already exists.")

# Connect to index
index = pc.Index(index_name)

# Init OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Select device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"🔥 Using device: {device}")

# Load CLIP
#model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
#processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_image(image: Image.Image):
    """Convert a PIL image into a 512-dim vector using CLIP."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding.cpu().numpy()[0].tolist()

def describe_image(image_path: str) -> str:
    """Use GPT-4o-mini to generate a short description of the image itself."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are helping create descriptions for a scavenger hunt prize list.\n"
        "Look at the uploaded image and generate a short description (1–2 sentences).\n"
        "Focus on what the object is, its color, and any distinctive features. Don't give the exact name of characters. \n"
        "Keep it under 30 words."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=80,
    )

    return resp.choices[0].message.content.strip()

# -------- Main preload --------
vectors = []

for filename in os.listdir(IMAGES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(IMAGES_DIR, filename)
        image = Image.open(path).convert("RGB")

        # CLIP embedding
        vector = embed_image(image)
        item_id = os.path.splitext(filename)[0]

        # Description from LLM
        description = describe_image(path)

        # Image will be served from frontend/public/items/
        #image_url = f"/items/{filename}"

        image_url = f"https://scavenger-hunt-items.s3.amazonaws.com/{filename}"

        vectors.append(
            (
                item_id,
                vector,
                {
                    "name": item_id,
                    "image_url": image_url,
                    "description": description,
                },
            )
        )
        print(f"✅ {filename}: {description}")

if vectors:
    index.upsert(vectors)
    print(f"🎉 Uploaded {len(vectors)} images with metadata to Pinecone")
else:
    print("⚠️ No images found in ./images/")

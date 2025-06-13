import os
import json
import requests
from PIL import Image
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

# Load the model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load your scraped data
with open("tds_discourse_rich_thread_aware_images_url.json", "r") as f:
    posts = json.load(f)

# Output file for saving image descriptions
output_path = "tds_image_descriptions_vit_gpt2.json"
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        described_images = {entry["url"]: entry["description"] for entry in json.load(f)}
else:
    described_images = {}

# Function to generate description
def describe_image_from_url(image_url):
    try:
        image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"❌ Error describing {image_url}: {e}")
        return None

# Define function to skip logos, emojis, gifs, avatars, etc.
def is_relevant_image(url):
    return (
        url.endswith((".png", ".jpg", ".jpeg", ".webp"))
        and not any(skip in url for skip in ["logo", "emoji", "smiley", "avatar", "gif"])
    )

# Describe images
output = []
for post in tqdm(posts, desc="Processing posts"):
    if "images" not in post or not post["images"]:
        continue
    for url in post["images"]:
        if not is_relevant_image(url):
            print(f"⏩ Skipping irrelevant image: {url}")
            continue
        if url in described_images:
            continue
        print(f"🔍 Describing: {url}")
        description = describe_image_from_url(url)
        if description:
            output.append({"url": url, "description": description})
            described_images[url] = description
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
        else:
            print(f"⚠️ Skipped due to error: {url}")

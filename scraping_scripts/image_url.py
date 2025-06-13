import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

def extract_image_urls_from_html(cooked_html):
    soup = BeautifulSoup(cooked_html, "html.parser")
    image_tags = soup.find_all("img")

    urls = []
    for img in image_tags:
        src = img.get("src")
        if not src:
            continue

        full_url = BASE_URL + src if src.startswith("/") else src

        # Skip emojis, avatars, Discourse UI icons
        if any(x in full_url for x in ["emoji", "user_avatar", "amazonaws.com/assets"]):
            continue

        if re.search(r"\.(png|jpe?g|webp|gif)", full_url, re.IGNORECASE):
            urls.append(full_url)

    return urls

def add_image_urls_to_posts(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        posts = json.load(f)

    for post in tqdm(posts, desc="Extracting image URLs"):
        if "image_urls" not in post or not post["image_urls"]:
            post["image_urls"] = extract_image_urls_from_html(post.get("cooked", ""))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)

    print(f"✅ Added image URLs to {len(posts)} posts → saved to {output_path}")

if __name__ == "__main__":
    input_file = "tds_discourse_rich_thread_aware_with_gemini.json"  # replace if different
    output_file = "tds_discourse_basic_with_image_urls.json"
    add_image_urls_to_posts(input_file, output_file)

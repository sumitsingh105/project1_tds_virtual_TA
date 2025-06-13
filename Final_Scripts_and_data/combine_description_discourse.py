import json

# 🔁 Input paths
DISCOURSE_JSON = "tds_discourse_rich_thread_aware_images_url.json"
DESCRIPTIONS_JSON = "tds_image_descriptions.json"
OUTPUT_JSON = "Discourse_scraped_data_with_image_descriptions.json"
import json


# Load image descriptions
with open(DESCRIPTIONS_JSON, "r") as f:
    desc_lookup = {entry["url"]: entry["description"] for entry in json.load(f)}

# Load scraped discourse posts
with open(DISCOURSE_JSON, "r") as f:
    posts = json.load(f)

# Merge descriptions into image dicts
for post in posts:
    image_urls = post.get("images", [])
    image_blocks = []

    for url in image_urls:
        image_blocks.append({
            "url": url,
            "description": desc_lookup.get(url, None)
        })

    post["images"] = image_blocks  # Replace list of URLs with list of dicts

# Save enriched output
with open(OUTPUT_JSON, "w") as f:
    json.dump(posts, f, indent=2)

print(f"✅ Inline image descriptions saved to {OUTPUT_JSON}")

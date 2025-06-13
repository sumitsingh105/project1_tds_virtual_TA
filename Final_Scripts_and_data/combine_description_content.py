import json

# File paths
COURSE_JSON = "tds_rich_chunks.json"
IMAGE_DESC_JSON = "tds_rich_chunks_image_descriptions_for_content.json"
OUTPUT_JSON = "Course_content_with_image_descriptions.json"

# Load image descriptions
with open(IMAGE_DESC_JSON, "r") as f:
    desc_lookup = {entry["url"]: entry["description"] for entry in json.load(f)}

# Load course content
with open(COURSE_JSON, "r") as f:
    course_data = json.load(f)

# Inject image descriptions inline
for chunk in course_data:
    image_urls = chunk.get("image_urls", [])
    image_blocks = []

    for url in image_urls:
        image_blocks.append({
            "url": url,
            "description": desc_lookup.get(url, None)
        })

    chunk["image_urls"] = image_blocks  # Replace list of URLs with list of dicts

# Save result
with open(OUTPUT_JSON, "w") as f:
    json.dump(course_data, f, indent=2)

print(f"✅ Course content with inline image descriptions saved to {OUTPUT_JSON}")

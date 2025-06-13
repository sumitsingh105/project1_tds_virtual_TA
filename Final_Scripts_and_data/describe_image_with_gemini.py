import os, json, time, io
import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# ───────────────────────────────
# 🔧 Config
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.0-flash")

MAX_RETRIES = 3
MAX_TOTAL_WAIT = 120  # Max total wait time for a single image
REQUEST_DELAY = 6     # Delay between requests to avoid quota burst

SCRAPED_JSON = "tds_discourse_rich_thread_aware_images_url.json"
OUTPUT_JSON = "tds_image_descriptions.json"

# 📤 Resume from saved
if os.path.exists(OUTPUT_JSON):
    with open(OUTPUT_JSON, "r") as f:
        existing_results = {entry["url"]: entry["description"] for entry in json.load(f)}
else:
    existing_results = {}

# ❌ Skip irrelevant images
def is_irrelevant_image(url):
    return any(sub in url for sub in [
        "user_avatar", "emoji", "discourse-cdn.com/assets", "logo", ".gif", ".webp"
    ])

# 🧠 Image description prompt
PROMPT = (
    "Analyze the attached image and extract all visible text and visual information. "
    "Describe layout, textual content, labels, and any diagrams or charts. "
    "Return structured plain text for search embedding."
)

# 📸 Describe image using Gemini
def describe_image(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    total_wait = 0

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=15)
            if response.status_code != 200:
                print(f"❌ Failed to fetch image: {url}")
                return None

            image_data = Image.open(io.BytesIO(response.content)).convert("RGB")
            result = gemini.generate_content([PROMPT, image_data])

            if not result.text or "I'm not able to help" in result.text.lower():
                raise ValueError("Empty or unhelpful Gemini response")

            return result.text.strip()

        except (UnidentifiedImageError, ValueError) as e:
            print(f"⚠️ Skipping unreadable or bad image: {url} ({e})")
            return None
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                wait = (2 ** attempt) * 10
                total_wait += wait
                if total_wait > MAX_TOTAL_WAIT:
                    print(f"⛔ Too long waiting for quota. Skipping: {url}")
                    return None
                print(f"⚠️ Gemini quota hit. Retrying in {wait}s... ({url})")
                time.sleep(wait)
            else:
                print(f"⚠️ Error on image ({url}): {e}")
                return None

    return None

# 📚 Load scraped posts
with open(SCRAPED_JSON, "r") as f:
    posts = json.load(f)

# 🧠 Extract and describe new images
new_results = []
for post in tqdm(posts, desc="Processing posts"):
    images = post.get("images", [])
    if not images:
        continue

    for url in images:
        if url in existing_results:
            continue
        if is_irrelevant_image(url):
            print(f"⏩ Skipping irrelevant image: {url}")
            continue

        print(f"\n🔍 Describing: {url}")
        desc = describe_image(url)
        time.sleep(REQUEST_DELAY)  # ⏱️ Rate-limiting between requests

        if desc:
            print(f"✅ Description saved for: {url}")
            result = {"url": url, "description": desc}
            new_results.append(result)
            existing_results[url] = desc

            # 💾 Save progress every image
            with open(OUTPUT_JSON, "w") as f:
                json.dump(
                    [{"url": u, "description": d} for u, d in existing_results.items()],
                    f, indent=2
                )
        else:
            print(f"❌ Failed to describe after retries: {url}")

print(f"\n✅ Done! Total image descriptions saved: {len(existing_results)}")

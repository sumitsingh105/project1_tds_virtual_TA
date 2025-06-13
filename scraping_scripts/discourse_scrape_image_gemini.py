import requests, json, time, os, re, io
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
from dotenv import load_dotenv
from PIL import Image
import google.generativeai as genai

# ───────────────────────────────────────────
# 🧪 CONFIG
load_dotenv()
DISCOURSE_COOKIE = os.getenv("DISCOURSE_COOKIE")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34

start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 15, 23, 59, 59, tzinfo=timezone.utc)

session = requests.Session()
session.cookies.set("_t", "rGCHGhiH3lvhSnUNy2oy%2F9fH9jFAb2ku%2Fyz7AMUQhgKectbhmERaBR83i6PpkHqbL69gVWQofdhoICmLIGgnAvBXoB5vQc3ZRcrB500Oxroi%2F4lQfoX%2FXPLoj%2FxT3%2F5Mmky6%2FDKTT4NhddNr3TJtjBIlG7BcMO4vgKYUxzM%2FdbZXOlKRVrU7TSk9srdV9cVNkyajOY6eyIrhI6MUkHdzStN7kmNGiEWVE2puJzEYt7VF8tg4d7bCvb5ChBw9nYB5WXH1UaRVQmdAGoPWUwKVqnWbpKg4kihaotiVqfMKsuRJl%2FU4dCG%2FclywtkGq%2BVb1UGV8lw%3D%3D--P%2FvCWYgHWCSkP%2F6B--GBbhw6GHm%2BTmZa5uR7Kj2Q%3D%3D", domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


from urllib.parse import urlparse

def is_relevant_image(url):
    parsed = urlparse(url)
    path = parsed.path.lower()

    # Skip common irrelevant image types
    if any(x in path for x in ["user_avatar", "emoji", "logo", "banner", "icon", "badge"]):
        return False

    # Skip gifs
    if path.endswith(".gif"):
        return False

    # Skip small images (likely UI elements)
    try:
        head = session.head(url)
        if "Content-Length" in head.headers:
            size = int(head.headers["Content-Length"])
            if size < 10_000:  # ~10 KB
                return False
    except:
        pass

    return True

MAX_RETRIES = 2

# ───────────────────────────────────────────
# 📸 Describe Image
# ───────────────────────────────────────────
# 📊 Load and Update Gemini Quota
def load_quota_tracker():
    tracker_path = "gemini_quota_tracker.json"
    today = datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(tracker_path):
        with open(tracker_path, "w") as f:
            json.dump({"date": today, "count": 0}, f)

    with open(tracker_path, "r") as f:
        data = json.load(f)

    if data.get("date") != today:
        data = {"date": today, "count": 0}

    return data


def save_quota_tracker(data):
    with open("gemini_quota_tracker.json", "w") as f:
        json.dump(data, f)

# ───────────────────────────────────────────
# 📸 Describe Image (with Gemini quota check)
def describe_image_with_gemini(url):
    MAX_RPD = 1500  # Gemini Flash daily quota
    quota = load_quota_tracker()

    if quota["count"] >= MAX_RPD:
        print(f"🚫 Gemini quota exhausted ({quota['count']}/{MAX_RPD}). Skipping image: {url}")
        return None

    prompt = (
        "Analyze the attached image and extract all visible text and visual information. "
        "Describe layout, textual content, labels, and any diagrams or charts. "
        "Return your response in clear, structured plain text for search embedding."
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                return None

            image_data = Image.open(io.BytesIO(response.content))
            result = gemini_model.generate_content([prompt, image_data], stream=False)

            # Update quota after successful call
            quota["count"] += 1
            save_quota_tracker(quota)

            return result.text.strip()

        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                wait = (2 ** attempt) * 10
                print(f"⚠️ Gemini quota hit. Retrying in {wait}s... ({url})")
                time.sleep(wait)
            else:
                print(f"⚠️ Gemini image error ({url}): {e}")
                return None

    return None

# ───────────────────────────────────────────
def get_topic_ids():
    topics = []
    for page in tqdm(range(0, 20), desc="Fetching topic list"):
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            print(f"❌ Failed to get topics for page {page} (Status: {r.status_code})")
            break
        data = r.json()
        new_topics = data["topic_list"].get("topics", [])
        if not new_topics:
            break
        topics.extend(new_topics)
        time.sleep(0.3)
    return topics

def extract_context(posts, current_index, n_context=3):
    context = []
    for i in range(max(0, current_index - n_context), current_index):
        soup = BeautifulSoup(posts[i]["cooked"], "html.parser")
        context.append({
            "post_number": posts[i]["post_number"],
            "username": posts[i]["username"],
            "content": soup.get_text(separator="\n").strip(),
            "created_at": posts[i]["created_at"]
        })
    return context

def slugify(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")

def get_thread_posts(topic_id, topic_title, category_id):
    all_posts = []
    page = 0
    topic_slug = slugify(topic_title)

    while True:
        url = f"{BASE_URL}/t/{topic_id}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            break

        data = r.json()
        posts_data = data["post_stream"]["posts"]
        if not posts_data:
            break

        for i, post in enumerate(posts_data):
            created = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
            if not (start_date <= created <= end_date):
                continue

            soup = BeautifulSoup(post["cooked"], "html.parser")
            content = soup.get_text(separator="\n").strip()

            image_tags = soup.find_all("img")
            image_urls = []
            image_descriptions = []

            MAX_IMAGES = 2
            
            for img in image_tags:
                src = img.get("src")
                if not src:
                    continue
            
                full_url = BASE_URL + src if src.startswith("/") else src
            
                if not is_relevant_image(full_url):
                    continue
                
            
                image_urls.append(full_url)
            
                if len(image_descriptions) >= MAX_IMAGES:
                    break
            
                desc = describe_image_with_gemini(full_url)
                image_descriptions.append(desc if desc else "")
            

            post_data = {
                "id": post["id"],
                "topic_id": post["topic_id"],
                "topic_title": topic_title,
                "category_id": category_id,
                "url": f"{BASE_URL}/t/{topic_slug}/{topic_id}/{post['post_number']}",
                "username": post["username"],
                "post_number": post["post_number"],
                "content": content,
                "created_at": post["created_at"],
                "context": extract_context(posts_data, i),
                "image_descriptions": image_descriptions
            }

            all_posts.append(post_data)

        if len(posts_data) < 20:
            break

        page += 1
        time.sleep(0.3)

    return all_posts

def scrape_all_posts():
    topics = get_topic_ids()
    all_posts = []

    for topic in tqdm(topics, desc="Scraping topics"):
        topic_id = topic["id"]
        topic_title = topic["title"]
        posts = get_thread_posts(topic_id, topic_title, CATEGORY_ID)
        all_posts.extend(posts)

    return all_posts

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ───────────────────────────────────────────
if __name__ == "__main__":
    print("🔍 Scraping Discourse posts with Gemini image descriptions...")
    all_posts = scrape_all_posts()
    output_path = "tds_discourse_rich_thread_aware_with_gemini.json"
    save_json(all_posts, output_path)
    print(f"\n✅ Saved {len(all_posts)} posts to {output_path}")

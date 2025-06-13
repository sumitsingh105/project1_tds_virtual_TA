import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import json
import time
import base64
import os
from dotenv import load_dotenv
import re  # moved here for convenience

load_dotenv()
DISCOURSE_COOKIE = os.getenv("DISCOURSE_COOKIE")

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34

start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 15, 23, 59, 59, tzinfo=timezone.utc)

session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

def get_topic_ids():
    topics = []
    for page in tqdm(range(0, 20), desc="Fetching topic list"):
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
        print(f"Fetching topics from: {url}")
        r = session.get(url)
        print(f"Status code: {r.status_code}")
        if r.status_code != 200:
            print(f"Failed to get topics for page {page}")
            break
        data = r.json()
        new_topics = data["topic_list"].get("topics", [])
        print(f"Found {len(new_topics)} topics on page {page}")
        if not new_topics:
            break
        topics.extend(new_topics)
        time.sleep(0.3)
    print(f"Total topics fetched: {len(topics)}")
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
    slug = re.sub(r"[^\w\s-]", "", slug)      # remove unwanted chars except space and dash
    slug = re.sub(r"\s+", "-", slug)          # replace spaces with dash
    slug = re.sub(r"-+", "-", slug)           # replace multiple dashes with single dash
    slug = slug.strip("-")                     # strip leading/trailing dash
    return slug

def fetch_image_as_base64(url: str) -> str:
    try:
        if url.startswith("/"):
            url = BASE_URL + url
        response = session.get(url, stream=True)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
        else:
            print(f"❌ Failed to fetch image: {url}")
            return None
    except Exception as e:
        print(f"⚠️ Exception fetching image: {e}")
        return None

def describe_image_with_ai_pipe_base64(image_base64: str) -> str:
    prompt = """
    Analyze the attached image and extract all visible text and visual information.
    Describe layout, textual content, labels, and any diagrams or charts.
    Return your response in clear, structured plain text for search embedding.
    """

    payload = {
        "model": "openai/gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a visual analysis assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ],
        "max_tokens": 1024,
        "temperature": 0.3
    }

    headers = {
        "Authorization": f"Bearer {AI_PIPE_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]



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

            for img in image_tags:
                src = img.get("src")
                if src:
                    if src.startswith("/"):
                        src = BASE_URL + src
                    image_urls.append(src)
                    image_base64 = fetch_image_as_base64(src)
                    if image_base64:
                        try:
                            description = describe_image_with_ai_pipe_base64(image_base64)
                        except Exception as e:
                            print(f"⚠️ Image AI description failed: {e}")
                            description = None
                        image_descriptions.append(description)
                    

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
                "image_urls": image_urls,
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

if __name__ == "__main__":
    print("🔍 Scraping Discourse posts with image descriptions...")
    all_posts = scrape_all_posts()
    output_path = "tds_discourse_rich_thread_aware_no_images1.json"
    save_json(all_posts, output_path)
    print(f"\n✅ Saved {len(all_posts)} posts to {output_path}")

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import json
import time
import os
from dotenv import load_dotenv

# Load Discourse login cookie
load_dotenv()
DISCOURSE_COOKIE = os.getenv("DISCOURSE_COOKIE")

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_SLUG = "courses/tds-kb"
CATEGORY_ID = 34

# Date range
start_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 14, 23, 59, 59, tzinfo=timezone.utc)

# HTTP session
session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

def get_topic_ids():
    topics = []
    for page in tqdm(range(0, 20), desc="Fetching topic list"):
        url = f"{BASE_URL}/c/{CATEGORY_SLUG}/{CATEGORY_ID}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        new_topics = data["topic_list"]["topics"]
        if not new_topics:
            break
        topics.extend(new_topics)
        time.sleep(0.3)
    return topics

def extract_context(posts, current_index, n_context=3):
    context = []
    for i in range(max(0, current_index - n_context), current_index):
        context.append({
            "post_number": posts[i]["post_number"],
            "username": posts[i]["username"],
            "content": posts[i]["content"],
            "created_at": posts[i]["created_at"]
        })
    return context

def extract_images_from_html(html):
    soup = BeautifulSoup(html, "html.parser")
    images = []
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src")
        if src and src.startswith("http"):
            images.append(src)
    return images

def get_thread_posts(topic_id, topic_title, category_id):
    r = session.get(f"{BASE_URL}/t/{topic_id}.json")
    if r.status_code != 200:
        return []

    data = r.json()
    posts_raw = data["post_stream"]["posts"]
    posts_cleaned = []

    for post in posts_raw:
        created = datetime.fromisoformat(post["created_at"].replace("Z", "+00:00"))
        if not (start_date <= created <= end_date):
            continue

        soup = BeautifulSoup(post["cooked"], "html.parser")
        content = soup.get_text(separator="\n").strip()
        images = extract_images_from_html(post["cooked"])

        post_data = {
            "thread_id": topic_id,
            "thread_title": topic_title,
            "category_id": category_id,
            "post_number": post["post_number"],
            "post_id": post["id"],
            "username": post["username"],
            "created_at": post["created_at"],
            "content": content,
            "images": images,  # 👈 new field
            "post_url": f"{BASE_URL}/t/{topic_id}/{post['post_number']}",
            "reply_to": post.get("reply_to_post_number"),
            "reply_count": post.get("reply_count", 0),
            "like_count": post.get("like_count", 0)
        }

        post_data["thread_context"] = extract_context(posts_cleaned, len(posts_cleaned), n_context=3)
        posts_cleaned.append(post_data)

    return posts_cleaned

def scrape_all_posts():
    topics = get_topic_ids()
    all_posts = []

    for topic in tqdm(topics, desc="Scraping topics"):
        topic_id = topic["id"]
        topic_title = topic["title"]
        posts = get_thread_posts(topic_id, topic_title, CATEGORY_ID)
        all_posts.extend(posts)

    return all_posts

def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    print("🔍 Scraping Discourse posts...")
    all_posts = scrape_all_posts()
    output_path = "tds_discourse_rich_thread_aware.jsonl"
    save_jsonl(all_posts, output_path)
    print(f"\n✅ Saved {len(all_posts)} posts to {output_path}")

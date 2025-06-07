import json
import time
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, urljoin, urldefrag
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
# options.add_argument("--headless")  # Optional: run headless

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
wait = WebDriverWait(driver, 20)

def split_long_text(text, max_len=1000):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_len:
            current += (" " if current else "") + sentence
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks

def get_url_metadata(url):
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {}
        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.strip() if soup.title else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        desc = desc_tag['content'].strip() if desc_tag and 'content' in desc_tag.attrs else ""
        return {"title": title, "description": desc}
    except Exception:
        return {}

def clean_url(base_url, url):
    abs_url = urljoin(base_url, url)
    parsed_url = urlparse(abs_url)
    fragment = parsed_url.fragment

    # Remove only artificial chunk anchors
    if fragment.startswith("chunk"):
        abs_url, _ = urldefrag(abs_url)
    return abs_url

base_url = "https://tds.s-anand.net/#/2025-01/"
driver.get(base_url)

wait.until(lambda d: d.find_element(By.CLASS_NAME, 'markdown-section').text.strip() != '')
sidebar_links = driver.find_elements(By.CSS_SELECTOR, 'li.file > a[href^="#/"]')

hrefs = []
for link in sidebar_links:
    href = link.get_attribute('href')
    if href and href not in hrefs:
        hrefs.append(href)

# Exclude README and ensure base_url is first
hrefs = [h for h in hrefs if not h.endswith("/README")]

if base_url not in hrefs:
    hrefs.insert(0, base_url)
else:
    hrefs.remove(base_url)
    hrefs.insert(0, base_url)

all_chunks = []

for href in hrefs:
    print(f"Loading {href} ...")
    driver.get(href)
    wait.until(lambda d: d.find_element(By.CLASS_NAME, 'markdown-section').text.strip() != '')
    time.sleep(1)

    content_div = driver.find_element(By.CLASS_NAME, 'markdown-section')
    full_text = content_div.text.strip()

    embedded_urls_raw = [a.get_attribute('href') for a in content_div.find_elements(By.TAG_NAME, 'a') if a.get_attribute('href')]
    image_urls = [img.get_attribute('src') for img in content_div.find_elements(By.TAG_NAME, 'img') if img.get_attribute('src')]

    embedded_urls = [clean_url(href, url) for url in embedded_urls_raw if url.startswith("http")]
    metadata = {url: get_url_metadata(url) for url in embedded_urls if url.startswith("http")}

    paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
    chunk_index = 0
    cleaned_href = clean_url(base_url, href)  # keeps real section anchors
    page_title = driver.title.strip()

    for para in paragraphs:
        small_chunks = split_long_text(para)
        for chunk_text in small_chunks:
            chunk_index += 1
            chunk_data = {
                "url": cleaned_href,
                "chunk_id": f"{cleaned_href}#chunk{chunk_index}",
                "title": page_title,
                "text": chunk_text,
                "embedded_urls": embedded_urls,
                "image_urls": image_urls,
                "metadata": metadata,
                "source_type": "tds_handbook",
                "source_name": "TDS 2025 Handbook"
            }
            all_chunks.append(chunk_data)

with open("tds_rich_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print("\n✅ Saved with text, embedded URLs, image links, and metadata.")
driver.quit()

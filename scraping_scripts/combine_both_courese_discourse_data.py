import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def transform_discourse_post(post):
    # Combine image URLs + descriptions into objects
    images = post.get("images", [])
    image_blocks = []

    for item in images:
        if isinstance(item, dict):
            image_blocks.append({
                "image_url": item.get("image_url"),
                "description": item.get("description")
            })
        else:
            # Backward compatibility for URL-only format
            image_blocks.append({"image_url": item, "description": None})

    return {
        "source": "discourse",
        "id": str(post["id"]),
        "title": post.get("topic_title", ""),
        "text": post.get("content", ""),
        "url": post.get("url", ""),
        "created_at": post.get("created_at"),

        "metadata": {
            "username": post.get("username"),
            "post_number": post.get("post_number"),
            "topic_id": post.get("topic_id"),
            "category_id": post.get("category_id"),
            "context": post.get("context", []),
            "images": image_blocks  # ✅ final unified structure
        }
    }

def transform_course_chunk(chunk):
    metadata_obj = list(chunk.get("metadata", {}).values())[0] if chunk.get("metadata") else {}

    return {
        "source": "course_content",
        "id": chunk.get("chunk_id", ""),
        "title": metadata_obj.get("title", ""),
        "text": chunk.get("text", ""),
        "url": chunk.get("url", ""),
        "created_at": None,

        "metadata": {
            "description": metadata_obj.get("description", ""),
            "image_urls": chunk.get("image_urls", []),   # already enriched
            "embedded_urls": chunk.get("embedded_urls", [])
        }
    }


def combine_sources(course_chunks_path, discourse_posts_path, output_path):
    print("📥 Loading data...")
    course_chunks = load_json(course_chunks_path)
    discourse_posts = load_json(discourse_posts_path)

    print("🔄 Transforming course content...")
    combined_data = [transform_course_chunk(chunk) for chunk in course_chunks]

    print("🔄 Transforming discourse posts...")
    combined_data += [transform_discourse_post(post) for post in discourse_posts]

    print(f"💾 Saving combined data to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Combined {len(combined_data)} records saved.")

if __name__ == "__main__":
    COURSE_CHUNKS_PATH = "Course_content_with_image_descriptions.json"                # Replace with your actual file
    DISCOURSE_POSTS_PATH = "Discourse_scraped_data_with_image_descriptions.json"  # Already JSON
    OUTPUT_PATH = "Combined_course_and_discourse_data.json"

    combine_sources(COURSE_CHUNKS_PATH, DISCOURSE_POSTS_PATH, OUTPUT_PATH)

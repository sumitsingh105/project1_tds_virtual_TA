import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def transform_discourse_post(post):
    return {
        "source": "discourse",
        "id": str(post["id"]),
        "title": post.get("topic_title", ""),         # your schema uses "topic_title"
        "text": post.get("content", ""),              # "content" field has post body
        "url": post.get("url", ""),                    # full post URL already present
        "created_at": post.get("created_at"),         

        "metadata": {
            "username": post.get("username"),
            "post_number": post.get("post_number"),
            "topic_id": post.get("topic_id"),
            "category_id": post.get("category_id"),
            "context": post.get("context", []),        # context array of previous posts
        }
    }


def transform_course_chunk(chunk):
    metadata_obj = list(chunk.get("metadata", {}).values())[0] if chunk.get("metadata") else {}
    return {
        "source": "course_content",
        "id": chunk.get("chunk_id", ""),
        "title": metadata_obj.get("title", ""),
        "text": chunk.get("text", ""),
        "url": chunk.get("url", ""),  # ✅ use only the source file URL
        "created_at": None,  # not available in course content
        "metadata": {
            "description": metadata_obj.get("description", ""),
            "image_urls": chunk.get("image_urls", []),
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
    COURSE_CHUNKS_PATH = "tds_rich_chunks.json"                # Replace with your actual file
    DISCOURSE_POSTS_PATH = "tds_discourse_rich_thread_aware_no_images1.json"  # Already JSON
    OUTPUT_PATH = "combined_tds_rag_data2.json"

    combine_sources(COURSE_CHUNKS_PATH, DISCOURSE_POSTS_PATH, OUTPUT_PATH)

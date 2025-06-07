from openai import OpenAI
from faiss import IndexFlatL2
import numpy as np

# Load metadata
with open("combined_tds_rag_data.json", "r") as f:
    data = json.load(f)

# Get text chunks
texts = [item["text"] for item in data]
vectors = [get_embedding(text) for text in texts]

# Build FAISS index
index = faiss.IndexFlatL2(1536)
index.add(np.array(vectors, dtype=np.float32))

# Save
faiss.write_index(index, "tds_faiss.index")
with open("tds_metadata.json", "w") as f:
    json.dump(data, f, indent=2)

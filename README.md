
# 🎓 TDS Virtual TA – AI Assistant for IITM Online Degree

This project is a Virtual Teaching Assistant (TA) designed for the **Tools for Data Science (TDS)** course in the IIT Madras Online Degree program. It answers student queries using course material, discussion threads, and relevant links.

---

## 📌 Features

- FastAPI backend
- RAG-based answering (using Typesense Indexing + Nomic Embedding)
- Supports image (OCR) and text input
- Promptfoo test suite for evaluation
- Deployed on Render / can run locally with ngrok

---

## ⚙️ Project Structure
project-root/
├── app/ # FastAPI backend
│ ├── main.py # Entry point
│ ├── api.py # /api/ endpoint
│ ├── build_prompt.py # Builds structured prompts
│ ├── loader.py # Load and index documents
│ └── ... # Additional utilities
├── test/ # promptfoo test suite
│ └── test1.yaml
├── data/ # Indexed or raw content
├── requirements.txt
└── README.md # This file


---

## Main File
### For Discourse Content Scraping
- discourse_scraper4.py

### for Course Content Scraping
- scrape_tds_content.py

### for Nomic embedding and Typesense indexing script run this .sh file. These are in utils folder.
- run_indexing_batches.sh
- typesense_indexer_range.py

- ✅ Instructions
  - Run permissions for the script:
   - chmod +x run_indexing_batches.sh

  - Execute it:
   - ./run_indexing_batches.sh


---

## 🚀 Running Locally

1. **Clone the repo & install dependencies**

```bash
git clone https://github.com/your-username/tds-virtual-ta.git
cd tds-virtual-ta
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


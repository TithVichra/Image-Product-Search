# FashionLens AI - Multimodal Fashion Product Search Engine

FashionLens AI is a high-performance multimodal fashion product search engine that pairs **OpenAI CLIP (ViT-B/32)**, **YOLOv8 object detection with focus-cropping**, and **embedded Qdrant vector database**. It enables users to search fashion catalogs using either natural descriptive text or uploaded photos.

---

## Key Features

- **Dual-Modal Search**:
  - **Search by Description**: 75% text similarity + 25% calibrated image similarity, boosted by stem-aware N-gram contextual coherence.
  - **Search by Image**: 75% visual similarity + 25% text similarity with automatic focus cropping.
- **YOLOv8 Focus-Crop & Background Elimination**:
  - Automatically detects the primary fashion item and crops away clutter and background noise.
  - Out-of-domain detection rejects non-fashion entities (e.g. dogs, vehicles, balls) before expensive vector searches.
- **Stem-Aware N-Gram Context Coherence**:
  - Morphological stem normalization handles singular/plural inflections (e.g. `sock` <-> `socks`, `bra` <-> `bras`, `lipsticks` <-> `lipstick`).
  - Comprehensive fashion lexicon spanning apparel, footwear, accessories, and personal care.
- **Rank Gap Noise Elimination**:
  - Detects score cliffs between adjacent ranks and removes low-confidence tail results.
- **Embedded Vector Database**:
  - Utilizes embedded Qdrant with Dot Product similarity on L2-normalized 512-dimensional embeddings.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Embeddings & Vision** | OpenAI CLIP (`openai/clip-vit-base-patch32`), Ultralytics YOLOv8 |
| **Vector Engine** | Qdrant (Embedded Local Storage) |
| **Web Framework** | Django 5.0+ & FastAPI |
| **Frontend** | HTML5, Vanilla CSS (Glassmorphism design), JavaScript |
| **Dataset Source** | Param Aggarwal Fashion Product Images Small (`kagglehub`) |

---

## Project Structure

```
Image Product Search/
├── ImgSearchWebApp/         # Django project configuration & settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── api/                     # FastAPI alternative endpoints
│   └── fastapi_app.py
├── search_app/              # Django search application & frontend
│   ├── templates/index.html # Search web UI
│   ├── static/              # CSS, JavaScript & icons
│   ├── urls.py
│   └── views.py
├── search_engine/           # Core AI & vector search pipeline
│   ├── clip_service.py      # OpenAI CLIP embedding service
│   ├── detector.py          # YOLOv8 object detector & salient contour cropper
│   ├── chunking.py          # Fashion lexicon, intent extraction & stemming
│   ├── scoring.py           # Dual-modal score blending & noise filtering
│   ├── vector_db.py         # Qdrant client & dual-collection indexing
│   └── ingestion.py        # Dataset batch ingestion pipeline
├── scripts/                 # Utility scripts & test assets
│   └── prepare_data.py      # Dataset bootstrap & image generation
├── tests/                   # Automated test suite
├── media/                   # User uploads, crops & served catalog images
├── requirements.txt         # Project dependencies
├── .gitignore               # Git exclusion rules
└── README.md
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fashionlens-ai.git
cd fashionlens-ai
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv env
.\env\Scripts\activate

# Linux / macOS
python3 -m venv env
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare Dataset & Indexing
```bash
# Bootstrap sample catalog and images
python scripts/prepare_data.py
```

To ingest and index products from the Kaggle dataset via batch ingestion:
- Open the web application and click **Settings** in the header.
- Select the catalog size (e.g. 1,000, 2,500, or 10,000 items) and click **Start Batch Ingestion**.

### 5. Run the Server

#### Django Web App (Default)
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000` in your web browser.

#### FastAPI Alternative
```bash
uvicorn api.fastapi_app:fastapi_app --reload --port 8000
```

---

## API Endpoints

- `GET /api/status` - Returns vector database status and vector counts.
- `POST /api/search/text` - Text-based fashion search with JSON body:
  ```json
  {
    "query": "navy blue check shirt",
    "eliminate_noise": true,
    "gap_threshold": 0.12,
    "max_results": 10
  }
  ```
- `POST /api/search/image` - Multipart form image upload with optional YOLO crop toggle.
- `POST /api/detect` - Upload an image to preview YOLOv8 bounding boxes and focus crops.
- `POST /api/index` - Trigger asynchronous background batch ingestion.

---

## Running Tests

Run the test suite to verify search scoring and out-of-domain detection:
```bash
pytest tests/
```

---

## License

MIT License. Dataset from Kaggle (`paramaggarwal/fashion-product-images-small`).

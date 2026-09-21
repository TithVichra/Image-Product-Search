# FashionLens AI - Multimodal Fashion Product Search Engine

FashionLens AI is a fast and intelligent fashion product search engine that pairs **OpenAI CLIP (ViT-B/32)**, **YOLOv8 object detection with focus-cropping**, and an **embedded Qdrant vector database**. 

It lets users search fashion catalogs in two natural ways:
1. **Search by Description**: Type natural text queries (e.g. *"navy blue sports shoes"*, *"red summer maxi dress"*).
2. **Search by Image**: Upload any photo, snapshot, or screenshot to instantly find matching clothing, footwear, and accessories.

Both **Django** (with a modern glassmorphic web UI) and **FastAPI** are fully supported out of the box.

---

## How It Works

1. **Shared 512-D Latent Space**: Product titles and photos are encoded into 512-dimensional vectors using OpenAI CLIP. Because text and images share the same vector space, cross-modal comparisons are direct and lightning-fast.
2. **Unified Vector Table**: Catalog images and standardized product titles are indexed together in a single embedded Qdrant collection (`fashion_unified_vectors`) using hardware-accelerated Dot Product (Cosine) similarity.
3. **Smart Focus-Cropping**: When you upload a picture, YOLOv8 and computer vision contour detection locate the primary fashion item, crop out background clutter, and search with the cleaned subject.
4. **Out-of-Domain Guardrails**: Non-fashion uploads (such as dogs, cars, sports balls, or food) are automatically caught and rejected before running expensive searches.
5. **Score Fusion & Reranking**: Modality scores ($S_{\text{image}}$ and $S_{\text{title}}$) are blended together (e.g. 75% title + 25% image for text queries; 85% image + 15% title for visual searches).
6. **Noise Elimination**: If candidate relevance drops steeply (a score "cliff"), low-confidence tail results are automatically filtered out.

---

## Key Features

- **Dual-Modal Search**:
  - **Text Search**: 75% title similarity + 25% image similarity, boosted by prompt ensembling and fashion synonym context.
  - **Image Search**: 85% visual similarity + 15% title similarity, with visual fidelity preservation for near-identical matches.
- **YOLOv8 & Salient Contour Focus-Cropping**:
  - Automatically identifies clothing, bags, shoes, and accessories.
  - Crops out room interiors, clutter, and distracting backgrounds so search focuses purely on the garment.
- **Multi-Stage Out-of-Domain Guardrails**:
  - Discards non-fashion queries (e.g. animal photos, cars, balls) before running database searches.
  - Catalog match protection: If an image strongly matches a catalog item ($\ge 0.80$ similarity), it is safely recognized as in-domain.
- **Category & Gender Guardrails**:
  - Queries for *"red dress"* strictly isolate dresses (preventing irrelevant items like t-shirts or bras).
  - Queries specifying gender (e.g. *"for women"*) strictly filter out male apparel.
- **Stem-Aware N-Gram Lexicon**:
  - Handles singular and plural inflections naturally (e.g. `sock` $\leftrightarrow$ `socks`, `bra` $\leftrightarrow$ `bras`, `lipsticks` $\leftrightarrow$ `lipstick`).
  - Covers hundreds of fashion terms across topwear, bottomwear, footwear, ethnic wear, and accessories.
- **Embedded Vector Database**:
  - Uses embedded Qdrant running locally—no separate database server installation or setup required.
- **Modern Glassmorphic Web UI**:
  - Responsive dark-mode interface with live canvas bounding box previews, real-time search latency, and a built-in catalog indexing manager.

---

## Tech Stack

| Layer | Technology | Description |
|---|---|---|
| **AI Models** | OpenAI CLIP (`ViT-B/32`) | Multimodal text & image embeddings (512 dimensions) |
| **Object Detection** | Ultralytics YOLOv8 (`yolov8n.pt`) | Real-time object detection & bounding boxes |
| **Computer Vision** | OpenCV (`cv2`) & Pillow (`PIL`) | Salient contour detection, Otsu thresholding, image crops |
| **Vector Database** | Qdrant (Embedded) | Local persistent vector storage with Dot Product similarity |
| **Web Frameworks** | Django 5.0+ & FastAPI | Dual backend support (full web app & fast async REST API) |
| **Frontend** | HTML5, Vanilla CSS, JavaScript | Glassmorphic dark UI with interactive bounding box canvas |
| **Dataset** | Kaggle Fashion Dataset | `paramaggarwal/fashion-product-images-small` |

---

## Project Structure

```
Image Product Search/
├── ImgSearchWebApp/         # Django project settings and root routing
│   ├── settings.py          # Media, static, and app configurations
│   ├── urls.py
│   └── wsgi.py
├── api/                     # FastAPI alternative backend
│   └── fastapi_app.py       # REST API endpoints & background tasks
├── search_app/              # Main Django web application
│   ├── templates/index.html # Search UI template & settings modal
│   ├── static/              # Glassmorphic CSS and JavaScript
│   ├── urls.py              # Application routing
│   └── views.py             # Unified search pipeline & view controllers
├── search_engine/           # Core AI & vector engine modules
│   ├── clip_service.py      # CLIP embeddings & zero-shot domain classification
│   ├── detector.py          # YOLOv8 + Salient contour focus cropper
│   ├── chunking.py          # Fashion lexicon, intent extraction & stemming
│   ├── scoring.py           # Multi-modal score blending & noise filtering
│   ├── vector_db.py         # Embedded Qdrant client & unified vector table
│   └── ingestion.py        # Dataset validation, batching & indexing pipeline
├── scripts/                 # Utility scripts & sample test images
│   └── prepare_data.py      # Dataset bootstrap & sample catalog generator
├── tests/                   # Automated test suite
│   ├── test_unified_pipeline.py         # Contract compliance tests
│   ├── test_out_of_domain_and_gender.py # Domain & guardrail tests
│   ├── test_search_engine.py            # Unit tests for scoring & chunking
│   └── test_live_api.py                 # Live server API tests
├── data/                    # Dataset directory (styles.csv, images/)
├── media/                   # User uploads, focus crops & catalog images
├── qdrant_storage/          # Embedded Qdrant local storage
├── requirements.txt         # Project dependencies
├── .env.example             # Example environment configuration
└── README.md                # Project documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/fashionlens-ai.git
cd fashionlens-ai
```

### 2. Set Up Virtual Environment
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

### 4. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

### 5. Prepare Dataset & Indexing
Bootstrap the local fashion dataset and generate catalog images:
```bash
python scripts/prepare_data.py
```
> **Tip**: You can also index larger catalogs directly inside the Web UI! Open the app, click **Settings** in the header, choose your desired catalog size (e.g. 1,000 or 2,500 items), and click **Start Batch Ingestion**.

### 6. Run the Server

#### Option A: Django Web App (Recommended)
Starts the full web interface on port 8000:
```bash
python manage.py runserver
```
Visit **`http://127.0.0.1:8000`** in your browser.

#### Option B: FastAPI Alternative
Starts the async FastAPI server:
```bash
uvicorn api.fastapi_app:fastapi_app --reload --port 8000
```
Visit **`http://127.0.0.1:8000/docs`** for interactive Swagger documentation.

---

## API Guide

FashionLens AI provides two simple, unified search endpoints alongside extended endpoints for UI controls.

### 1. Unified Search Contract

#### `POST /search/text`
Search the catalog using natural text descriptions.

- **Request Body**:
  ```json
  {
    "query": "navy blue sports shoes",
    "top_k": 10
  }
  ```
- **Response** (`200 OK`):
  ```json
  [
    {
      "product_id": 1004,
      "name": "White Air Cushion Athletic Running Shoes",
      "image_url": "/media/images/1004.jpg",
      "final_score": 0.8412,
      "breakdown": {
        "image_score": 0.8123,
        "title_score": 0.8701
      },
      "rank": 1,
      "gender": "Men",
      "articleType": "Sports Shoes",
      "baseColour": "White"
    }
  ]
  ```

#### `POST /search/image`
Search the catalog using an uploaded photo.

- **Form Data**:
  - `file`: Product image file (JPEG/PNG/WebP)
  - `top_k`: Number of results (default `10`)
- **Response** (`200 OK`):
  Returns the top 10 unique matching products in the exact same format as `/search/text`.

---

### 2. Extended Web UI Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | `GET` | Returns vector database status, active device (CUDA/CPU), and indexed item counts |
| `/api/search/text` | `POST` | Text search with tail noise elimination and latency metadata |
| `/api/search/image` | `POST` | Image search with YOLO focus crop preview and out-of-domain checks |
| `/api/detect` | `POST` | Previews detected YOLO bounding boxes and crop regions without searching |
| `/api/index` | `POST` | Triggers background batch ingestion for scalable catalog indexing |

---

## Running Tests

Run the test suite to verify search accuracy, contract compliance, and guardrails:

```bash
# 1. Verify contract endpoints (/search/text & /search/image)
python tests/test_unified_pipeline.py

# 2. Test out-of-domain rejection and gender/category guardrails
python -m unittest tests/test_out_of_domain_and_gender.py

# 3. Test core scoring, dot products, and chunking
python -m unittest tests/test_search_engine.py

# 4. Run all tests with pytest
pytest tests/
```

---

## License & Credits

- **License**: MIT License.
- **Dataset**: Param Aggarwal's Fashion Product Images Small from Kaggle (`paramaggarwal/fashion-product-images-small`).
- **Models**: OpenAI CLIP ViT-B/32 (Hugging Face) and Ultralytics YOLOv8.

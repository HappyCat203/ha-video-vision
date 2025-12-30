# 👤 Facial Recognition Setup Guide

This guide will help you set up the DeepFace facial recognition server for HA Video Vision.

## Overview

Facial recognition runs on a **separate server** (not on Home Assistant) because:
- It requires GPU for fast inference
- DeepFace/TensorFlow are heavy dependencies
- Keeps HA lightweight and responsive

## Requirements

- A machine with:
  - Python 3.9+
  - NVIDIA GPU (recommended) or powerful CPU
  - 4GB+ RAM
- Network access from Home Assistant to this machine

## Installation

### 1. Install Dependencies

```bash
pip install fastapi uvicorn deepface pillow numpy python-multipart tf-keras
```

### 2. Create the Server Script

Save as `facial_recognition_server.py`:

```python
"""Facial Recognition API Server using DeepFace."""
import os
import logging
import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
FACES_DIR = os.getenv("FACES_DIR", "./faces")
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.50"))
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet512")

app = FastAPI(title="Facial Recognition API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

known_faces: dict[str, list[np.ndarray]] = {}
deepface_initialized = False


class IdentifyRequest(BaseModel):
    image_base64: str


class IdentifyResponse(BaseModel):
    faces_detected: int
    people: list[dict[str, Any]]
    summary: str


def load_known_faces():
    """Load all known face embeddings from the faces directory."""
    global known_faces, deepface_initialized
    from deepface import DeepFace

    known_faces.clear()
    faces_path = Path(FACES_DIR)

    if not faces_path.exists():
        logger.warning(f"Faces directory not found: {FACES_DIR}")
        faces_path.mkdir(parents=True, exist_ok=True)
        return

    for person_dir in faces_path.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        embeddings = []

        for img_file in person_dir.glob("*"):
            if img_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    result = DeepFace.represent(
                        str(img_file),
                        model_name=MODEL_NAME,
                        enforce_detection=False
                    )
                    if result:
                        embeddings.append(np.array(result[0]["embedding"]))
                        logger.info(f"Loaded: {person_name}/{img_file.name}")
                except Exception as e:
                    logger.error(f"Failed to process {img_file}: {e}")

        if embeddings:
            known_faces[person_name] = embeddings
            logger.info(f"Loaded {len(embeddings)} embeddings for {person_name}")

    deepface_initialized = True
    logger.info(f"Ready! Known people: {list(known_faces.keys())}")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine distance between two vectors."""
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def identify_face(embedding: np.ndarray) -> tuple[str, float]:
    """Match embedding against known faces."""
    best_match = "Unknown"
    best_distance = float("inf")

    for name, embeddings in known_faces.items():
        for known_emb in embeddings:
            dist = cosine_distance(embedding, known_emb)
            if dist < best_distance:
                best_distance = dist
                if dist < DISTANCE_THRESHOLD:
                    best_match = name

    confidence = max(0, min(100, int((1 - best_distance) * 100)))
    return best_match, confidence


@app.on_event("startup")
async def startup():
    load_known_faces()


@app.post("/identify", response_model=IdentifyResponse)
async def identify(request: IdentifyRequest):
    """Identify faces in an image."""
    from deepface import DeepFace

    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        img_array = np.array(image)

        # Get face embeddings
        faces = DeepFace.represent(
            img_array,
            model_name=MODEL_NAME,
            enforce_detection=False
        )

        people = []
        for face in faces:
            embedding = np.array(face["embedding"])
            name, confidence = identify_face(embedding)
            people.append({"name": name, "confidence": confidence})

        # Build summary
        known = [p for p in people if p["name"] != "Unknown"]
        if known:
            summary = ", ".join([f"{p['name']} ({p['confidence']}%)" for p in known])
        else:
            summary = "No known faces detected"

        return IdentifyResponse(
            faces_detected=len(faces),
            people=people,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status():
    """Get server status."""
    return {
        "status": "ready" if deepface_initialized else "initializing",
        "known_people": list(known_faces.keys()),
        "model": MODEL_NAME
    }


@app.post("/reload")
async def reload():
    """Reload known faces from disk."""
    load_known_faces()
    return {
        "status": "reloaded",
        "known_people": list(known_faces.keys())
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
```

### 3. Create Face Database

Create a folder structure with photos of each person:

```
faces/
├── Carlos/
│   ├── front.jpg
│   ├── left_angle.jpg
│   ├── right_angle.jpg
│   └── outdoor.jpg
├── Elise/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Mom/
    └── photo1.jpg
```

**Photo Tips for Best Results:**

| ✅ Do | ❌ Don't |
|-------|---------|
| Clear, front-facing | Blurry or dark |
| Good lighting | Heavy shadows |
| Multiple angles | Only one photo |
| Similar quality to cameras | Very different lighting |
| 3-5 photos per person | Too few or too many |

### 4. Run the Server

```bash
python facial_recognition_server.py
```

Or with environment variables:

```bash
FACES_DIR=/path/to/faces DISTANCE_THRESHOLD=0.45 python facial_recognition_server.py
```

### 5. Test the Server

```bash
curl http://localhost:8100/status
```

Should return:
```json
{
  "status": "ready",
  "known_people": ["Carlos", "Elise", "Mom"],
  "model": "Facenet512"
}
```

### 6. Configure in Home Assistant

1. Go to **Settings → Devices & Services → HA Video Vision → Configure**
2. Select **Facial Recognition**
3. Enter:
   - **URL**: `http://YOUR_SERVER_IP:8100`
   - **Enable**: ✅
   - **Minimum Confidence**: 50% (adjust as needed)

## Running as a Service (Linux)

Create `/etc/systemd/system/facial-recognition.service`:

```ini
[Unit]
Description=Facial Recognition Server
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/server
Environment="FACES_DIR=/path/to/faces"
ExecStart=/usr/bin/python3 facial_recognition_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable facial-recognition
sudo systemctl start facial-recognition
```

## Running with Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install fastapi uvicorn deepface pillow numpy python-multipart tf-keras

COPY facial_recognition_server.py .
COPY faces/ ./faces/

EXPOSE 8100
CMD ["python", "facial_recognition_server.py"]
```

```bash
docker build -t facial-recognition .
docker run -d -p 8100:8100 -v /path/to/faces:/app/faces facial-recognition
```

## Tuning

### Distance Threshold

- **Lower (0.40)**: Stricter matching, fewer false positives
- **Higher (0.55)**: More lenient, may have false positives
- **Default (0.50)**: Good balance

### Model Choice

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| Facenet512 | Fast | Very Good | 92MB |
| VGG-Face | Medium | Good | 500MB |
| ArcFace | Slow | Excellent | 250MB |

Set via environment: `MODEL_NAME=ArcFace`

## Troubleshooting

### "No known faces detected" for everyone

- Check photo quality
- Add more photos from different angles
- Lower the DISTANCE_THRESHOLD

### Server slow to start

- First run downloads the model (~100MB)
- GPU acceleration significantly helps

### Connection refused from HA

- Check firewall allows port 8100
- Verify server IP is correct
- Test with curl from HA machine

## Privacy Note

All facial recognition runs **locally on your hardware**. No images are ever sent to external servers. Your family's biometric data stays in your home.

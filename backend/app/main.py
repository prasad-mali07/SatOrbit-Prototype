import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models import ChangeDetectionRequest
from app.aois import AOI_LIBRARY
from app.pipeline import run_change_detection_pipeline

app = FastAPI(title="SatOrbit Change Detection API")

# Define Base Output Directories
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = BASE_DIR / "data" / "outputs"
RAW_DIR = OUTPUTS_DIR / "raw"
PROCESSED_DIR = OUTPUTS_DIR / "processed"

# Ensure all output directories exist before mounting static files
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# CORS middleware enabling Vite/React frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to prevent browsers from blocking interactive map iframe embedding
@app.middleware("http")
async def allow_iframe_embedding(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    return response

# Static file mount for serving map HTMLs and processed PNG images
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")


@app.get("/")
def root():
    """Healthcheck endpoint to verify API and static mount status."""
    return {
        "status": "online",
        "service": "SatOrbit Change Detection API",
        "outputs_route": "http://localhost:8000/outputs",
    }


@app.get("/api/aois")
def get_aois():
    """Returns available AOIs formatted for frontend selection controls."""
    return [
        {
            "id": key,
            "name": key.capitalize(),
            "coordinates": data["coords"],
            "default_dates": data.get("default_dates", []),
        }
        for key, data in AOI_LIBRARY.items()
    ]


@app.post("/api/change-detection")
@app.post("/analyze")
def detect_changes(payload: ChangeDetectionRequest):
    """Executes the change detection pipeline for a specified AOI and date range."""
    # Resolve coordinates: use payload coordinates if provided, otherwise fallback to library lookup
    coords = getattr(payload, "coords", None)

    if not coords:
        if payload.aoi_id in AOI_LIBRARY:
            coords = AOI_LIBRARY[payload.aoi_id]["coords"]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"AOI '{payload.aoi_id}' not found in AOI_LIBRARY and no coordinates were provided.",
            )

    try:
        results = run_change_detection_pipeline(
            aoi_id=payload.aoi_id,
            coords=coords,
            date1=payload.date1,
            date2=payload.date2,
        )
        return results
    except Exception as e:
        print(f"[Pipeline Error]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
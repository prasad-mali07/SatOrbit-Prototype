from pydantic import BaseModel
from typing import List, Optional

class AOIPoint(BaseModel):
    lon: float
    lat: float

class AOIResponse(BaseModel):
    id: str
    name: str
    coordinates: List[List[float]]
    default_dates: List[str]

class ChangeDetectionRequest(BaseModel):
    aoi_id: str
    date1: str
    date2: str

class ClassificationStats(BaseModel):
    new_construction_km2: float
    vegetation_loss_km2: float
    surface_water_km2: float

class ChangeDetectionResponse(BaseModel):
    aoi_id: str
    date1: str
    date2: str
    total_pixels: int
    changed_pixels: int
    change_percentage: float
    changed_area_km2: float
    classification: ClassificationStats
    image_date1_url: str
    image_date2_url: str
    change_map_url: str
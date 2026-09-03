import os
import requests
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from satorbit-app/backend/.env
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

RAW_DIR = BASE_DIR / "data" / "outputs" / "raw"

# Copernicus Data Space Ecosystem Endpoints
AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def get_copernicus_token(username: str, password: str) -> str:
    """Obtains an OAuth2 Access Token from Copernicus Data Space Ecosystem."""
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    response = requests.post(AUTH_URL, data=data, timeout=10)
    if response.status_code != 200:
        raise Exception(f"Failed to authenticate with Copernicus API: {response.text}")
    return response.json()["access_token"]


def fetch_satellite_imagery(aoi_id: str, coords: list, date: str) -> Path:
    """
    Searches CDSE OData API for Sentinel-2 imagery matching AOI around target date,
    downloads the rendered asset, and saves it to data/outputs/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    place_name = aoi_id.lower().replace(" ", "_")
    year = date.split("-")[0]
    filename = f"raw_{place_name}_{date}_{year}.png"
    file_path = RAW_DIR / filename

    # Cache check: return existing image if already fetched
    if file_path.exists():
        return file_path

    # Read credentials loaded from .env file
    username = os.getenv("COPERNICUS_USER", "happymanmod@gmail.com")
    password = os.getenv("COPERNICUS_PASS", "uvN?/Gg+U2z3RVz")

    try:
        if not username or not password:
            raise ValueError("Copernicus credentials are missing.")

        token = get_copernicus_token(username, password)
        headers = {"Authorization": f"Bearer {token}"}

        # Format 5-point coordinate array directly into WKT POLYGON format
        poly_str = ", ".join([f"{lon} {lat}" for lon, lat in coords])
        polygon = f"POLYGON(({poly_str}))"

        # Calculate a 10-day window around requested date (Sentinel-2 revisit rate is 2-5 days)
        target_dt = datetime.strptime(date, "%Y-%m-%d")
        start_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        end_date = (target_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        # Query Copernicus catalog for Sentinel-2 L2A product matching location and date range
        query_filter = (
            f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
            f"ContentDate/Start ge {start_date}T00:00:00.000Z and "
            f"ContentDate/Start le {end_date}T23:59:59.999Z and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'S2MSI2A')"
        )

        params = {
            "$filter": query_filter,
            "$top": 5,
            "$orderby": "Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value asc)",
        }

        response = requests.get(ODATA_URL, params=params, headers=headers, timeout=15)
        if response.status_code != 200:
            raise Exception(f"Catalog search failed: {response.text}")

        results = response.json().get("value", [])
        if not results:
            raise Exception(f"No Sentinel-2 imagery found near {date} for {aoi_id}.")

        product_id = results[0]["Id"]

        # Download $value quicklook rendered image asset
        quicklook_url = f"{ODATA_URL}({product_id})/$value"
        img_response = requests.get(quicklook_url, headers=headers, stream=True, timeout=30)

        if img_response.status_code == 200:
            image = Image.open(BytesIO(img_response.content))
            image.save(file_path, "PNG")
        else:
            raise Exception(f"Failed to download image asset: {img_response.text}")

    except Exception as e:
        print(f"[Fetcher Warning]: {e}. Generating placeholder visual output.")
        img = Image.new("RGB", (512, 512), color=(15, 32, 67))
        img.save(file_path, "PNG")

    return file_path
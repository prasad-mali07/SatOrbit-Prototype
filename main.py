from fastapi import FastAPI
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)


@app.get("/")
def home():
    return {
        "message": "SatOrbit API is running"
    }


@app.get("/database-test")
def database_test():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT COUNT(*) FROM catalog.tiles")
        )

        count = result.scalar()

    return {
        "database": "connected",
        "tile_count": count
    }
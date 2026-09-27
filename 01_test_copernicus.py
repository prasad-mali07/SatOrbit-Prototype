# Step 1: Import required libraries
import os
import boto3
from dotenv import load_dotenv

# Step 2: Load environment variables
load_dotenv()

# Step 3: Read Copernicus credentials
ACCESS_KEY = os.getenv("CDSE_S3_ACCESS_KEY")
SECRET_KEY = os.getenv("CDSE_S3_SECRET_KEY")

# Step 4: Check credentials
if not ACCESS_KEY or not SECRET_KEY:
    raise RuntimeError("Copernicus S3 credentials are missing from .env")

# Step 5: Create Copernicus S3 connection
s3 = boto3.client(
    "s3",
    endpoint_url="https://eodata.dataspace.copernicus.eu",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="default",
)

# Step 6: Print success message
print("Copernicus connection created successfully.")
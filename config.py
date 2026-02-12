"""
Configuration for the Half Marathon Predictor application.
Load environment variables from .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

# Digital Ocean Spaces
DO_SPACES_CONFIG = {
    "region_name":           os.getenv("DO_SPACES_REGION", "fra1"),
    "endpoint_url":          os.getenv("DO_SPACES_ENDPOINT", "https://fra1.digitaloceanspaces.com"),
    "aws_access_key_id":     os.getenv("DO_SPACES_KEY"),
    "aws_secret_access_key": os.getenv("DO_SPACES_SECRET"),
}
DO_SPACES_BUCKET       = os.getenv("DO_SPACES_BUCKET", "halfmarathon-predictor")
DO_SPACES_DATA_PREFIX  = "data/"
DO_SPACES_MODEL_PREFIX = "models/"

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Langfuse
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# Model
MODEL_NAME = "halfmarathon_model.pkl"
FEATURES   = ["age", "gender_enc", "time_5km_s"]

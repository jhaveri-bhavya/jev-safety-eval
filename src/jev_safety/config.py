"""Environment and model settings, loaded once from .env."""

import os

from dotenv import load_dotenv

from jev_safety import ROOT

# override=True: the project's .env wins over stale user-level variables of the same name
load_dotenv(ROOT / ".env", override=True)

TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LISA_API_KEY = os.getenv("LISA_API_KEY", "")

JEV_MODEL = "jev-1.13.0"
GEMMA_HOST = "https://ollama.com"
GEMMA_MODEL = "gemma4:31b"

DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-1.0"

import os
from pathlib import Path

from dotenv import load_dotenv

# Load OPENAI_API_KEY from the shared .env in the folder above this project
load_dotenv(Path(__file__).parent.parent / ".env")

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY is not set. Add it to the .env file in the project root.")

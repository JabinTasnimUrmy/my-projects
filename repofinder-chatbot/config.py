# config.py
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_API_TOKEN")


GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
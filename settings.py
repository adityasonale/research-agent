import os
from dotenv import load_dotenv
import yaml

load_dotenv()

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DEEPSEEK_R1_CONFIG = cfg["deepseek_r1"]
MINILM_CONFIG = cfg["minilm"]

LLM_SERVICE = cfg["llm_service"]
LOCAL_MODEL = cfg["local_model"]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
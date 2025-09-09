import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    token: str = os.getenv("GITHUB_TOKEN", "")
    owner: str = os.getenv("OWNER", "apache")
    repo: str = os.getenv("REPO", "airflow")
    state: str = os.getenv("STATE", "closed")  # open|closed|all
    per_page: int = int(os.getenv("PER_PAGE", "100"))
    target_count: int = int(os.getenv("TARGET_COUNT", "100"))
    max_pages: int = int(os.getenv("MAX_PAGES", "50"))
    sleep_between_pages: float = float(os.getenv("SLEEP_BETWEEN_PAGES", "0"))
    out_csv: str = os.getenv("OUT", "prs.csv")

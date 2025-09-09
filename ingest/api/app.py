from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
import os

from ingest.pr_fetcher.config import Settings
from ingest.pr_fetcher.extract import FIELDS
from ingest.pr_fetcher.sinks import write_csv
from ingest.pr_fetcher.service import fetch_rows

app = FastAPI(title="PR Fetcher API")

# Where to save by default
DATA_RAW = Path(os.getenv("DATA_RAW_DIR", "data/raw")).resolve()
DATA_RAW.mkdir(parents=True, exist_ok=True)

class FetchRequest(BaseModel):
    owner: str = Field(..., examples=["apache"])
    repo: str  = Field(..., examples=["airflow"])
    state: str = Field("closed", pattern="^(open|closed|all)$")
    per_page: int = 100
    target_count: int = 100      # used when state=closed
    max_pages: int = 50
    sleep_between_pages: float = 0.0
    filename: str | None = None  # optional override for output file name

class FetchResponse(BaseModel):
    saved_path: str
    rows: int
    owner: str
    repo: str
    state: str

@app.post("/fetch_prs", response_model=FetchResponse)
def fetch_prs(req: FetchRequest):
    # build settings
    s = Settings()
    s.owner = req.owner
    s.repo = req.repo
    s.state = req.state
    s.per_page = req.per_page
    s.target_count = req.target_count
    s.max_pages = req.max_pages
    s.sleep_between_pages = req.sleep_between_pages

    # pick output file under data/raw
    if req.filename:
        out = DATA_RAW / req.filename
        if not out.name.endswith(".csv"):
            out = out.with_suffix(".csv")
    else:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = DATA_RAW / f"{s.owner}_{s.repo}_{s.state}_{ts}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    s.out_csv = str(out)

    try:
        rows = fetch_rows(s)
        write_csv(rows, s.out_csv, fieldnames=FIELDS)
        return FetchResponse(saved_path=s.out_csv, rows=len(rows), owner=s.owner, repo=s.repo, state=s.state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

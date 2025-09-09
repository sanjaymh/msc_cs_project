# GitHub PR Fetcher (refactor)

Fetch GitHub pull requests to CSV with robust pagination, rate-limit handling, and a clean CSV schema compatible with your existing pipeline.

## Quickstart
1. Python 3.10+
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and set `GITHUB_TOKEN`.
4. Run:

```bash
python -m pr_fetcher.cli --owner apache --repo airflow --state closed --target-count 100 --out prs.csv
```

## Notes
- If `merged_by` is missing from the list endpoint, we fetch the PR detail.
- We keep only merged PRs when `--state closed`.
- Body is cleaned with your `utility.clean_body`.
- CSV columns: `id,number,title,author,body,created_at,state,labels,reviewers`.

## To run the server
```bash
# set env once at repo root
export GITHUB_TOKEN=ghp_...
# optional: change where files go
# export DATA_RAW_DIR=/absolute/path/to/data/raw

# start API (from repo root)
uvicorn ingest.api.app:app --reload --port 8080
```
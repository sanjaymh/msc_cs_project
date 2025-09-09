import argparse, os
from pathlib import Path
from .config import Settings
from .sinks import write_csv
from .extract import FIELDS
from .service import fetch_rows

def main():
    ap = argparse.ArgumentParser(description="Fetch GitHub PRs to CSV")
    ap.add_argument("--owner", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--state", choices=["open","closed","all"], default=None)
    ap.add_argument("--per-page", type=int, default=None)
    ap.add_argument("--target-count", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--sleep-between-pages", type=float, default=None)
    ap.add_argument("--out", default=None, help="Output CSV path (defaults to data/raw/<owner>_<repo>_<state>_<ts>.csv)")
    args = ap.parse_args()

    s = Settings()
    if args.owner: s.owner = args.owner
    if args.repo: s.repo = args.repo
    if args.state: s.state = args.state
    if args.per_page: s.per_page = args.per_page
    if args.target_count: s.target_count = args.target_count
    if args.max_pages: s.max_pages = args.max_pages
    if args.sleep_between_pages is not None: s.sleep_between_pages = args.sleep_between_pages

    # default output under data/raw/
    if args.out:
        out_path = Path(args.out)
    else:
        from datetime import datetime
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = Path("data/raw") / f"{s.owner}_{s.repo}_{s.state}_{ts}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    s.out_csv = str(out_path)

    rows = fetch_rows(s)
    write_csv(rows, s.out_csv, fieldnames=FIELDS)
    print(f"Written {len(rows)} rows to {s.out_csv}")

if __name__ == "__main__":
    main()

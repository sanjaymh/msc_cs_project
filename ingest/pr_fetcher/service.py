from .config import Settings
from .github import GitHubClient
from .extract import flatten_pr
from ingest.pr_fetcher.utility import clean_body, pr_is_complete

def fetch_rows(s: Settings):
    client = GitHubClient(s)
    collected = []

    for pr in client.list_pulls(s.owner, s.repo, s.state, s.per_page):
        # keep only merged when closed
        if s.state == "closed" and pr.get("merged_at") is None:
            continue
        # fill merged_by if missing
        if (pr.get("merged_at") is not None) and not (pr.get("merged_by") or {}).get("login"):
            full = client.get_pull(s.owner, s.repo, pr.get("number"))
            pr["merged_by"] = full.get("merged_by")

        if not pr_is_complete(pr):
            continue

        collected.append(pr)
        if s.state == "closed" and len(collected) >= s.target_count:
            break

    rows = []
    for pr in collected:
        flat = flatten_pr(pr)
        flat["body"] = clean_body(flat["body"])
        rows.append(flat)
    return rows

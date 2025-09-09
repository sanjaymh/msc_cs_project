from typing import Dict, Any

FIELDS = [
    "id","number","title","author","body","created_at","state","labels","reviewers"
]

def _join(arr, key):
    return ";".join(sorted({(x or {}).get(key, "") for x in (arr or []) if (x or {}).get(key)}))

def flatten_pr(pr: Dict[str, Any]) -> Dict[str, Any]:
    author = (pr.get("user") or {}).get("login", "")
    labels = _join(pr.get("labels", []), "name")
    reviewers = (pr.get("merged_by") or {}).get("login", "")
    return {
        "id": pr.get("id"),
        "number": pr.get("number"),
        "title": (pr.get("title") or "").replace("\n", " ").strip(),
        "author": author,
        "body": pr.get("body") or "",
        "created_at": pr.get("created_at") or "",
        "state": pr.get("state") or "",
        "labels": labels,
        "reviewers": reviewers,
    }

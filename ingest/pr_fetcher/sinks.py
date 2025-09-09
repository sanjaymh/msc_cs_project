import csv
from pathlib import Path
from typing import Iterable, Dict, Any, List

def write_csv(rows: Iterable[Dict[str, Any]], path: str, fieldnames: List[str]):
    rows = list(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r = {k: ("" if v is None else v) for k, v in r.items()}
            w.writerow(r)

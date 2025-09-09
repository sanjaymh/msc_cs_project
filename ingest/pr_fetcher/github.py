from __future__ import annotations
import time
import requests
from typing import Dict, Any, Iterator
from .config import Settings

API_BASE = "https://api.github.com"

class RateLimit(Exception):
    pass

class GitHubClient:
    def __init__(self, settings: Settings):
        self.s = requests.Session()
        self.s.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if settings.token:
            self.s.headers["Authorization"] = f"Bearer {settings.token}"
        self.settings = settings

    def _sleep_until_reset(self, resp: requests.Response):
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset:
            reset_s = max(0, int(reset) - int(time.time()) + 1)
            time.sleep(reset_s)
        else:
            time.sleep(5)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        for attempt in range(5):
            resp = self.s.request(method, url, timeout=30, **kwargs)
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                self._sleep_until_reset(resp)
                continue
            if 500 <= resp.status_code < 600:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp
        resp.raise_for_status()
        return resp

    def list_pulls(self, owner: str, repo: str, state: str, per_page: int) -> Iterator[Dict[str, Any]]:
        page = 1
        seen_empty_pages = 0
        while page <= self.settings.max_pages:
            params = {"state": state, "per_page": per_page, "page": page, "sort": "updated", "direction": "desc"}
            url = f"{API_BASE}/repos/{owner}/{repo}/pulls"
            resp = self._request("GET", url, params=params)
            items = resp.json() or []
            if not items:
                seen_empty_pages += 1
                if seen_empty_pages >= 2:
                    break
            for it in items:
                yield it
            page += 1
            if self.settings.sleep_between_pages:
                time.sleep(self.settings.sleep_between_pages)

    def get_pull(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        url = f"{API_BASE}/repos/{owner}/{repo}/pulls/{number}"
        return self._request("GET", url).json()

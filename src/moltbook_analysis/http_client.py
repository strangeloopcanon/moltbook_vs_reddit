from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpClient:
    user_agent: str = "moltbook_analysis/0.1 (+https://www.moltbook.com)"
    timeout_s: float = 30.0
    max_retries: int = 3
    backoff_s: float = 1.0

    def get_json(self, url: str) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = resp.read()
                return json.loads(raw.decode("utf-8"))
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
            ) as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.backoff_s * (2**attempt))
        raise RuntimeError("unreachable") from last_err

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


def _json_loads_maybe(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def parse_havelock_sse(payload: str) -> dict[str, Any]:
    """
    Parse the SSE response body from /gradio_api/call/analyze/{event_id}.

    Havelock may return:
      - data: {...} (object)
      - data: [{...}] (list with a single object)
      - data: ["## Results:", {...}] (list with header + object)
    """
    for line in payload.splitlines():
        if not line.startswith("data: "):
            continue
        raw = line[len("data: ") :].strip()
        if not raw:
            continue
        parsed = _json_loads_maybe(raw)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], dict):
                return parsed[0]
            if len(parsed) >= 2 and isinstance(parsed[1], dict):
                return parsed[1]
    raise ValueError("No parseable 'data: ' line found in SSE payload")


@dataclass(frozen=True)
class HavelockClient:
    base_url: str = "https://thestalwart-havelock-demo.hf.space"
    timeout_s: float = 60.0
    max_retries: int = 3
    backoff_s: float = 1.0
    user_agent: str = "moltbook_analysis/0.1 (havelock.ai)"

    def analyze(self, *, text: str, include_sentences: bool = False) -> dict[str, Any]:
        # Step 1: submit
        submit_url = f"{self.base_url}/gradio_api/call/analyze"
        event_id = self._post_json(
            submit_url,
            {"data": [text, bool(include_sentences)]},
        ).get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise RuntimeError("Havelock submit did not return event_id")

        # Step 2: fetch SSE result (usually completes immediately; retry just in case)
        result_url = f"{self.base_url}/gradio_api/call/analyze/{event_id}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._get_text(result_url)
                return parse_havelock_sse(payload)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.backoff_s * (2**attempt))

        raise RuntimeError("unreachable") from last_err

    def _post_json(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        parsed = _json_loads_maybe(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("Havelock submit returned non-JSON response")
        return parsed

    def _get_text(self, url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")

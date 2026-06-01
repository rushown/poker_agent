#!/usr/bin/env python3
"""DeepSeek subagent helper — called by Claude Code for heavy research tasks.

Usage:
  python scripts/ds_query.py "your prompt here"
  python scripts/ds_query.py --reasoner "complex analysis prompt"
  python scripts/ds_query.py --system "custom system prompt" "user prompt"
  python scripts/ds_query.py --no-cache "fresh query, skip cache"

Returns JSON or plain text to stdout. Errors go to stderr.
Exit 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
CACHE_DIR = Path(".deepseek_cache")
CACHE_TTL = 3600

DEFAULT_SYSTEM = (
    "You are a fast research assistant. Be concise. "
    "When asked for JSON, return only valid JSON with no markdown fences. "
    "When asked for text, be direct and skip preamble."
)


def cache_key(prompt: str, model: str, system: str) -> str:
    blob = f"{model}:{system}:{prompt}"
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def cache_get(key: str) -> str | None:
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if time.time() - data.get("ts", 0) > CACHE_TTL:
            path.unlink(missing_ok=True)
            return None
        return data["content"]
    except Exception:
        return None


def cache_set(key: str, content: str) -> None:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    try:
        path.write_text(json.dumps({"ts": time.time(), "content": content}))
    except Exception:
        pass


def call_api(prompt: str, model: str, system: str, timeout: int = 30) -> str:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }).encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:200]}", file=sys.stderr)
            if attempt == 2:
                sys.exit(1)
            time.sleep(1.5 * (attempt + 1))
        except Exception as e:
            print(f"Error (attempt {attempt + 1}): {e}", file=sys.stderr)
            if attempt == 2:
                sys.exit(1)
            time.sleep(1.5 * (attempt + 1))

    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek subagent query")
    parser.add_argument("prompt", help="Prompt to send")
    parser.add_argument("--reasoner", action="store_true", help="Use deepseek-reasoner for complex analysis")
    parser.add_argument("--system", default=DEFAULT_SYSTEM, help="System prompt override")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache, always call API")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    args = parser.parse_args()

    model = "deepseek-reasoner" if args.reasoner else "deepseek-chat"
    key = cache_key(args.prompt, model, args.system)

    if not args.no_cache:
        cached = cache_get(key)
        if cached is not None:
            print(cached)
            return

    result = call_api(args.prompt, model, args.system, timeout=args.timeout)
    if not args.no_cache:
        cache_set(key, result)

    print(result)


if __name__ == "__main__":
    main()

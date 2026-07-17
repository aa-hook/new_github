from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: float = 120.0,
) -> dict:
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"invalid service response: {result}")
    return result


def wait_until_ready(base_url: str, *, timeout: float) -> dict:
    deadline = time.monotonic() + float(timeout)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return request_json(f"{base_url.rstrip('/')}/health", timeout=2.0)
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(f"rank_v11 service was not ready in {timeout}s: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit one image to a rank_v11 service.")
    parser.add_argument("image")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--mode", choices=["fast", "accurate"], default="accurate")
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--json", action="store_true", dest="print_json")
    args = parser.parse_args()

    image = Path(args.image).expanduser().resolve()
    if not image.is_file():
        print(f"image not found: {image}", file=sys.stderr)
        return 2
    wait_until_ready(args.url, timeout=args.ready_timeout)
    result = request_json(
        f"{args.url.rstrip('/')}/solve",
        payload={"image": str(image), "mode": args.mode},
        timeout=args.request_timeout,
    )
    if args.print_json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(int(result["answer_index"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

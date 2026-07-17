from __future__ import annotations

import argparse
import json

from solve import wait_until_ready


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for a rank_v11 service to be ready.")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    result = wait_until_ready(args.url, timeout=args.timeout)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

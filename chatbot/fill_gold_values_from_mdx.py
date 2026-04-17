from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Allow running as a top-level script from any directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from OLAP.OLAP_connection import build_default_awdw, build_from_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute mdx_query for each question and fill gold_value with scalar result."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input questions JSON file (list of objects with mdx_query).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file path. Defaults to --input (in-place update).",
    )
    return parser.parse_args()


def extract_first_numeric_value(rows: list[tuple[Any, ...]]) -> float:
    numeric_token = re.compile(r"[-+]?\d+(?:[\s,]\d{3})*(?:\.\d+)?")

    for row in rows:
        for cell in row:
            if cell is None or isinstance(cell, bool):
                continue

            if isinstance(cell, (int, float)):
                value = float(cell)
                if not (math.isnan(value) or math.isinf(value)):
                    return value

            try:
                value = float(cell)
                if not (math.isnan(value) or math.isinf(value)):
                    return value
            except Exception:  # noqa: BLE001
                pass

            for attr in ("value", "Value"):
                wrapped = getattr(cell, attr, None)
                if wrapped is None:
                    continue
                try:
                    value = float(wrapped)
                    if not (math.isnan(value) or math.isinf(value)):
                        return value
                except Exception:  # noqa: BLE001
                    continue

            stripped = str(cell).strip().replace(",", "")
            try:
                value = float(stripped)
                if not (math.isnan(value) or math.isinf(value)):
                    return value
            except ValueError:
                match = numeric_token.search(str(cell))
                if not match:
                    continue
                token = match.group(0).replace(" ", "").replace(",", "")
                try:
                    value = float(token)
                except ValueError:
                    continue
                if not (math.isnan(value) or math.isinf(value)):
                    return value

    raise ValueError("No numeric value found in MDX result set")


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input

    items = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list")

    try:
        conn = build_from_env()
        print("[olap] Connected using environment configuration")
    except Exception:
        conn = build_default_awdw()
        print("[olap] Connected using default Adventure Works configuration")

    updated = 0
    failed = 0

    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue

        mdx_query = str(item.get("mdx_query", "")).strip()
        if not mdx_query:
            item["gold_value"] = None
            item["gold_value_error"] = "Missing mdx_query"
            failed += 1
            continue

        try:
            rows, _headers = conn.execute_mdx(mdx_query)
            value = extract_first_numeric_value(rows)
            item["gold_value"] = value
            item.pop("gold_value_error", None)
            updated += 1
            print(f"[{idx}] {item.get('id', idx)} -> gold_value={value}")
        except Exception as exc:  # noqa: BLE001
            item["gold_value"] = None
            item["gold_value_error"] = str(exc)
            failed += 1
            print(f"[{idx}] {item.get('id', idx)} -> ERROR: {exc}")

    output_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Updated: {updated}, Failed: {failed}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check local prerequisites and optional Google API access for scraping."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


DEFAULT_PUBLIC_DRIVE_FILE_ID = "1FNlY8pZWBLhkDlLHv9k4Q_PENkw9dMMP"


def check_import(module: str) -> tuple[bool, str]:
    try:
        imported = __import__(module)
        version = getattr(imported, "__version__", "ok")
        return True, str(version)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def check_drive_api(api_key: str, file_id: str, timeout: float = 20.0) -> tuple[bool, str]:
    query = urlencode({"fields": "id,name,mimeType,size", "key": api_key})
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?{query}"
    request = Request(url, headers={"User-Agent": "cp-detector-scraper/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return True, f"{body.get('name')} ({body.get('mimeType')})"
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8")
        except Exception:
            body = ""
        return False, f"HTTP {exc.code}: {body[:300]}"
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--drive-api-key",
        default=os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        help="Defaults to GOOGLE_DRIVE_API_KEY, then GOOGLE_API_KEY.",
    )
    parser.add_argument("--drive-file-id", default=DEFAULT_PUBLIC_DRIVE_FILE_ID)
    args = parser.parse_args()

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")

    ok = True
    for module in ["numpy", "PIL", "cv2", "pypdfium2"]:
        passed, detail = check_import(module)
        print(f"{module}: {'ok' if passed else 'missing'} ({detail})")
        ok = ok and passed

    if args.drive_api_key:
        passed, detail = check_drive_api(args.drive_api_key, args.drive_file_id)
        print(f"Google Drive API: {'ok' if passed else 'failed'} ({detail})")
        ok = ok and passed
    else:
        print("Google Drive API: skipped (set GOOGLE_DRIVE_API_KEY or GOOGLE_API_KEY)")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

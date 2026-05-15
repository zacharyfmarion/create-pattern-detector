"""Download helpers for real CP scraping."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class DownloadResult:
    status: str
    local_path: Path | None = None
    attempted_url: str | None = None
    content_type: str | None = None
    bytes_written: int = 0
    error: str | None = None


def download_url(
    url: str,
    destination: str | Path,
    timeout: float = 60.0,
    retries: int = 2,
    delay: float = 0.0,
) -> DownloadResult:
    """Download a URL to destination with small retry/backoff handling."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "cp-detector-scraper/0.1"}
    last_error: str | None = None

    for attempt in range(retries + 1):
        if delay and attempt > 0:
            time.sleep(delay * attempt)
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
                partial = destination.with_name(f"{destination.name}.part")
                partial.write_bytes(data)
                partial.replace(destination)
                return DownloadResult(
                    status="downloaded",
                    local_path=destination,
                    attempted_url=url,
                    content_type=response.headers.get("Content-Type"),
                    bytes_written=len(data),
                )
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)

    return DownloadResult(status="failed", attempted_url=url, error=last_error)


def _redact_api_key(url: str) -> str:
    prefix, separator, _ = url.partition("key=")
    if separator:
        return f"{prefix}key=REDACTED"
    return url


def drive_api_download_url(file_id: str, api_key: str) -> str:
    """Build a Google Drive API media-download URL for a public file."""
    query = urlencode({"alt": "media", "key": api_key})
    return f"https://www.googleapis.com/drive/v3/files/{file_id}?{query}"


def drive_download_urls(file_id: str, category: str, image_download_size: int = 1024) -> list[str]:
    """Return best-effort public Google Drive URLs for a file id."""
    usercontent = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    direct = f"https://drive.google.com/uc?export=download&id={file_id}"
    if category == "image":
        urls: list[str] = []
        if image_download_size > 0:
            urls.append(f"https://drive.google.com/thumbnail?id={file_id}&sz=w{image_download_size}")
        urls.extend([usercontent, direct])
        return urls
    return [usercontent, direct]


def download_drive_file(
    file_id: str,
    category: str,
    destination: str | Path,
    api_key: str | None = None,
    prefer_api: bool = True,
    image_download_size: int = 1024,
    timeout: float = 60.0,
    retries: int = 2,
    delay: float = 0.0,
) -> DownloadResult:
    """Download a public Google Drive file, trying fallbacks where useful."""
    failures: list[str] = []
    api_key = api_key or os.environ.get("GOOGLE_DRIVE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    urls: list[tuple[str, str]] = []
    if prefer_api and api_key:
        api_url = drive_api_download_url(file_id, api_key)
        urls.append((api_url, _redact_api_key(api_url)))
    urls.extend((url, url) for url in drive_download_urls(file_id, category, image_download_size=image_download_size))

    for url, manifest_url in urls:
        result = download_url(url, destination, timeout=timeout, retries=retries, delay=delay)
        if result.status == "downloaded" and result.bytes_written > 0:
            result.attempted_url = manifest_url
            return result
        failures.append(f"{manifest_url}: {result.error or 'empty response'}")
    return DownloadResult(status="failed", error="; ".join(failures))

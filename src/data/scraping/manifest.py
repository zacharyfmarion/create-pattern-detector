"""Manifest and filesystem helpers for real CP scraping runs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable


SCRAPED_SUBDIRS = (
    "source_snapshots",
    "raw_assets",
    "crops",
    "native",
    "native/raw",
    "native/converted_fold",
    "manifests",
    "review",
    "review/debug",
)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
    ".avif",
    ".heic",
}
NATIVE_EXTS = {".fold", ".cp", ".ori", ".opx", ".svg"}
PDF_EXTS = {".pdf"}

IMAGE_MIMES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/gif",
    "image/webp",
    "image/tiff",
    "image/avif",
    "image/heif",
}
NATIVE_MIMES = {
    "text/cp",
    "text/origami",
    "text/xml",
    "image/svg+xml",
}


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for manifests."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def make_run_id(prefix: str) -> str:
    """Create a compact, sortable scrape run id."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}-{stamp}"


def ensure_output_tree(output_root: str | Path) -> Path:
    """Create the standard scraped-data output directory tree."""
    root = Path(output_root)
    for subdir in SCRAPED_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def sanitize_slug(value: str, max_len: int = 96) -> str:
    """Make a stable filesystem-safe slug while preserving useful words."""
    value = value.strip().replace("&", " and ")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    if not value:
        value = "unnamed"
    return value[:max_len].rstrip("-._") or "unnamed"


def classify_asset(filename: str, mime_type: str | None = None) -> str:
    """Classify an asset into image, pdf, native, or other."""
    mime_type = (mime_type or "").lower()
    suffix = Path(filename).suffix.lower()
    if suffix in NATIVE_EXTS or mime_type in NATIVE_MIMES:
        return "native"
    if suffix in PDF_EXTS or mime_type == "application/pdf":
        return "pdf"
    if suffix in IMAGE_EXTS or mime_type in IMAGE_MIMES:
        return "image"
    return "other"


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_path(directory: str | Path, filename: str) -> Path:
    """Return a non-conflicting path in directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_slug(Path(filename).stem)
    suffix = Path(filename).suffix.lower()
    candidate = directory / f"{safe_name}{suffix}"
    if not candidate.exists():
        return candidate
    for i in range(1, 10_000):
        candidate = directory / f"{safe_name}-{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique path for {filename} in {directory}")


def relative_to_root(path: str | Path | None, root: str | Path) -> str | None:
    """Return a posix relative path for manifest readability."""
    if path is None:
        return None
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and paths into JSON-safe values."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(value), f, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: str | Path, records: Iterable[Any]) -> int:
    """Write records to JSONL and return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(to_jsonable(record), sort_keys=True))
            f.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest into dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def append_jsonl(path: str | Path, record: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(record), sort_keys=True))
        f.write("\n")

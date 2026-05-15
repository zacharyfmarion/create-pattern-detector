"""Source-specific parsers for real crease pattern collections."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

from .manifest import classify_asset


CPOOGLE_MODELS_URL = "https://bogdanthegeek.github.io/CPoogle/models.json"
CPOOGLE_PAGE_URL = "https://bogdanthegeek.github.io/CPoogle/?q="
OBB_URL = "https://www.obb.design/c"


@dataclass(frozen=True)
class SourceAssetCandidate:
    """A downloadable source asset discovered from a scrape source."""

    source: str
    asset_id: str
    filename: str
    category: str
    mime_type: str | None
    source_url: str
    download_url: str | None = None
    drive_file_id: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    author_name: str | None = None
    author_id: str | None = None
    priority: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)


def fetch_text(url: str, timeout: float = 30.0) -> str:
    request = Request(url, headers={"User-Agent": "cp-detector-scraper/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_cpoogle_models(source_json: str | Path | None = None) -> list[dict[str, Any]]:
    """Load CPOogle's static models.json from disk or the live URL."""
    if source_json:
        with Path(source_json).open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(fetch_text(CPOOGLE_MODELS_URL))


def iter_cpoogle_assets(
    models: Iterable[dict[str, Any]],
    include_images: bool = True,
    include_pdfs: bool = True,
    include_native: bool = True,
    include_other: bool = False,
    square_first: bool = True,
) -> list[SourceAssetCandidate]:
    """Enumerate CPOogle files as downloadable asset candidates."""
    candidates: list[SourceAssetCandidate] = []

    for model in models:
        author = model.get("author") or {}
        model_id = model.get("id")
        model_name = model.get("name") or "unknown-model"
        paper_shapes = model.get("paper_shape") or []
        design_styles = model.get("design_style") or []
        files = model.get("files") or []

        if not files and model.get("id"):
            files = [
                {
                    "id": model["id"],
                    "name": model.get("name", model["id"]),
                    "mimeType": model.get("type"),
                }
            ]

        for file_info in files:
            filename = file_info.get("name") or file_info.get("id") or "unnamed"
            mime_type = file_info.get("mimeType")
            category = classify_asset(filename, mime_type)
            if category == "image" and not include_images:
                continue
            if category == "pdf" and not include_pdfs:
                continue
            if category == "native" and not include_native:
                continue
            if category == "other" and not include_other:
                continue

            file_id = file_info.get("id")
            if not file_id:
                continue

            is_square = "Square" in paper_shapes
            priority = 0
            if square_first and not is_square:
                priority += 50
            priority += {"native": 0, "image": 10, "pdf": 20, "other": 40}.get(category, 40)

            candidates.append(
                SourceAssetCandidate(
                    source="cpoogle",
                    asset_id=f"cpoogle:{file_id}",
                    filename=filename,
                    category=category,
                    mime_type=mime_type,
                    source_url=f"https://drive.google.com/file/d/{file_id}",
                    drive_file_id=file_id,
                    model_id=model_id,
                    model_name=model_name,
                    author_name=author.get("name"),
                    author_id=author.get("id"),
                    priority=priority,
                    metadata={
                        "paper_shape": paper_shapes,
                        "design_style": design_styles,
                        "tags": model.get("tags") or [],
                        "has_pd": model.get("has_pd"),
                        "model_type": model.get("type"),
                        "source_page": CPOOGLE_PAGE_URL,
                    },
                )
            )

    candidates.sort(key=lambda c: (c.priority, c.model_name or "", c.filename))
    return candidates


@dataclass
class OBBItem:
    title: str
    description: str
    designed: str | None
    url: str | None
    images: list[str]


class _OBBHTMLParser(HTMLParser):
    """Small Webflow parser for the OBB crease-pattern gallery."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[OBBItem] = []
        self.quick_links: dict[str, str] = {}
        self._current: dict[str, Any] | None = None
        self._div_depth = 0
        self._capture: str | None = None
        self._capture_chunks: list[str] = []
        self._last_quick_href: str | None = None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k: v or "" for k, v in attrs_list}
        class_attr = attrs.get("class", "")

        if tag == "a" and "link-4" in class_attr:
            self._last_quick_href = attrs.get("href")
            self._capture = "quick_link_text"
            self._capture_chunks = []
            return

        if tag == "div" and self._current is not None:
            self._div_depth += 1

        if tag == "div" and "collection-item" in class_attr and self._current is None:
            self._current = {
                "title": "",
                "description": "",
                "designed_values": [],
                "images": [],
            }
            self._div_depth = 1
            return

        if self._current is None:
            return

        if tag == "h1" and "heading-19" in class_attr:
            self._capture = "title"
            self._capture_chunks = []
        elif tag == "div" and "text-block-14" in class_attr:
            self._capture = "description"
            self._capture_chunks = []
        elif tag == "div" and "text-block-15" in class_attr:
            self._capture = "designed"
            self._capture_chunks = []
        elif tag == "img":
            src = attrs.get("src")
            if src:
                self._current["images"].append(src)

    def handle_data(self, data: str) -> None:
        if self._capture:
            self._capture_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture and (
            (tag == "a" and self._capture == "quick_link_text")
            or (tag in {"h1", "div"} and self._capture != "quick_link_text")
        ):
            text = " ".join("".join(self._capture_chunks).split())
            if self._capture == "quick_link_text":
                if text and self._last_quick_href:
                    self.quick_links[text] = self._last_quick_href
                self._last_quick_href = None
            elif self._current is not None:
                if self._capture == "designed":
                    if text and text.lower() != "designed:":
                        self._current["designed_values"].append(text)
                else:
                    self._current[self._capture] = text
            self._capture = None
            self._capture_chunks = []

        if tag == "div" and self._current is not None:
            self._div_depth -= 1
            if self._div_depth <= 0:
                title = self._current["title"]
                designed_values = self._current["designed_values"]
                self.items.append(
                    OBBItem(
                        title=title,
                        description=self._current["description"],
                        designed=designed_values[-1] if designed_values else None,
                        url=self.quick_links.get(title),
                        images=list(dict.fromkeys(self._current["images"])),
                    )
                )
                self._current = None
                self._div_depth = 0


def parse_obb_html(html: str) -> list[OBBItem]:
    parser = _OBBHTMLParser()
    parser.feed(html)
    return [item for item in parser.items if item.title and item.images]


def load_obb_html(source_html: str | Path | None = None) -> str:
    if source_html:
        return Path(source_html).read_text(encoding="utf-8")
    return fetch_text(OBB_URL)


def iter_obb_assets(items: Iterable[OBBItem], include_images: bool = True) -> list[SourceAssetCandidate]:
    """Enumerate OBB gallery images as asset candidates."""
    if not include_images:
        return []

    candidates: list[SourceAssetCandidate] = []
    for item_idx, item in enumerate(items):
        for image_idx, image_url in enumerate(item.images):
            filename = image_url.split("/")[-1].split("?")[0] or f"obb-{item_idx}-{image_idx}.png"
            category = classify_asset(filename, "image/png")
            if category != "image":
                continue

            filename_lower = filename.lower()
            cpish_name = any(
                token in filename_lower
                for token in ("cp", "crease", "pattern", "grid", "base", "full")
            )
            priority = 0 if image_idx == 0 or cpish_name else 20
            candidates.append(
                SourceAssetCandidate(
                    source="obb",
                    asset_id=f"obb:{item_idx}:{image_idx}",
                    filename=filename,
                    category="image",
                    mime_type=None,
                    source_url=image_url,
                    download_url=image_url,
                    model_id=item.url,
                    model_name=item.title,
                    author_name="Origami By Boice",
                    priority=priority,
                    metadata={
                        "designed": item.designed,
                        "description": item.description,
                        "model_page": f"https://www.obb.design{item.url}" if item.url else OBB_URL,
                        "source_page": OBB_URL,
                        "image_index": image_idx,
                        "filename_cp_hint": cpish_name,
                    },
                )
            )

    candidates.sort(key=lambda c: (c.priority, c.model_name or "", c.filename))
    return candidates

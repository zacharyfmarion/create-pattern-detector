"""Optional Gemini-based CP crop classification and cost estimation."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import math
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
GEMINI_TOKENS_URL = "https://ai.google.dev/gemini-api/docs/tokens"


@dataclass(frozen=True)
class GeminiModelPricing:
    """Approximate Gemini API text/image-token pricing per 1M tokens."""

    model: str
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float | None = None
    note: str = ""


# Keep these conservative and visible. Run with --gemini-pricing-url to see the
# current source before a large scrape.
DEFAULT_PRICING: dict[str, GeminiModelPricing] = {
    "gemini-2.5-flash-lite": GeminiModelPricing(
        model="gemini-2.5-flash-lite",
        input_per_million=0.10,
        output_per_million=0.40,
        note="low-cost crop classifier default",
    ),
    "gemini-2.5-flash": GeminiModelPricing(
        model="gemini-2.5-flash",
        input_per_million=0.30,
        output_per_million=2.50,
        note="stronger but more expensive classifier",
    ),
}


@dataclass
class GeminiCostEstimate:
    model: str
    images: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    pricing_url: str = GEMINI_PRICING_URL
    tokens_url: str = GEMINI_TOKENS_URL


@dataclass
class GeminiClassification:
    status: str
    is_crease_pattern: bool | None
    confidence: float | None
    label: str | None
    reason: str | None
    raw_response: dict[str, Any] | None = None
    error: str | None = None


def estimate_image_tokens(width: int, height: int, tile_size: int = 768, tokens_per_tile: int = 258) -> int:
    """Estimate Gemini image tokens using the documented tiled-image mental model.

    The Gemini docs describe small images as a fixed token cost and larger images
    as tiles. This estimate intentionally rounds up so scrape budgeting is not
    optimistic.
    """
    if width <= 0 or height <= 0:
        return tokens_per_tile
    tiles = max(1, math.ceil(width / tile_size) * math.ceil(height / tile_size))
    return tiles * tokens_per_tile


def estimate_gemini_cost_for_images(
    image_paths: list[str | Path],
    model: str = "gemini-2.5-flash-lite",
    prompt_tokens_per_image: int = 120,
    output_tokens_per_image: int = 80,
) -> GeminiCostEstimate:
    pricing = DEFAULT_PRICING.get(model)
    if pricing is None:
        raise ValueError(f"No built-in pricing for {model}; add it to DEFAULT_PRICING first")

    input_tokens = 0
    for image_path in image_paths:
        with Image.open(image_path) as image:
            width, height = image.size
        input_tokens += estimate_image_tokens(width, height) + prompt_tokens_per_image
    output_tokens = len(image_paths) * output_tokens_per_image
    cost = (
        input_tokens * pricing.input_per_million
        + output_tokens * pricing.output_per_million
    ) / 1_000_000.0
    return GeminiCostEstimate(
        model=model,
        images=len(image_paths),
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        estimated_cost_usd=float(cost),
    )


class GeminiCPClassifier:
    """Tiny REST client for optional Gemini CP classification."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        timeout: float = 45.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY to use Gemini classification")

    @staticmethod
    def prompt() -> str:
        return (
            "You are filtering a dataset of origami crease-pattern examples. "
            "Classify the image crop. A positive example must be one clean, "
            "non-hand-drawn origami crease pattern only: a complete straight-line "
            "network on a square or near-square sheet, either light or dark mode. "
            "Reject folded-model photos, finished models, multi-panel folding "
            "instructions, numbered step diagrams, transition diagrams, partial "
            "fold diagrams, CP-adjacent explanatory diagrams, images where text or "
            "legends dominate, sketches, hand-drawn patterns, noisy photos, logos, "
            "or any image that is not itself a standalone crease pattern. "
            "If the image shows several panels or a sequence of folds, reject it "
            "even if some panels contain crease lines. Return only JSON with keys: "
            "is_crease_pattern boolean, label string, confidence number 0-1, reason string."
        )

    def classify_image(self, image_path: str | Path) -> GeminiClassification:
        image_path = Path(image_path)
        mime_type = "image/png"
        if image_path.suffix.lower() in {".jpg", ".jpeg"}:
            mime_type = "image/jpeg"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt()},
                        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 160,
                "responseMimeType": "application/json",
            },
        }
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return GeminiClassification(
                status="error",
                is_crease_pattern=None,
                confidence=None,
                label=None,
                reason=None,
                error=str(exc),
            )

        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(text)
            return GeminiClassification(
                status="classified",
                is_crease_pattern=bool(data.get("is_crease_pattern")),
                confidence=float(data.get("confidence", 0.0)),
                label=str(data.get("label", "")),
                reason=str(data.get("reason", "")),
                raw_response=body,
            )
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return GeminiClassification(
                status="error",
                is_crease_pattern=None,
                confidence=None,
                label=None,
                reason=None,
                raw_response=body,
                error=f"response_parse_failed: {exc}",
            )

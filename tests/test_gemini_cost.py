import json

from PIL import Image

from src.data.scraping.gemini_classifier import (
    GeminiCPClassifier,
    estimate_gemini_cost_for_images,
    estimate_image_tokens,
)


def test_estimate_image_tokens_rounds_up_by_tiles():
    assert estimate_image_tokens(256, 256) == 258
    assert estimate_image_tokens(769, 256) == 516
    assert estimate_image_tokens(1024, 1024) == 1032


def test_estimate_gemini_cost_for_images(tmp_path):
    image_path = tmp_path / "cp.png"
    Image.new("RGB", (512, 512), "white").save(image_path)

    estimate = estimate_gemini_cost_for_images([image_path])

    assert estimate.images == 1
    assert estimate.input_tokens > 0
    assert 0 < estimate.estimated_cost_usd < 0.001


def test_gemini_prompt_rejects_step_diagrams():
    prompt = GeminiCPClassifier.prompt()

    assert "numbered step diagrams" in prompt
    assert "sequence of folds" in prompt
    assert "standalone crease pattern" in prompt


def test_gemini_classifier_accepts_single_object_list_response(monkeypatch, tmp_path):
    image_path = tmp_path / "cp.png"
    Image.new("RGB", (64, 64), "white").save(image_path)

    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                [
                                    {
                                        "is_crease_pattern": True,
                                        "confidence": 0.95,
                                        "label": "clean",
                                        "reason": "standalone crease pattern",
                                    }
                                ]
                            )
                        }
                    ]
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response_body).encode("utf-8")

    monkeypatch.setattr("src.data.scraping.gemini_classifier.urlopen", lambda *args, **kwargs: FakeResponse())

    classification = GeminiCPClassifier(api_key="fake-key").classify_image(image_path)

    assert classification.status == "classified"
    assert classification.is_crease_pattern is True
    assert classification.confidence == 0.95

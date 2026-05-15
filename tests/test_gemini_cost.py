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

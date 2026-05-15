from pathlib import Path

from PIL import Image

from src.data.scraping.gemini_classifier import GeminiClassification
from src.data.scraping.manifest import read_jsonl, write_jsonl
from src.data.scraping.workflow import (
    build_final_real_cp_dataset,
    classify_existing_crops,
    select_refetch_candidates,
)


def test_select_refetch_candidates_keeps_screened_images_only(tmp_path):
    manifests = tmp_path / "manifests"
    run_id = "screen-run"
    write_jsonl(
        manifests / f"cpoogle_candidates_{run_id}.jsonl",
        [
            {
                "asset_id": "cpoogle:image",
                "filename": "image.png",
                "category": "image",
                "source": "cpoogle",
                "source_url": "https://drive.google.com/file/d/image",
                "drive_file_id": "image",
                "priority": 10,
                "metadata": {"paper_shape": ["Square"]},
            },
            {
                "asset_id": "cpoogle:pdf",
                "filename": "page.pdf",
                "category": "pdf",
                "source": "cpoogle",
                "source_url": "https://drive.google.com/file/d/pdf",
                "drive_file_id": "pdf",
                "priority": 20,
            },
            {
                "asset_id": "cpoogle:rejected",
                "filename": "bad.png",
                "category": "image",
                "source": "cpoogle",
                "source_url": "https://drive.google.com/file/d/rejected",
                "drive_file_id": "rejected",
                "priority": 10,
            },
        ],
    )
    write_jsonl(
        manifests / f"cpoogle_crops_{run_id}.jsonl",
        [
            {"asset_id": "cpoogle:image", "status": "review", "crop_path": "crop.png", "perceptual_hash": "aa"},
            {"asset_id": "cpoogle:pdf", "status": "review", "crop_path": "pdf.png"},
            {"asset_id": "cpoogle:rejected", "status": "rejected"},
        ],
    )

    candidates = select_refetch_candidates(tmp_path, run_id)

    assert [candidate.asset_id for candidate in candidates] == ["cpoogle:image"]
    assert candidates[0].metadata["screening_run_id"] == run_id
    assert candidates[0].metadata["screening_statuses"] == ["review"]


def test_classify_existing_crops_writes_merged_manifest(monkeypatch, tmp_path):
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (512, 512), "white").save(image_path)
    crop_manifest = tmp_path / "crops.jsonl"
    output_manifest = tmp_path / "classified.jsonl"
    write_jsonl(
        crop_manifest,
        [
            {
                "asset_id": "asset-1",
                "crop_path": image_path.as_posix(),
                "status": "review",
                "needs_review": True,
                "reasons": [],
            }
        ],
    )

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify_image(self, path: Path) -> GeminiClassification:
            return GeminiClassification(
                status="classified",
                is_crease_pattern=True,
                confidence=0.92,
                label="clean_cp",
                reason="clean line network",
            )

    monkeypatch.setattr("src.data.scraping.workflow.GeminiCPClassifier", FakeClassifier)

    summary = classify_existing_crops(crop_manifest, output_manifest=output_manifest)
    rows = read_jsonl(output_manifest)

    assert summary.classified == 1
    assert summary.promoted == 1
    assert rows[0]["status"] == "accepted"
    assert rows[0]["needs_review"] is False
    assert rows[0]["gemini"]["confidence"] == 0.92


def test_classify_existing_crops_resumes_existing_output(monkeypatch, tmp_path):
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (512, 512), "white").save(image_path)
    crop_manifest = tmp_path / "crops.jsonl"
    output_manifest = tmp_path / "classified.jsonl"
    write_jsonl(
        crop_manifest,
        [
            {
                "asset_id": "asset-1",
                "crop_path": image_path.as_posix(),
                "status": "review",
                "needs_review": True,
                "reasons": [],
            },
            {
                "asset_id": "asset-2",
                "crop_path": image_path.as_posix(),
                "status": "review",
                "needs_review": True,
                "reasons": [],
            },
        ],
    )
    write_jsonl(
        output_manifest,
        [
            {
                "asset_id": "asset-1",
                "crop_path": image_path.as_posix(),
                "status": "accepted",
                "needs_review": False,
                "reasons": ["gemini_promoted"],
                "gemini": {"status": "classified", "confidence": 0.9},
            },
            {
                "asset_id": "asset-2",
                "crop_path": image_path.as_posix(),
                "status": "review",
                "needs_review": True,
                "reasons": [],
            },
        ],
    )
    calls = []

    class FakeClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify_image(self, path: Path) -> GeminiClassification:
            calls.append(path)
            return GeminiClassification(
                status="classified",
                is_crease_pattern=False,
                confidence=0.97,
                label="folded-model",
                reason="not a crease pattern",
            )

    monkeypatch.setattr("src.data.scraping.workflow.GeminiCPClassifier", FakeClassifier)

    summary = classify_existing_crops(
        crop_manifest,
        output_manifest=output_manifest,
        workers=2,
        checkpoint_interval=1,
    )
    rows = read_jsonl(output_manifest)

    assert len(calls) == 1
    assert summary.selected == 2
    assert summary.pending == 1
    assert summary.already_classified == 1
    assert summary.classified == 2
    assert summary.promoted == 1
    assert summary.rejected == 1
    assert rows[0]["status"] == "accepted"
    assert rows[1]["status"] == "rejected"


def test_build_final_dataset_prefers_originals_and_records_duplicates(tmp_path):
    scraped_root = tmp_path / "scraped"
    manifests = scraped_root / "manifests"
    final_root = scraped_root / "final"

    write_jsonl(
        manifests / "cpoogle_native_native-run.jsonl",
        [{"source_path": "raw/a.cp", "preserved_path": "native/raw/a.cp", "status": "converted"}],
    )
    write_jsonl(
        manifests / "cpoogle_originals_assets_orig-run.jsonl",
        [
            {
                "asset_id": "cpoogle:image",
                "source": "cpoogle",
                "category": "image",
                "source_url": "https://drive.google.com/file/d/image",
                "drive_file_id": "image",
                "local_path": "raw_assets/cpoogle/image/image.png",
                "download_variant": "original",
                "usage_policy": "internal-only",
            }
        ],
    )
    write_jsonl(
        manifests / "cpoogle_originals_crops_orig-run.jsonl",
        [
            {
                "asset_id": "cpoogle:image",
                "crop_path": "crops/cpoogle/image.png",
                "source_path": "raw_assets/cpoogle/image/image.png",
                "status": "accepted",
                "needs_review": False,
                "page_index": None,
                "cp_score": 0.95,
                "perceptual_hash": "same-phash",
                "content_sha256": "same-content",
            }
        ],
    )
    write_jsonl(
        manifests / "obb_assets_obb-run.jsonl",
        [
            {
                "asset_id": "obb:dupe",
                "source": "obb",
                "category": "image",
                "source_url": "https://cdn.example/dupe.png",
                "local_path": "raw_assets/obb/image/dupe.png",
                "download_variant": "original",
            }
        ],
    )
    write_jsonl(
        manifests / "obb_crops_obb-run.jsonl",
        [
            {
                "asset_id": "obb:dupe",
                "crop_path": "crops/obb/dupe.png",
                "source_path": "raw_assets/obb/image/dupe.png",
                "status": "review",
                "needs_review": True,
                "page_index": None,
                "cp_score": 0.70,
                "perceptual_hash": "same-phash",
                "content_sha256": "same-content",
            }
        ],
    )
    write_jsonl(
        manifests / "cpoogle_assets_screen-run.jsonl",
        [
            {
                "asset_id": "cpoogle:pdf",
                "source": "cpoogle",
                "category": "pdf",
                "source_url": "https://drive.google.com/file/d/pdf",
                "drive_file_id": "pdf",
                "local_path": "raw_assets/cpoogle/pdf/page.pdf",
            }
        ],
    )
    write_jsonl(
        manifests / "cpoogle_crops_screen-run.jsonl",
        [
            {
                "asset_id": "cpoogle:pdf",
                "crop_path": "crops/cpoogle/pdf.png",
                "source_path": "raw_assets/cpoogle/pdf/page.pdf",
                "status": "accepted",
                "needs_review": False,
                "page_index": 0,
                "cp_score": 0.90,
                "perceptual_hash": "pdf-phash",
                "content_sha256": "pdf-content",
            }
        ],
    )

    summary = build_final_real_cp_dataset(
        scraped_root=scraped_root,
        output_root=final_root,
        native_run="native-run",
        cpoogle_original_run="orig-run",
        obb_run="obb-run",
        cpoogle_screening_run="screen-run",
    )
    usable = read_jsonl(final_root / "final_usable_images.jsonl")
    rejects = read_jsonl(final_root / "final_rejects.jsonl")
    dedupe = read_jsonl(final_root / "dedupe_groups.jsonl")

    assert summary.native_rows == 1
    assert summary.usable_images == 2
    assert {row["asset_id"] for row in usable} == {"cpoogle:image", "cpoogle:pdf"}
    assert rejects[0]["status"] == "duplicate"
    assert dedupe[0]["kept_dataset_id"].startswith("cpoogle_original")

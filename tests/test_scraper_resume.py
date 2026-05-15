import pytest

from src.data.scraping.manifest import write_jsonl
from src.data.scraping.scraper import (
    _build_resume_cache,
    _candidate_output_name,
    _download_variant_for,
    _download_candidate,
)
from src.data.scraping.sources import SourceAssetCandidate


def _candidate() -> SourceAssetCandidate:
    return SourceAssetCandidate(
        source="cpoogle",
        asset_id="cpoogle:file-id",
        filename="pattern.png",
        category="image",
        mime_type="image/png",
        source_url="https://drive.google.com/file/d/file-id",
        drive_file_id="file-id",
        download_url="https://example.com/pattern.png",
        model_name="Test Model",
    )


def test_download_candidate_reuses_existing_raw_asset_without_manifest(monkeypatch, tmp_path):
    candidate = _candidate()
    raw_dir = tmp_path / "raw_assets" / candidate.source / candidate.category
    raw_dir.mkdir(parents=True)
    existing = raw_dir / _candidate_output_name(candidate, image_download_size=2048)
    existing.write_bytes(b"already here")

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called when raw asset exists")

    monkeypatch.setattr("src.data.scraping.scraper.download_drive_file", fail_download)

    record = _download_candidate(
        candidate,
        tmp_path,
        drive_api_key=None,
        drive_prefer_api=False,
        image_download_size=2048,
        resume=True,
        resume_cache={},
        timeout=1,
        retries=0,
        request_delay=0,
    )

    assert record["download_status"] == "reused"
    assert record["resume_source"] == "raw_asset"
    assert record["download_variant"] == "drive_thumbnail"
    assert record["image_download_size"] == 2048


def test_download_candidate_reuses_manifest_asset(monkeypatch, tmp_path):
    candidate = _candidate()
    cached_path = tmp_path / "raw_assets" / "cpoogle" / "image" / "custom-name.png"
    cached_path.parent.mkdir(parents=True)
    cached_path.write_bytes(b"manifest backed")
    write_jsonl(
        tmp_path / "manifests" / "cpoogle_assets_previous-run.jsonl",
        [
            {
                "run_id": "previous-run",
                "asset_id": candidate.asset_id,
                "category": candidate.category,
                "download_status": "downloaded",
                "download_variant": "drive_thumbnail",
                "image_download_size": 2048,
                "local_path": "raw_assets/cpoogle/image/custom-name.png",
                "attempted_url": "https://drive.google.com/thumbnail?id=file-id&sz=w2048",
            }
        ],
    )
    resume_cache = _build_resume_cache(tmp_path)

    def fail_download(*args, **kwargs):
        raise AssertionError("download should not be called when manifest asset exists")

    monkeypatch.setattr("src.data.scraping.scraper.download_drive_file", fail_download)

    record = _download_candidate(
        candidate,
        tmp_path,
        drive_api_key=None,
        drive_prefer_api=False,
        image_download_size=2048,
        resume=True,
        resume_cache=resume_cache,
        timeout=1,
        retries=0,
        request_delay=0,
    )

    assert record["download_status"] == "reused"
    assert record["resume_source"] == "manifest"
    assert record["resumed_from_run_id"] == "previous-run"
    assert record["local_path"] == "raw_assets/cpoogle/image/custom-name.png"


def test_force_download_ignores_resume_cache(monkeypatch, tmp_path):
    candidate = _candidate()
    raw_dir = tmp_path / "raw_assets" / candidate.source / candidate.category
    raw_dir.mkdir(parents=True)
    existing = raw_dir / _candidate_output_name(candidate, image_download_size=2048)
    existing.write_bytes(b"already here")

    def fail_download(*args, **kwargs):
        raise RuntimeError("forced network path")

    monkeypatch.setattr("src.data.scraping.scraper.download_drive_file", fail_download)

    with pytest.raises(RuntimeError, match="forced network path"):
        _download_candidate(
            candidate,
            tmp_path,
            drive_api_key=None,
            drive_prefer_api=False,
            image_download_size=2048,
            resume=False,
            resume_cache={},
            timeout=1,
            retries=0,
            request_delay=0,
        )


def test_non_drive_images_are_treated_as_originals_for_resume_keys():
    candidate = SourceAssetCandidate(
        source="obb",
        asset_id="obb:0:0",
        filename="pattern.png",
        category="image",
        mime_type="image/png",
        source_url="https://www.obb.design/c",
        download_url="https://cdn.example/pattern.png",
        drive_file_id=None,
    )

    assert _download_variant_for(candidate, image_download_size=2048) == ("original", None)
    assert _candidate_output_name(candidate, image_download_size=2048).endswith("-original.png")

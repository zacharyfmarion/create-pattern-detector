from PIL import Image

from src.data.scraping.manifest import read_jsonl, write_jsonl
from src.data.scraping.validation_report import build_validation_report


def test_build_validation_report_writes_status_sheets(tmp_path):
    image_path = tmp_path / "raw.png"
    Image.new("RGB", (256, 256), "white").save(image_path)
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (128, 128), "white").save(crop_path)
    manifests = tmp_path / "manifests"
    run_id = "validation-run"
    source = "cpoogle_validation"

    write_jsonl(
        manifests / f"{source}_assets_{run_id}.jsonl",
        [
            {"asset_id": "accepted", "local_path": image_path.as_posix(), "model_name": "Accepted"},
            {"asset_id": "rejected", "local_path": image_path.as_posix(), "model_name": "Rejected"},
        ],
    )
    write_jsonl(
        manifests / f"{source}_crops_{run_id}.jsonl",
        [
            {
                "asset_id": "accepted",
                "source_path": image_path.as_posix(),
                "crop_path": crop_path.as_posix(),
                "status": "accepted",
                "bbox": [0, 0, 128, 128],
                "cp_score": 0.92,
                "reasons": [],
            },
            {
                "asset_id": "rejected",
                "source_path": image_path.as_posix(),
                "crop_path": None,
                "status": "rejected",
                "bbox": [20, 20, 180, 180],
                "cp_score": 0.12,
                "reasons": ["few_long_lines"],
            },
        ],
    )

    summary = build_validation_report(tmp_path, run_id, source, output_dir=tmp_path / "report")

    assert summary.status_counts == {"accepted": 1, "rejected": 1}
    assert "accepted" in summary.sheets
    assert "rejected" in summary.sheets
    assert (tmp_path / "report" / "accepted.png").exists()
    assert (tmp_path / "report" / "rejected.png").exists()
    assert (tmp_path / "report" / "index.html").exists()
    records = read_jsonl(tmp_path / "report" / "validation_records.jsonl")
    assert len(records) == 2

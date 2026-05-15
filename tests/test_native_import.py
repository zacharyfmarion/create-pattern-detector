import json

from src.data.fold_parser import FOLDParser
from src.data.scraping.native_import import convert_cp_file, parse_cp_segments


def test_parse_cp_segments_accepts_assignment_prefixed_lines():
    segments = parse_cp_segments(
        """
        1 0 0 10 0
        2 10 0 10 10
        # ignored
        0 10 10 0 0
        """
    )

    assert len(segments) == 3
    assert segments[0].p0 == (0.0, 0.0)
    assert segments[0].p1 == (10.0, 0.0)


def test_convert_cp_file_splits_intersections_and_emits_fold(tmp_path):
    cp_path = tmp_path / "cross.cp"
    cp_path.write_text(
        "\n".join(
            [
                "1 0 5 10 5",
                "2 5 0 5 10",
                "3 0 0 10 10",
            ]
        )
    )
    fold_path = tmp_path / "cross.fold"

    result = convert_cp_file(cp_path, fold_path)
    data = json.loads(fold_path.read_text())
    parsed = FOLDParser().parse_dict(data)

    assert result.status == "converted"
    assert result.segment_count == 3
    assert result.candidate_pair_count <= 3
    assert result.timings is not None
    assert parsed.num_vertices >= 5
    assert parsed.num_edges >= 6
    assert set(data["edges_assignment"]).issubset({"U", "B"})

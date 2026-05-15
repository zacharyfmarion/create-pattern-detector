from src.data.scraping.sources import iter_cpoogle_assets, iter_obb_assets, parse_obb_html


def test_iter_cpoogle_assets_prioritizes_native_square_models():
    models = [
        {
            "id": "model-1",
            "name": "Rhino",
            "author": {"id": "author-1", "name": "Folder"},
            "paper_shape": ["Square"],
            "design_style": ["22.5°"],
            "files": [
                {"id": "photo", "name": "folded_photo.jpg", "mimeType": "image/jpeg"},
                {"id": "cp", "name": "rhino.cp", "mimeType": "text/cp"},
            ],
        },
        {
            "id": "model-2",
            "name": "Rectangle",
            "author": {"name": "Folder"},
            "paper_shape": ["Rectangle"],
            "design_style": ["Box Pleat"],
            "files": [{"id": "rect", "name": "rect.png", "mimeType": "image/png"}],
        },
    ]

    candidates = iter_cpoogle_assets(models)

    assert [c.asset_id for c in candidates][:2] == ["cpoogle:cp", "cpoogle:photo"]
    assert candidates[0].category == "native"
    assert candidates[0].metadata["paper_shape"] == ["Square"]
    assert candidates[-1].metadata["paper_shape"] == ["Rectangle"]


def test_parse_obb_html_groups_collection_images():
    html = """
    <a href="/crease-patterns/test" class="link-4">Test Model - 16x16 Grid</a>
    <div role="listitem" class="collection-item w-dyn-item">
      <div><h1 class="heading-19">Test Model - 16x16 Grid</h1>
      <div class="text-block-14">A clean CP.</div>
      <div class="text-block-15"><br/>Designed: </div>
      <div class="text-block-15">January 1, 2026</div>
      <img src="https://cdn.example/TestCP.png"/>
      <img src="https://cdn.example/folded.jpg"/>
      </div>
    </div>
    """

    items = parse_obb_html(html)
    candidates = iter_obb_assets(items)

    assert len(items) == 1
    assert items[0].title == "Test Model - 16x16 Grid"
    assert items[0].designed == "January 1, 2026"
    assert items[0].url == "/crease-patterns/test"
    assert [c.filename for c in candidates] == ["TestCP.png", "folded.jpg"]
    assert candidates[0].priority < candidates[1].priority

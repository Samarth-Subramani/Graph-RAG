import os
import json
import tempfile
from main import get_source_url_for_file, append_to_chunk_map, safe_extract_triplets


def test_get_source_url_for_file():
    meta = {"url_abc123.txt": {"source_url": "https://example.com"}}
    path = "/path/to/url_abc123.txt_content.txt"
    result = get_source_url_for_file(path, meta)
    assert result == "https://example.com"


def test_append_to_chunk_map_creates_and_appends():
    with tempfile.TemporaryDirectory() as tmpdir:
        entry = {"source_file": "file.txt", "triplet_output": "out.csv", "chunks": {}}
        append_to_chunk_map(tmpdir, entry)
        with open(os.path.join(tmpdir, "chunk_map.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        assert entry in data


def test_safe_extract_triplets_returns_list():
    # Fake extract_triplets to avoid loading models
    def fake_extract(*args, **kwargs):
        return [{"head": "A", "relation": "rel", "tail": "B"}]

    # Patch inside main module
    import main
    main.extract_triplets = fake_extract

    result = safe_extract_triplets(
        "some text here",
        "chunkid",
        tokenizer=None,
        model=None,
        device="cpu",
        max_tokens=10,
        max_new_tokens=5
    )
    assert isinstance(result, list)
    assert result[0]["head"] == "A"

import csv
from semantic_graph import normalize_entity, build_subgraph

def test_normalize_entity():
    assert normalize_entity("  ACME  Corp.  ") == "acme corp."

def test_build_subgraph(tmp_path, mocker):
    d = tmp_path / "csvs"; d.mkdir()
    f = d / "t.csv"
    with f.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["head","relation","tail","chunk_id"])
        w.writeheader()
        w.writerow({"head":"A","relation":"rel","tail":"B","chunk_id":"c1"})

    class MockModel:
        def encode(self, items, convert_to_tensor=True): return [[0.0] * 3] * len(items)
    G, clusters = build_subgraph(str(d), "topic1", MockModel(), threshold=0.0)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 1
    (u, v, data) = list(G.edges(data=True))[0]
    assert data["chunk_id"] == "c1"

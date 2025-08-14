from extractor import parse_triplets

def test_parse_triplets_basic():
    txt = """Head: A
Relation: rel
Tail: B
Head: X
Relation: r2
Tail: Y
"""
    out = parse_triplets(txt)
    assert out == [
        {"head": "A", "relation": "rel", "tail": "B"},
        {"head": "X", "relation": "r2", "tail": "Y"},
    ]

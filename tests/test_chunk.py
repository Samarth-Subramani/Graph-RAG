import types
from chunks import chunk_text_by_tokens

class MockTok:
    def __call__(self, text, return_tensors=None, truncation=False):
        # token length = words * 2 (fake)
        n = len(text.split()) * 2
        return types.SimpleNamespace(input_ids=[[0]*n])

def test_chunking_splits_on_token_limit():
    tok = MockTok()
    text = " ".join(f"w{i}" for i in range(30))  # 60 tokens
    chunks = chunk_text_by_tokens(text, tok, max_tokens=40)  # 20 words per chunk
    assert len(chunks) == 2
    assert "w0" in chunks[0] and "w29" in " ".join(chunks)

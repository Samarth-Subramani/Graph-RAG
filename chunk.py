def chunk_text_by_tokens(text, tokenizer, max_tokens):
    words = text.split()
    chunks, current_chunk = [], []

    for word in words:
        current_chunk.append(word)
        tokenized = tokenizer(" ".join(current_chunk), return_tensors="pt", truncation=False)
        if tokenized.input_ids.shape[1] > max_tokens:
            # Go back one word and push current chunk
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]  # Start new chunk with current word

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

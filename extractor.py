import torch

def extract_triplets(text, tokenizer, model, device, max_input_tokens=15000, max_new_tokens=256):
    prompt = (
        "You are an expert in information extraction. Your task is to extract factual "
        "(subject, relation, object) triplets from the given text.\n\n"
        "**Instructions:**\n"
        "- Each triplet must represent a real-world fact.\n"
        "- The 'head' and 'tail' should be entities.\n"
        "- The 'relation' should describe the factual connection.\n"
        "- Format: \n"
        "  Head: <entity>\n  Relation: <relation>\n  Tail: <entity>\n\n"
        "**Examples:**\n"
        "Text:\n"
        "\"FAU offers a Master's program in Data Science. This program is developed in collaboration "
        "with the University of Bologna. The courses are taught in English.\"\n\n"
        "Triplets:\n"
        "Head: FAU\nRelation: offers\nTail: Master's program in Data Science\n\n"
        "Head: Master's program in Data Science\nRelation: developed in collaboration with\nTail: University of Bologna\n\n"
        "Head: Courses\nRelation: are taught in\nTail: English\n\n"
        "---\n"
        f"Text:\n{text}\n\n"
        "Triplets:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Only keep output *after* "---"
    if '---' in decoded:
        decoded = decoded.split('---')[-1]

    return parse_triplets(decoded)

def parse_triplets(text):
    lines = text.split("\n")
    triplets, current = [], {}
    for line in lines:
        line = line.strip()
        if line.startswith("Head:"):
            current["head"] = line[5:].strip()
        elif line.startswith("Relation:"):
            current["relation"] = line[9:].strip()
        elif line.startswith("Tail:"):
            current["tail"] = line[5:].strip()
            if all(k in current for k in ("head", "relation", "tail")):
                triplets.append(current)
                current = {}
    return triplets

import os
import re
import csv
import json
import torch
import multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM
from translator import translate_to_english
from chunk import chunk_text_by_tokens
from extractor import extract_triplets

local_model_path = ".........."
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
INPUT_DIR = "....../final_data"
OUTPUT_DIR = "graph_csv"
MAX_INPUT_TOKENS = 6000
MAX_NEW_TOKENS = 1024
CHUNK_BUFFER = 512

LOG_PATH = "file_processing.log"

def log_error(file_path, error_msg):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{file_path} | {error_msg}\n")

def load_model(device):
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path).to(device)
    model.eval()
    print(f"✅ [GPU {device}] Model loaded and ready.")
    return tokenizer, model

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def read_text_from_csv(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            joined = ' '.join([str(cell) for cell in row if cell.strip()])
            if joined:
                texts.append(joined)
    return '\n'.join(texts)

def safe_extract_triplets(chunk, chunk_id, tokenizer, model, device, max_tokens, max_new_tokens, min_words=30):
    try:
        return extract_triplets(chunk, tokenizer, model, device, max_tokens, max_new_tokens)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"⚠️ OOM in {chunk_id}, attempting to split...")
            if len(chunk.split()) < min_words:
                print(f"🚫 Skipping tiny chunk {chunk_id} (too small to split).")
                log_error(chunk_id, "Too small after splitting. Skipped.")
                return []
            # Split the chunk in half and recurse
            midpoint = len(chunk) // 2
            split_point = chunk.find(" ", midpoint)  # split at nearest word boundary
            if split_point == -1:
                split_point = midpoint
            left_chunk = chunk[:split_point].strip()
            right_chunk = chunk[split_point:].strip()
            results = []
            results += safe_extract_triplets(left_chunk, chunk_id + "_1", tokenizer, model, device, max_tokens, max_new_tokens, min_words)
            results += safe_extract_triplets(right_chunk, chunk_id + "_2", tokenizer, model, device, max_tokens, max_new_tokens, min_words)
            return results
        else:
            raise

def get_source_url_for_file(source_file, metadata_dict):
    """
    Given a source_file path (e.g. .../url_xxx.txt_content.txt or .../url_xxx.txt_12.txt), 
    extract url_xxx.txt and find its source_url in metadata_dict.
    Returns the URL string or None.
    """
    base = os.path.basename(source_file)
    # Use regex to robustly extract url_xxx.txt from any filename
    match = re.search(r'(url_[a-zA-Z0-9]+\.txt)', base)
    if match:
        txt_key = match.group(1)
    else:
        txt_key = base  # fallback
    return metadata_dict.get(txt_key, {}).get("source_url")

def append_to_chunk_map(subdir_path, metadata_entry):
    chunk_map_path = os.path.join(subdir_path, "chunk_map.json")
    if os.path.exists(chunk_map_path):
        with open(chunk_map_path, 'r', encoding='utf-8') as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    existing.append(metadata_entry)
    with open(chunk_map_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

def process_file(file_path, tokenizer, model, gpu_id):
    print(f"📂 [GPU {gpu_id}] Starting processing of file: {file_path}")
    try:
        device = f"cuda:{gpu_id}"
        model.to(device)

        # Read raw content from file
        if file_path.endswith('.txt'):
            raw_text = read_text_from_file(file_path)
        elif file_path.endswith('.csv'):
            raw_text = read_text_from_csv(file_path)
        else:
            msg = f"Unsupported file type"
            print(f"❌ [GPU {gpu_id}] {msg}: {file_path}")
            log_error(file_path, msg)
            return

        if not raw_text.strip():
            msg = f"Empty file"
            print(f"⚠️ [GPU {gpu_id}] {msg}: {file_path}")
            log_error(file_path, msg)
            return

        # Build paths
        rel_path = os.path.relpath(file_path, INPUT_DIR)
        chunk_id_prefix = os.path.splitext(rel_path)[0].replace(os.sep, "_")
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + "_triplets.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subdir_path = os.path.dirname(output_path)

        metadata_entry = {
            "source_file": file_path,
            "triplet_output": output_path,
            "chunks": {}
        }

        print(f"🌐 [GPU {gpu_id}] Translating content...")
        translated = translate_to_english(raw_text)
        chunks = chunk_text_by_tokens(translated, tokenizer, MAX_INPUT_TOKENS - CHUNK_BUFFER)
        print(f"✅ [GPU {gpu_id}] Created {len(chunks)} chunks.")

        all_triplets = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{chunk_id_prefix}_chunk_{i+1}"
            metadata_entry["chunks"][chunk_id] = chunk
            try:
                triplets = safe_extract_triplets(chunk, chunk_id, tokenizer, model, device, MAX_INPUT_TOKENS, MAX_NEW_TOKENS)
                for t in triplets:
                    t["chunk_id"] = chunk_id
                all_triplets.extend(triplets)
                print(f"✅ [GPU {gpu_id}] Extracted {len(triplets)} triplets from {chunk_id}.")
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    msg = f"OOM in chunk {chunk_id}"
                    print(f"❌ [GPU {gpu_id}] {msg}")
                    log_error(file_path, msg)
                    continue
                else:
                    raise

        print(f"💾 [GPU {gpu_id}] Saving {len(all_triplets)} triplets to: {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["head", "relation", "tail", "chunk_id"])
            writer.writeheader()
            for t in all_triplets:
                writer.writerow(t)

        # Add the URL from metadata.json to metadata_entry
        metadata_path = os.path.join(INPUT_DIR, os.path.relpath(file_path, INPUT_DIR).split(os.sep)[0], "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata_dict = json.load(f)
            url = get_source_url_for_file(file_path, metadata_dict)
            if url:
                metadata_entry["url"] = url
        else:
            print(f"⚠️ metadata.json not found for {file_path}, URL will not be included in chunk_map.")

        append_to_chunk_map(subdir_path, metadata_entry)


        done_flag = output_path + ".done"
        with open(done_flag, 'w') as f:
            f.write("done")
        print(f"✅ [GPU {gpu_id}] Finished processing file: {file_path}")

    except Exception as e:
        msg = f"Unhandled error: {e}"
        print(f"❌ Error processing {file_path}: {msg}")
        log_error(file_path, msg)


def collect_files(input_dir, output_dir):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for fname in filenames:
            if fname.endswith((".txt", ".csv")):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, input_dir)
                done_flag = os.path.join(output_dir, os.path.splitext(rel_path)[0] + "_triplets.csv.done")
                if not os.path.exists(done_flag):
                    files.append(fpath)
    print(f"📄 Found {len(files)} files to process.")
    return files

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def worker(file_list, gpu_id):
    device = f"cuda:{gpu_id}"
    print(f"🔧 Worker on {device} processing {len(file_list)} files.")
    tokenizer, model = load_model(device)
    for f in file_list:
        process_file(f, tokenizer, model, gpu_id)

def main():
    files = collect_files(INPUT_DIR, OUTPUT_DIR)
    num_gpus = torch.cuda.device_count()
    print(f"🚀 Starting processing with {num_gpus} GPUs and {len(files)} files.")

    if num_gpus == 0:
        print("❌ No GPUs detected. Exiting.")
        return

    chunks = chunkify(files, num_gpus)
    processes = []

    for i in range(num_gpus):
        p = multiprocessing.Process(target=worker, args=(chunks[i], i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All files processed.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()

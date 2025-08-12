# GraphRAG 

This repository contains the full implementation of a **Graph-based Retrieval-Augmented Generation (GraphRAG)** pipeline developed as part of a master's thesis for the **EUMaster4HPC** program.  
The system extracts, processes, and organizes academic web data into a **semantic knowledge graph**, then provides a **Streamlit-based interface** for question answering using the knowledge graph and LLM reasoning.

---

## Overview

The project is divided into three main stages:

1. **Data Scraping** – Crawl and extract relevant academic data from EUMaster4HPC-related websites.
2. **Triplet Extraction** – Use LLM-based extraction to convert text into `(head, relation, tail)` knowledge triplets.
3. **Knowledge Graph QA** – Build a semantic knowledge graph, cluster related entities, and answer user queries via a Streamlit application.

---

## Project Structure

├── final_scrapping.py # Academic web scraping with text, table, and PDF extraction
├── translator.py # Automatic translation to English
├── chunk.py # Text chunking based on token count
├── extractor.py # LLM-based triplet extraction from text
├── main.py # Batch processing: translate → chunk → extract triplets → save CSV
├── semantic_graph.py # Build semantic knowledge graph from triplets with clustering & summaries
├── hf_load.py # Load local Hugging Face model for QA
├── app_design3.py # Streamlit-based GraphRAG interface


---

## Pipeline Details

### **1. Web Scraping (`final_scrapping.py`)**
- Iteratively crawls from the [EUMaster4HPC](https://eumaster4hpc.eu/) website and connected academic pages.
- Extracts:
  - **Text** (HTML body text with source URL)
  - **Tables** (saved as CSV after validity checks)
  - **PDFs** (in-memory extraction via PyMuPDF and PyPDF2 fallback)
- Saves metadata (`metadata.json`) with `source_url` for traceability.
- Skips irrelevant or invalid content based on heuristics.

### **2. Translation & Chunking**
- **`translator.py`** – Detects language and translates non-English text to English.
- **`chunk.py`** – Splits long texts into token-limited chunks to fit LLM input constraints.

### **3. Triplet Extraction (`main.py` + `extractor.py`)**
- Loads **local LLM models** (e.g., Mistral-7B) for triplet extraction.
- Generates `(head, relation, tail)` triplets for each chunk.
- Handles **CUDA OOM** gracefully by recursively splitting chunks.
- Saves extracted triplets to structured CSV files.
- Maintains `chunk_map.json` for mapping chunks to triplets and source URLs.

### **4. Knowledge Graph Construction (`semantic_graph.py`)**
- Reads all triplets from CSV directories.
- **Normalizes** entities (case, whitespace, punctuation).
- **Clusters** semantically similar entities using **Sentence Transformers**.
- Generates **cluster summaries** and stores embeddings.
- Builds **subgraphs per topic/university** and merges them into a global graph.
- Saves:
  - `.graphml` (graph structure)
  - `.json` (node-link format)
  - `*_clusters.json` (cluster metadata)

### **5. GraphRAG Application (`app_design3.py`)**
- **Streamlit** interface for interactive question answering.
- Features:
  - Topic/university selection or **automatic topic detection** from query.
  - Semantic subgraph extraction based on **cluster similarity** and **fuzzy keyword matching**.
  - Retrieval of **most relevant text chunk** for context.
  - **Hybrid RAG prompt** combining cluster summaries, selected triplets, and relevant source text.
  - Downloadable matched subgraph (`.graphml`).
  - Display of **cluster similarity scores** and **source URLs**.

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-enabled GPU(s)
- Hugging Face Transformers & Sentence Transformers
- Streamlit
- PyMuPDF, PyPDF2, BeautifulSoup, Requests
- LangChain & langchain-huggingface
- RapidFuzz

Install dependencies:
pip install -r requirements.txt

---
## Usage
### 1. Scrape Data
bash
Copy
Edit
python final_scrapping.py

### 2. Extract Triplets
python main.py
(Uses all available GPUs for parallel processing)

### 3. Build Knowledge Graph
python semantic_graph.py

### 4. Run Streamlit App
streamlit run graph_app.py
Access the UI at: http://localhost:8501

---
## Example Workflow
Scraper downloads HTML, CSV, and PDF content → saves as .txt and .csv with metadata.

Translator + Chunker preprocesses files for LLM input.

LLM Extractor produces triplets and stores them in structured CSV.

Semantic Graph Builder creates entity clusters and global knowledge graph.

Streamlit GraphRAG allows users to:

Select topic/university.

Ask natural language questions.

Get precise answers with sources & subgraph visualization.



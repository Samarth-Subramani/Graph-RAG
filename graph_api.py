import datetime
import streamlit as st
import networkx as nx
import spacy
import os
import re
from rapidfuzz import fuzz
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from hf_load import llm, model, tokenizer
from semantic_graph import main as graph_main
from semantic_graph import normalize_entity
from draw_graph2 import draw_graph
from langchain.chains import GraphQAChain
from triplet_extraction.extractor import extract_triplets
import threading
from io import BytesIO
from langchain_community.graphs import NetworkxEntityGraph
from langchain.prompts import PromptTemplate
import torch

EVAL_LOG_PATH = "ragas_eval_log.json"

# Load ground truth mapping (load once)
if os.path.exists("eval_questions_template.json"):
    with open("eval_questions_template.json", "r", encoding="utf-8") as f:
        eval_questions = json.load(f)
    gt_map = {item["question"]: item["ground_truth"] for item in eval_questions}
else:
    gt_map = {}
# --- Load NLP and embedding models ---
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
base_csv_dir = "/home/vault/iwia/iwia125h/exp_graph_csv"  
graph_dir = "/home/vault/iwia/iwia125h/workspace/graph2"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_graph():
    with st.spinner("📦 Building or loading knowledge graph..."):
        graphml_path = "eumaster4hpc_subgraph5.graphml"
        if os.path.exists(graphml_path):
            graph = nx.read_graphml(graphml_path)
            st.success(f"✅ Loaded graph from {graphml_path} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        else:
            st.warning("⚠️ GraphML not found. Building graph...")
            graph_main()
            graph = nx.read_graphml(graphml_path)
    return graph

def get_all_subdirs(cluster_dir="/home/vault/iwia/iwia125h"):
    return sorted([
        f.replace("_clusters5.json", "")
        for f in os.listdir(cluster_dir)
        if f.endswith("_clusters5.json")
    ])

def extract_keywords(question: str):
    doc = nlp(question)
    return set(token.lemma_.lower() for token in doc if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop)

def find_subdir_by_keywords(question, available_topics):
    keywords = extract_keywords(question)
    max_overlap = 0
    best_topic = None
    for topic in available_topics:
        topic_words = set(topic.lower().split("_") + topic.lower().split())
        overlap = sum(fuzz.partial_ratio(kw, tw) > 80 for kw in keywords for tw in topic_words)
        if overlap > max_overlap:
            max_overlap = overlap
            best_topic = topic
    if max_overlap > 0:
        return [best_topic]
    return []

def find_best_chunk_for_query(subgraph, base_csv_dir, question, embedder):
    question_emb = embedder.encode(question, convert_to_tensor=True)
    best_score = -1
    best_chunk_text = ""
    best_chunk_id = ""
    best_chunk = None
    best_subdir = ""

    # 1. Check edges for chunk_ids as before
    for u, v, data in subgraph.edges(data=True):
        subdir = data.get('subdir')
        chunk_id = data.get('chunk_id')
        if not subdir or not chunk_id:
            continue
        subdir_path = os.path.join(base_csv_dir, subdir)
        chunk_entry = load_chunk_text(subdir_path, chunk_id)
        if not chunk_entry or not chunk_entry["chunk_text"]:
            continue
        chunk_emb = embedder.encode(chunk_entry["chunk_text"], convert_to_tensor=True)
        sim = util.cos_sim(question_emb, chunk_emb).item()
        if sim > best_score:
            best_score = sim
            best_chunk = chunk_entry
            best_chunk_id = chunk_id
            best_subdir = subdir

    # 2. If no edge-based chunk found, check nodes for chunk info (if available)
    if not best_chunk_text:
        for node, data in subgraph.nodes(data=True):
            subdir = data.get('subdir')
            chunk_id = data.get('chunk_id')
            if not subdir or not chunk_id:
                continue
            subdir_path = os.path.join(base_csv_dir, subdir)
            chunk_entry = load_chunk_text(subdir_path, chunk_id)
            if not chunk_entry or not chunk_entry["chunk_text"]:
                continue
            chunk_emb = embedder.encode(chunk_entry["chunk_text"], convert_to_tensor=True)
            sim = util.cos_sim(question_emb, chunk_emb).item()
            if sim > best_score:
                best_score = sim
                best_chunk = chunk_entry
                best_chunk_id = chunk_id
                best_subdir = subdir
    if best_chunk:
        return best_chunk["chunk_text"], best_chunk_id, best_subdir, best_score, best_chunk.get("url")
    else:
        return "", "", "", -1, None

def load_chunk_text(subdir_path, chunk_id):
    chunk_map_path = os.path.join(subdir_path, "chunk_map.json")
    if not os.path.exists(chunk_map_path):
        return None
    with open(chunk_map_path, "r", encoding="utf-8") as f:
        chunk_map = json.load(f)
        for entry in chunk_map:
            if "chunks" in entry and chunk_id in entry["chunks"]:
                # Return both the chunk text and the full entry (with URL, etc)
                return {
                    "chunk_text": entry["chunks"][chunk_id],
                    "url": entry.get("url") or entry.get("source_url"),  # use whichever is available
                }
    return None

def get_top_clusters(cluster_map, question, relations, top_k=5, fuzzy_weight=0.1):

    # Extract keywords from question (for fuzzy matching)
    important_keywords = extract_keywords(question)
    if not important_keywords:
        important_keywords = [w for w in question.lower().split() if len(w) > 2]

    # Embed all relations
    relation_embeddings = embedder.encode(relations, convert_to_tensor=True)

    # Prepare cluster summaries and embeddings
    cluster_keys = []
    cluster_embs = []
    cluster_summaries = []
    for key, data in cluster_map.items():
        emb = np.array(data.get("embedding", []), dtype=np.float32)
        if emb.any():
            cluster_keys.append(key)
            cluster_embs.append(emb)
            cluster_summaries.append(data.get("summary", ""))

    if not cluster_embs:
        return []

    cluster_embs = torch.tensor(np.stack(cluster_embs))

    # Initialize scores
    cluster_scores = {key: 0.0 for key in cluster_keys}

    for rel_idx, rel_str in enumerate(relations):
        rel_emb = relation_embeddings[rel_idx].unsqueeze(0)
        cluster_embs = cluster_embs.to(rel_emb.device)
        sims = util.cos_sim(rel_emb, cluster_embs)[0].cpu().numpy()
        for j, key in enumerate(cluster_keys):
            summary_str = cluster_summaries[j]
            # Fuzzy between keywords and summary (take max for the most relevant keyword)
            kw_scores = [
                fuzz.partial_ratio(kw.lower(), summary_str.lower())
                for kw in important_keywords
            ]
            kw_fuzzy = max(kw_scores) if kw_scores else 0
            # Hybrid score
            hybrid_score = sims[j] + fuzzy_weight * kw_fuzzy
            # to include fuzzy score add this + fuzzy_weight * kw_fuzzy to the hybrid_score
            cluster_scores[key] += hybrid_score

    # Prepare results: (cluster key, total score, summary)
    scored_clusters = [
        (key, cluster_scores[key], cluster_summaries[i])
        for i, key in enumerate(cluster_keys)
    ]
    scored_clusters.sort(key=lambda x: x[1], reverse=True)
    return scored_clusters[:top_k]


def extract_subgraph_from_clusters(graph, selected_subdir, cluster_keys, cluster_map):
    subgraph = nx.MultiDiGraph()
    nodes_in_clusters = set()
    
    
    print(f"\n=== DEBUG: Subdir: {selected_subdir}")
    print(f"Cluster Keys: {cluster_keys}")

    for node, data in graph.nodes(data=True):  
        if data.get("subdir") == selected_subdir:
            node_norm = normalize_entity(str(node))
            for key in cluster_keys:
                members = cluster_map.get(key, {}).get("members", [])
                for member in members:
                    member_norm = normalize_entity(str(member))
                    if node_norm == member_norm:           
                        nodes_in_clusters.add(node)
    
    print(f"Nodes in clusters (final): {nodes_in_clusters}")
    normalized_nodes_in_clusters = set(normalize_entity(str(n)) for n in nodes_in_clusters)


    for u, v, k, data in graph.edges(data=True, keys=True):
        u_norm = normalize_entity(u)
        v_norm = normalize_entity(v)
        u_subdir_norm = normalize_entity(graph.nodes[u].get("subdir", ""))
        v_subdir_norm = normalize_entity(graph.nodes[v].get("subdir", ""))
        selected_subdir_norm = normalize_entity(selected_subdir)
    
        if u_subdir_norm != selected_subdir_norm and v_subdir_norm != selected_subdir_norm:
            continue
        if u_norm in normalized_nodes_in_clusters or v_norm in normalized_nodes_in_clusters:
            print(f"edge: {v_norm} in {normalized_nodes_in_clusters}") 
            subgraph.add_node(u, **graph.nodes[u])
            subgraph.add_node(v, **graph.nodes[v])
            subgraph.add_edge(u, v, key=k, **data)
            print(f"Added edge in subgraph: {u} --[{data.get('relation')}]--> {v}")
        #print(f"Subgraph now has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
        #print(f"Subgraph nodes: {list(subgraph.nodes())}")
        #print(f"Subgraph edges: {list(subgraph.edges(data=True))}")

    return subgraph

def extract_subgraph_for_topics(graph, question, relations, topics, high_threshold=0.82, max_clusters=5):
    merged_subgraph = nx.MultiDiGraph()
    merged_summaries = []
    all_cluster_scores = []

    for selected_subdir in topics:
        cluster_path = os.path.join(f"{selected_subdir}_clusters5.json")
        print("cluster_path:", cluster_path)
        if not os.path.exists(cluster_path):
            continue
        with open(cluster_path, 'r') as f:
            cluster_map = json.load(f)
        top_clusters = get_top_clusters(cluster_map, question, relations, top_k=max_clusters)
        print(f"Top clusters: {top_clusters}")
        all_cluster_scores.extend([(selected_subdir, name, score, summary) for (name, score, summary) in top_clusters])
        if top_clusters and top_clusters[0][1] >= high_threshold:
            selected_clusters = [top_clusters[0][0]]
            summary = top_clusters[0][2]
            print(f"Selected cluster (above threshold): {selected_clusters}")
        else:
            selected_clusters = [k for k, _, _, _ in top_clusters]
            summary = "; ".join([c[2] for c in top_clusters])
        merged_summaries.append(f"[{selected_subdir}] {summary}")
        subgraph = extract_subgraph_from_clusters(graph, selected_subdir, selected_clusters, cluster_map)
        merged_subgraph.add_nodes_from(subgraph.nodes(data=True))
        merged_subgraph.add_edges_from(subgraph.edges(data=True))
        st.write("DEBUG cluster_path for", selected_subdir, ":", cluster_path, "exists?", os.path.exists(cluster_path))
    cluster_summary = " | ".join(merged_summaries)
    all_cluster_scores.sort(key=lambda x: x[2], reverse=True)
    return merged_subgraph, cluster_summary, all_cluster_scores

def extract_answer(llm_response):
    if isinstance(llm_response, dict) and "result" in llm_response:
        text = llm_response["result"]
    elif isinstance(llm_response, str):
        text = llm_response
    else:
        return llm_response
    answer_match = re.search(r"Answer:\s*(.*?)(?:\nSource:|\Z)", text, re.IGNORECASE | re.DOTALL)
    source_match = re.search(r"Source:\s*(.*?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    not_found_match = re.search(r"Answer not found based on the knowledge graph", text, re.IGNORECASE)
    if not_found_match:
        return "Answer not found based on the knowledge graph."
    elif answer_match:
        answer = answer_match.group(1).strip()
        if source_match:
            source = source_match.group(1).strip()
            return f"**Answer:** {answer}\n\n**Source:** {source}"
        else:
            return answer
    else:
        return text[:400]

def embed_triplets_and_select(triplet_lines, question, top_n=7):
    """
    Rerank triplets by semantic similarity to the question using embeddings.
    Returns the top_n most relevant triplet lines.
    """
    if not triplet_lines:
        return []
    triplet_texts = [t for t in triplet_lines]
    # Embed triplets and question (batched for efficiency)
    triplet_embeddings = embedder.encode(triplet_texts, convert_to_tensor=True)
    question_embedding = embedder.encode([question], convert_to_tensor=True)
    # Cosine similarity (sentence-transformers)
    similarities = util.cos_sim(question_embedding, triplet_embeddings)[0].cpu().numpy()
    # Rank triplets by similarity
    sorted_idx = np.argsort(similarities)[::-1]
    top_idx = sorted_idx[:top_n]
    top_triplets = [triplet_lines[i] for i in top_idx]
    return top_triplets

def append_eval_log(entry):
    # Threaded write to avoid UI lag
    def _write():
        if os.path.exists(EVAL_LOG_PATH):
            with open(EVAL_LOG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(entry)
        with open(EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    threading.Thread(target=_write).start()

# --- UI Layout ---
st.set_page_config(page_title="Graph QA", layout="wide")

st.markdown("""
    <style>
    .sidebar-section {padding: 1.2em 1em; background: #f4f6fa; border-radius: 1.4em; margin:1.5em 0;}
    .topic-btn {display:flex;align-items:center;padding:.5em 1em;margin-bottom:0.6em;border-radius:999px;border:none;cursor:pointer;font-size:1.13em;background:#fff;box-shadow:0 1px 5px #0001;transition:background .15s;}
    .topic-btn.selected {background:#3477eb;color:#fff;box-shadow:0 2px 8px #3477eb30;}
    .topic-btn .circle {margin-right:.7em;width:1.2em;height:1.2em;border-radius:50%;border:2px solid #3477eb;background:#fff;display:inline-block;position:relative;}
    .topic-btn.selected .circle {background:#fff;border:2px solid #fff;box-shadow:0 0 0 2px #2857a8;}
    .try-again-btn {margin:1.5em 0 .5em 0;background:#f08d49;border:none;color:#fff;padding:.65em 1.5em;font-size:1em;border-radius:8px;cursor:pointer;box-shadow:0 2px 12px #f08d4922;transition:background .2s;}
    .try-again-btn:hover {background:#e26b11;}
    .select-all {margin-bottom:1em;font-weight:bold;color:#2857a8;cursor:pointer;display:block;}
    .main-section {padding:2.5em 2em 2em 2em;}
    </style>
""", unsafe_allow_html=True)

st.title("Graph-Based QA (Semantic Knowledge Graph)")
graph = load_graph()

col_sidebar, col_main = st.columns([0.32, 0.68], gap="large")

with col_sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## Topics/Universities")
    available_topics = get_all_subdirs()
    st.write("DEBUG available_topics:", available_topics)

    if 'selected_all' not in st.session_state:
        st.session_state.selected_all = False
    if 'selected_topics' not in st.session_state:
        st.session_state.selected_topics = set()

    if st.button("Select All" if not st.session_state.selected_all else "Clear All", key="select_all_btn"):
        if not st.session_state.selected_all:
            st.session_state.selected_topics = set(available_topics)
            st.session_state.selected_all = True
        else:
            st.session_state.selected_topics = set()
            st.session_state.selected_all = False

    for topic in available_topics:
        checked = topic in st.session_state.selected_topics
        if st.checkbox(f" {topic}", key=f"topic_{topic}", value=checked):
            st.session_state.selected_topics.add(topic)
        else:
            st.session_state.selected_topics.discard(topic)
    st.markdown('</div>', unsafe_allow_html=True)

with col_main:
    answer = None

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("## Question")
    question = st.text_input("Ask a question:", key="question_input")
    st.markdown('</div>', unsafe_allow_html=True)

    triplets = extract_triplets(question, tokenizer, model, device, max_input_tokens=15000, max_new_tokens=256)
    all_relations = list(set(t["relation"] for t in triplets))

    st.write("triplets:", triplets)
    st.write("relations:", all_relations)

    selected_topics = list(st.session_state.selected_topics)
    if not selected_topics and question:
        detected_topics = find_subdir_by_keywords(question, available_topics)
        if detected_topics:
            st.info(f"✨ Automatically detected topic: **{detected_topics[0]}** from your question.")
            selected_topics = detected_topics

    if selected_topics and question:
        subgraph, cluster_summary, cluster_scores = extract_subgraph_for_topics(graph, question, all_relations, selected_topics)
        best_chunk_text, best_chunk_id, best_subdir, best_score, best_chunk_url = find_best_chunk_for_query(
            subgraph, 
            base_csv_dir, 
            question, 
            embedder
        )

        # print("best_chunk_text:", best_chunk_text)
        st.write("DEBUG subgraph nodes:", len(subgraph.nodes()))
        st.write("DEBUG subgraph edges:", len(subgraph.edges()))
        if len(subgraph.nodes()) == 0 and not best_chunk_text:
            st.warning("No matching data or context found in the selected topics.")
        else:
            if len(subgraph.edges()) == 0:
                st.info("No relationships (edges) found for the matched topic/entity, but context from relevant text chunk will be used.")

            edges_with_relation = [
                (u, v, d) for u, v, d in subgraph.edges(data=True) if "relation" in d
            ]
            G_sub = nx.DiGraph()
            G_sub.add_nodes_from(subgraph.nodes(data=True))
            G_sub.add_edges_from(edges_with_relation)
            triplet_lines = [
                f"{subgraph.nodes[u].get('label', u)} --[{d['relation']}]--> {subgraph.nodes[v].get('label', v)} (source: {d.get('source_file', '')})"
                for u, v, d in G_sub.edges(data=True)
            ]

            top_triplets = embed_triplets_and_select(triplet_lines, question, top_n=20)
            context_str = "\n".join(top_triplets)

            prompt_template = PromptTemplate(
    input_variables=["query", "context", "summary", "chunk_text", "subdir"],
    template="""
You are an expert assistant answering questions using the provided knowledge graph and additional relevant source text.

A cluster summary describing the main topic(s) relevant to this question is provided below:
Cluster Summary: {summary}

Each triplet below is highly relevant to the question, based on semantic similarity.
Triplets are structured as:
  head --[relation]--> tail (source: filename)

Additionally, the most relevant original text chunk from the source data is provided below to help clarify or enrich your answer:
Relevant Source Chunk:
{chunk_text}

Your steps:
1. Use the cluster summary above to focus only on the most relevant information in the triplets below.
2. Carefully interpret the user's question, and look for both literal and semantic matches in the knowledge triplets below.
3. If the user's question asks about requirements or specific data (e.g., 'What IELTS/TOEFL scores are required?') and you find a triplet with numbers or dates (such as '85 on TOEFL' or '5 on IELTS'), include that information directly in your answer.
4. If a triplet is unclear, ambiguous, or incomplete, use the relevant source chunk above to provide additional details, clarification, or support.
5. Focus on the meaning of the question, not just literal words.
6. Use your reasoning to combine or interpret the triplets to provide a precise, concise answer.
7. Select and combine the most relevant triplet(s) to answer the question.
8. Give a single, concise answer using only information found in the triplets, summary, and relevant source chunk.
9. If you cannot find a direct answer, state that the answer is not found based on the knowledge graph.
10. If the question is not answerable with the provided information, respond with "Answer not found based on the knowledge graph."
11. if available, include the source url in your answer to help the user locate the information. usually found in the chunk_text.
12. You must restrict your answer to only the information relevant to the current topic or domain, as indicated by the provided cluster summary and triplets (from the subdir: {subdir}). Do not use information from other topics, universities, or domains, and do not generalize beyond the current subdir.

Respond ONLY in this format:

Answer: <your concise answer>
Source: <relation or source_file or chunk_id>

You must base your answer ONLY on the triplets, summary, and provided source chunk.
If the answer cannot be found *verbatim* in the triplets or the relevant chunk, reply: 'Answer not found based on the knowledge graph.' 
Never guess, infer, or use knowledge outside the triplets, summary, and chunk.

Triplets:
{context}

Question: {query}
"""
            )
            graph_chain = GraphQAChain.from_llm(
                llm=llm,
                prompt=prompt_template,
                graph=NetworkxEntityGraph(G_sub),
                verbose=True
            )
            print(f"question: {question}, context_str: {context_str}, cluster_summary: {cluster_summary}, best_chunk_text: {best_chunk_text}, best_subdir: {best_subdir}")
            response = graph_chain.invoke({
                "query": question,
                "context": context_str,
                "summary": cluster_summary,
                "chunk_text": None,
                "subdir": best_subdir
            })
            answer = extract_answer(response)
            st.markdown("### 🧠 Answer", unsafe_allow_html=True)
            st.write(answer)
            if best_chunk_url and answer:
                st.markdown(f"**Source URL:** [{best_chunk_url}]({best_chunk_url})")
                
            # Prepare log entry only if both question and answer exist
            if question and answer:
                log_entry = {
                    "question": question,
                    "answer": answer,
                    "contexts": [best_chunk_text] + top_triplets,  # best_chunk_text first, then triplets (as in prompt)
                    "ground_truth": gt_map.get(question, ""),      # match ground truth if found
                    "timestamp": datetime.datetime.now().isoformat()
                }
                append_eval_log(log_entry)
            
            st.markdown("### 🔍 Cluster Similarity Scores")
            for idx, (subdir, name, score, summary) in enumerate(cluster_scores):
                st.markdown(f"**{idx+1}.** `[Topic: {subdir}] {name}` → **{score:.2f}** — *{summary}*")

            st.markdown("### 🕸️ Matched Subgraph")
            st.write(f"Nodes: {len(G_sub.nodes)}, Edges: {len(G_sub.edges)}")

            graphml_buf = BytesIO()
            nx.write_graphml(G_sub, graphml_buf)
            graphml_buf.seek(0)
            st.download_button(
                "Download Subgraph (.graphml)",
                data=graphml_buf.read(),
                file_name="subgraph.graphml"
            )

            st.markdown("### 🔗 Most Relevant Source Chunk")
            st.write(best_chunk_text)
            st.markdown("### 🗺️ Visualize Subgraph")
            draw_graph(G_sub)
    st.markdown("</div>", unsafe_allow_html=True)




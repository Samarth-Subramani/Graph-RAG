import os
import csv
import json
import re
import networkx as nx
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import numpy as np

# -------------------- Normalize Entities --------------------
def normalize_entity(entity):
    entity = entity.strip().lower()
    entity = re.sub(r"[’‘]", "'", entity)
    entity = re.sub(r"\s+", " ", entity)
    return entity

# -------------------- Edge Semantic Clustering --------------------
def cluster_relations_semantically(relations, model, threshold=0.85):
    relations = sorted(set([rel.strip().lower() for rel in relations]))
    embeddings = model.encode(relations, convert_to_tensor=True)
    relation_map = {}
    cluster_map = defaultdict(list)
    clustered = set()

    for i, rel in enumerate(relations):
        if rel in clustered:
            continue
        sim_scores = util.cos_sim(embeddings[i], embeddings)[0]
        canonical = rel
        cluster_map[canonical].append(rel)
        relation_map[rel] = canonical
        for j, score in enumerate(sim_scores):
            if score >= threshold and relations[j] not in clustered:
                clustered.add(relations[j])
                relation_map[relations[j]] = canonical
                cluster_map[canonical].append(relations[j])
    return relation_map, cluster_map

# -------------------- Entity Semantic Clustering --------------------
def cluster_entities_semantically(entities, model, threshold=0.85):
    entities = sorted(set([e.strip().lower() for e in entities]))
    embeddings = model.encode(entities, convert_to_tensor=True)
    entity_map = {}
    cluster_map = defaultdict(list)
    clustered = set()
    for i, entity in enumerate(entities):
        if entity in clustered:
            continue
        sim_scores = util.cos_sim(embeddings[i], embeddings)[0]
        canonical = entity
        cluster_map[canonical].append(entity)
        entity_map[entity] = canonical
        for j, score in enumerate(sim_scores):
            if score >= threshold and entities[j] not in clustered:
                clustered.add(entities[j])
                entity_map[entities[j]] = canonical
                cluster_map[canonical].append(entities[j])
    return entity_map, cluster_map

# -------------------- Generate Summary from Embeddings --------------------
def summarize_cluster(entities, model):
    if not entities:
        return ""
    embeddings = model.encode(entities, convert_to_tensor=True)
    mean_vector = embeddings.mean(dim=0)
    similarities = util.cos_sim(mean_vector, embeddings)[0]
    best_index = int(np.argmax(similarities.cpu().numpy()))
    return entities[best_index], mean_vector.cpu().tolist()

# -------------------- Build Subgraph from CSV Directory --------------------
def build_subgraph(csv_dir, subdir, model, threshold=0.85):
    G = nx.MultiDiGraph()
    seen = set()
    triplets = []

    # 1. Collect all triplets and entities from CSVs
    all_entities = set()
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            with open(os.path.join(csv_dir, file), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    head = normalize_entity(row['head'])
                    tail = normalize_entity(row['tail'])
                    relation = row['relation'].strip().lower()
                    triplets.append((head, relation, tail))
                    all_entities.add(head)
                    all_entities.add(tail)

    print(f"\n🔍 {subdir}: Total unique triplets collected: {len(triplets)}")
    print(f"🔍 {subdir}: Total unique entities collected: {len(all_entities)}")

    # 2. Cluster entities semantically
    entity_map, entity_cluster_map = cluster_entities_semantically(all_entities, model, threshold)

    # 3. Map triplet entities to their canonical forms
    canonical_triplets = []
    for head, relation, tail in triplets:
        can_head = entity_map[head]
        can_tail = entity_map[tail]
        canonical_triplets.append((can_head, relation, can_tail))
    print(f"     .Entity Clustering: {len(entity_cluster_map)} clusters found")

    # 4. Cluster relations semantically (as before)
    all_relations = [relation for (_, relation, _) in canonical_triplets]
    relation_map, raw_cluster_map = cluster_relations_semantically(all_relations, model, threshold)

    # 5. Build the relation cluster map (as before)
    cluster_map = {}
    for canonical, rels_in_cluster in raw_cluster_map.items():
        # Collect all canonical nodes/entities from triplets with relation in this cluster
        cluster_nodes = set()
        for h, r, t in canonical_triplets:
            if relation_map[r] == canonical:
                cluster_nodes.add(h)
                cluster_nodes.add(t)
        summary, embedding = summarize_cluster(rels_in_cluster, model)
        cluster_map[canonical] = {
            "members": sorted(cluster_nodes),
            "summary": summary,
            "embedding": embedding,
            "relations": rels_in_cluster
        }
        print(f"   • Relation Cluster '{summary}' ({canonical}) → {len(cluster_nodes)} nodes, {len(rels_in_cluster)} relations")

    # 6. Build the subgraph with canonical entities and relations
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            with open(os.path.join(csv_dir, file), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    head = normalize_entity(row['head'])
                    tail = normalize_entity(row['tail'])
                    relation = row['relation'].strip().lower()
                    chunk_id = row.get('chunk_id', "")
                    source_file = os.path.relpath(os.path.join(csv_dir, file), csv_dir)
                    can_head = entity_map[head]
                    can_tail = entity_map[tail]
                    canonical_relation = relation_map[relation]
                    triplet_key = (can_head, canonical_relation, can_tail, chunk_id)
                    if triplet_key in seen:
                        continue
                    seen.add(triplet_key)
                    G.add_node(can_head, label=can_head, subdir=normalize_entity(subdir)) # label can be canonical or original, your choice
                    G.add_node(can_tail, label=can_tail, subdir=normalize_entity(subdir))
                    G.add_edge(
                        can_head, 
                        can_tail,
                        relation=relation,
                        canonical_relation=canonical_relation,
                        subdir=subdir,
                        source_file=source_file,
                        chunk_id=chunk_id
                    )
    return G, cluster_map, entity_cluster_map  # (return both for later use!)

# -------------------- Sanitize the Graph --------------------
def sanitize_graph(graph: nx.Graph) -> nx.Graph:
    def clean_value(val):
        if isinstance(val, str):
            val = re.sub(r"[\x00-\x1F\x7F]", "", val)
            return val.strip()
        elif isinstance(val, (int, float, bool)):
            return val
        else:
            return str(val)

    clean_graph = nx.MultiDiGraph() if graph.is_directed() else nx.MultiGraph()

    for node, attrs in graph.nodes(data=True):
        clean_node = clean_value(str(node))
        clean_attrs = {k: clean_value(v) for k, v in attrs.items()}
        clean_graph.add_node(clean_node, **clean_attrs)

    for u, v, attrs in graph.edges(data=True):
        clean_u = clean_value(str(u))
        clean_v = clean_value(str(v))
        clean_attrs = {k: clean_value(vv) for k, vv in attrs.items()}
        clean_graph.add_edge(clean_u, clean_v, **clean_attrs)

    return clean_graph

# -------------------- Main --------------------
def main():
    csv_input_dir = "/home/vault/iwia/iwia125h/exp_graph_csv"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  

    global_graph = nx.MultiDiGraph()
    global_cluster_map = {}

    for subdir in sorted(os.listdir(csv_input_dir)):
        subdir_path = os.path.join(csv_input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        subgraph_path = f"{subdir}_subgraph.graphml"
        if os.path.isfile(subgraph_path):
            print(f"Skipping {subdir} - subgraph already exists.")
            continue  # Skip if subgraph already exists

        print(f"\n▶ Processing {subdir}...")
        subgraph, cluster_map, entity_cluster_map = build_subgraph(subdir_path, subdir, model)

        # Sanitize subgraph
        subgraph = sanitize_graph(subgraph)

        # Save per-subdir subgraph
        nx.write_graphml(subgraph, subgraph_path)

        # Save cluster map with summaries and embeddings
        with open(f"{subdir}_clusters.json", "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, indent=2, ensure_ascii=False)
       
        entity_cluster_map_structured = {}
        for canonical, members in entity_cluster_map.items():
            summary, embedding = summarize_cluster(members, model)
            entity_cluster_map_structured[canonical] = {
                "members": members,
                "summary": summary,
                "embedding": embedding
            }

        with open(f"{subdir}_entity_clusters4.json", "w", encoding="utf-8") as f:
            json.dump(entity_cluster_map_structured, f, indent=2, ensure_ascii=False)

        # Add to global graph as disconnected component
        for node, data in subgraph.nodes(data=True):
            global_graph.add_node(node, **data)

        for u, v, data in subgraph.edges(data=True):
            global_graph.add_edge(u, v, **data)

        global_cluster_map[subdir] = cluster_map

    # Save final graph
    nx.write_graphml(global_graph, "knowledge_graph.graphml")
    with open("knowledge_graph.json", "w", encoding="utf-8") as f:
        json.dump(nx.readwrite.json_graph.node_link_data(global_graph, edges="links"), f, indent=2, ensure_ascii=False)

    print("\n✅ Knowledge Graph Built")

if __name__ == "__main__":
    main()





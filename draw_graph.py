import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
import matplotlib.patches as mpatches
def draw_graph(graph, max_label_length=200):
    """Draws a subgraph in Streamlit, adapted for hub/spoke or small graphs."""
    num_nodes = len(graph)
    figsize = (6, 4.5)

    # Use circular layout for star-like graphs
    if num_nodes < 15 or nx.density(graph) < 0.15:
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=42, k=0.6 / (num_nodes**0.5 + 1))

    node_types = {}
    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "default")
        node_types.setdefault(node_type, []).append(node)

    color_map = {
        "default": "#8ecae6",
        "person": "#b7e4c7",
        "organization": "#f4a259",
        "location": "#bdb2ff",
        "program": "#ffb4a2"
    }

    plt.figure(figsize=figsize)
    plt.gca().set_facecolor('white')

    for node_type, nodes in node_types.items():
        color = color_map.get(node_type, "#8ecae6")
        nx.draw_networkx_nodes(
            graph, pos, nodelist=nodes, node_color=color, label=node_type.title(),
            node_size=350, alpha=0.92
        )

    nx.draw_networkx_edges(graph, pos, edge_color="#adb5bd", arrows=True, width=1.2, alpha=0.5)

    def short(label):
        return (label[:max_label_length] + "…") if len(label) > max_label_length else label

    node_labels = {n: short(graph.nodes[n].get("label", str(n))) for n in graph.nodes}
    nx.draw_networkx_labels(
        graph, pos, labels=node_labels, font_size=8, font_color="#263238",
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.5)
    )

    if len(node_types) > 1:
        handles = [mpatches.Patch(color=color_map.get(t, "#8ecae6"), label=t.title()) for t in sorted(node_types)]
        plt.legend(handles=handles, loc='lower left', frameon=True, fontsize=9)

    plt.tight_layout(pad=0.5)
    plt.axis("off")
    st.pyplot(plt.gcf())
    plt.clf()

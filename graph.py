
master_agreement = "MA-2025-001"

sub_agreements = [
    {"id": "SA-101", "name": "Sub Agreement 1"},
    {"id": "SA-102", "name": "Sub Agreement 2"},
]

supporting_docs = {
    "SA-101": [
        {"doc_id": "DOC-11", "file": "invoice_11.pdf"},
        {"doc_id": "DOC-12", "file": "po_12.pdf"},
    ],
    "SA-102": [
        {"doc_id": "DOC-21", "file": "annexure_21.docx"}
    ],
}

from streamlit_agraph import agraph, Node, Edge, Config

nodes = []
edges = []

nodes.append(Node(id=master_agreement, label=master_agreement, size=40, color="#1f77b4"))

for sa in sub_agreements:
    nodes.append(Node(id=sa["id"], label=sa["id"], size=30, color="#2ca02c"))
    edges.append(Edge(source=master_agreement, target=sa["id"]))

    for d in supporting_docs.get(sa["id"], []):
        nodes.append(Node(id=d["doc_id"], label=d["file"], size=20, color="#ff7f0e"))
        edges.append(Edge(source=sa["id"], target=d["doc_id"]))

config = Config(width=800, height=500, directed=True, physics=True)

agraph(nodes=nodes, edges=edges, config=config)
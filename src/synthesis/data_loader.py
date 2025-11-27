from pathlib import Path
from rdflib import Graph
from tqdm import tqdm

def load_knowledge_graph(root_dir: Path) -> Graph:
    print(f"Loading knowledge graph from: {root_dir}")
    g = Graph()
    ttl_files = [f for f in root_dir.glob("*.ttl")]

    print("Parsing .ttl files ...")
    for p in tqdm(ttl_files, desc="Parsing", leave=True, position=0):
        try:
            g.parse(p, format="turtle")
        except Exception as e:
            print("warn: skip", p, "err:", e)

    print("Done. triples:", len(g))
    return g

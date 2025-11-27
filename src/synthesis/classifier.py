from typing import Literal

from rdflib import Graph, Node, URIRef
from rdflib.namespace import RDF
from tqdm import tqdm

from ..create_graph.rdf_utils import HAS_PREDS, VLEGAL, name

RAW_TEXT = URIRef(VLEGAL + "rawText")

POINTER_KEYWORDS = [
    "sửa đổi",
    "bổ sung",
    "thay thế",
    "bãi bỏ",
    "hết hiệu lực",
    "hiệu lực thi hành",
]

POINTER_RELS = [
    "replaces",
    "amends",
    "disables",
    "pursuantsOn",
    "amendedBy",
    "repealedBy",
    "repeals",
    "supersedes",
    "supersededBy",
]


def _v(uri_name: str) -> URIRef:
    return URIRef(VLEGAL + uri_name)

# Node = URIRef
# _SubjectType = _PredicateType = _ObjectType = Node

ClassifierOutput = dict[
    Node,
    Literal[
        "Pointer Node",
        "Structural Node",
        "Content Node",
    ],
]

class NodeClassifier:
    def __init__(self, g: Graph):
        self.g = g
        self.raw = {s: str(o) for s, o in g.subject_objects(RAW_TEXT)}
        self.types = {s: name(str(o)) for s, o in g.subject_objects(RDF.type)}

        self.children = self._children_map()
        self.node2doc = self._node2doc_map()
        self.pointer_docs = self._pointer_docs()

        self.article_nodes = set(g.subjects(RDF.type, _v("Article")))

    def _children_map(self):
        child_map: dict[Node, set[Node]] = {}
        for s, p, o in self.g:
            if str(p) not in HAS_PREDS: continue
            child_map.setdefault(s, set()).add(o)
        return child_map

    def _node2doc_map(self):
        # climb down from hasArticle edges to attach doc→descendants
        node2doc: dict[Node, Node | None] = {
            node: doc
            for doc, _, node in self.g.triples((None, _v("hasArticle"), None))
        }
        q = list(node2doc.keys())
        seen = set(q)
        while q:
            u = q.pop(0)
            doc = node2doc.get(u)

            childs = self.children.get(u)
            if childs is None: continue

            for v in childs:
                if v in seen: continue
                seen.add(v)
                node2doc[v] = doc
                q.append(v)

        return node2doc

    def _pointer_docs(self):
        rels = {_v(r) for r in POINTER_RELS}
        return set(
            subj
            for rel in rels
            for subj, _, _ in self.g.triples((None, rel, None))
        )

    def classify_nodes(self):
        print("classifying ...")
        out: ClassifierOutput = {}

        for node in tqdm(
            self.article_nodes, desc="nodes", leave=True, position=0
        ):
            text = self.raw.get(node, "").lower()
            has_kids = node in self.children
            doc = self.node2doc.get(node)
            intrinsic = any(k in text for k in POINTER_KEYWORDS)
            in_ptr = (doc in self.pointer_docs) if doc else False

            if intrinsic and in_ptr:
                out[node] = "Pointer Node"
            elif has_kids:
                out[node] = "Structural Node"
            else:
                out[node] = "Content Node"

        # small summary number of nodes for each type
        cnt = {}
        for v in out.values():
            cnt[v] = cnt.get(v, 0) + 1
        print("summary:", cnt)
        return out

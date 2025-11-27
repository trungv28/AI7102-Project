from dataclasses import dataclass
from os.path import isfile
from typing import Collection

import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.typing import EdgeType

from ..helper import LocalGraphInfo
from . import rdf_utils
from .store import EmbStore

MAX_TREE_DEPTH = 30
MAX_PER_TYPE = 200
MAX_XDOCS = 1
ENC_BATCH = 128
ADD_CROSS_EDGES = True
CROSS_KEEP_NAMES = True

ALLOWED = {
    'article','chapter','section','document','clause','paragraph',
    'title','subtitle','preamble','annex','appendix','schedule',
    'definition','provision','exception','condition','penalty',
    'right','obligation','prohibition','point','item','part',
    'subsection','subclause','subparagraph','subpoint'
}

@dataclass
class LocalGraphBuilder:
    uri2file: dict[str, str]
    uri2text: dict[str, str]
    doc2file: dict[str, str]
    model: SentenceTransformer
    store: EmbStore

    add_star: bool = False
    root_rel: str = "root_to"
    allowed_types: set[str] | None = None

    max_depth = MAX_TREE_DEPTH
    max_per_type = MAX_PER_TYPE
    max_xdocs = MAX_XDOCS
    cross_keep_names=CROSS_KEEP_NAMES
    add_cross_edges=ADD_CROSS_EDGES

    def build(self, uri: str):
        path_ = self.uri2file.get(uri)
        if not path_ or not isfile(path_): return None, None

        parents, kids, types, texts, xdocs = rdf_utils.parse(path_).get_all()
        root = rdf_utils.find_root(uri, parents)
        types[root] = 'document'
        tree = rdf_utils.tree(root, kids, types, self.max_depth)

        # all_u = set(tree); all_t = dict(types); all_x = dict(texts)
        # all_p, all_k = defaultdict(list), defaultdict(list)
        # for k, v in parents.items(): all_p[k].extend(v)
        # for k, v in kids.items():    all_k[k].extend(v)

        # gather cross relations seen inside main doc tree
        # list of (pred_name, target_uri)
        xrels: list[tuple[str, str]] = [
            (pred, tgt) for u in tree for pred, tgt in xdocs.get(u, [])
        ]

        # unique by target uri, keep order
        seen = set()
        rels = []
        for pred, tgt in rels:
            if tgt in seen: continue
            seen.add(tgt)
            rels.append((pred, tgt))
        xrels = rels
        del seen, rels

        xrels = xrels[:self.max_xdocs]

        # (root_doc_uri, related_root_uri, pred_name)
        cross_pairs: list[tuple[str, str, str]] = []
        # (related_uri, related_root_uri)
        added_docs: list[tuple[str, str]] = []

        for pred, xuri in xrels:
            xfile = rdf_utils.find_file(xuri, self.uri2file, self.doc2file)
            if not xfile or xfile == path_ or not isfile(xfile): continue

            # rp, rk, rt, rx, _ = rdf_utils.parse(xfile).get_all()
            xparents, xkids, xtypes, xtexts, _ = rdf_utils.parse(xfile).get_all()

            roots = [u for u in xtypes if (xuri in u or u in xuri)]
            if not roots: roots = list(xtypes)
            if not roots: continue

            xroot = rdf_utils.find_root(roots[0], xparents)
            xtypes[xroot] = 'document'
            xtree = rdf_utils.tree(xroot, xkids, xtypes, self.max_depth//2)

            tree.update(xtree)
            types.update(xtypes)
            texts.update(xtexts)

            for u, v in xparents.items(): parents[u].extend(v)
            for u, v in xkids.items(): kids[u].extend(v)

            # hook the two docs structurally
            kids[root].append(xroot)
            parents[xroot].append(root)

            cross_pairs.append((root, xroot, pred))
            added_docs.append((xuri, xroot))

        g = self._hetero(
            tree,
            types,
            texts,
            kids,
            cross_pairs
        )

        root_type, root_idx = None, None
        if g is not None and self.add_star:
            root_type, root_idx = add_root_edge(g, uri, types, self.root_rel)

        info: LocalGraphInfo = {
            "cand_uri": uri,
            "doc_root": root,
            "cand_in_graph": (
                g is not None
                and hasattr(g, "node_types")
                and any(
                    hasattr(g[nt], "uri") and uri in g[nt].uri
                    for nt in (g.node_types or [])
                )
            ),
            "added_docs": [ruri for ruri, _ in added_docs][:5],
            "root_type": root_type,
            "root_idx": root_idx,
        }
        return g, info

    def _hetero(
        self,
        tree: Collection[str],
        types: dict[str, str],
        texts: dict[str, str],
        kids: dict[str, list[str]],
        cross_pairs: list[tuple[str, str, str]],
        enc_batch=ENC_BATCH,
    ):
        allowed = self.allowed_types or ALLOWED
        type2uris: dict[str, list[str]] = {}
        for u in tree:
            t = types.get(u, "unknown")
            if t in allowed:
                type2uris.setdefault(t, []).append(u)

        # all types has no uris
        if not any(type2uris.values()): return None

        # NodeType = str
        g = HeteroData()
        need_texts: list[str] = []
        need_uris: list[str] = []

        for t, uris in type2uris.items():
            uris = uris[:self.max_per_type]
            type2uris[t] = uris
            if not uris: continue

            batch, keep = self.store.gets(uris)
            if len(keep) == len(uris):
                g[t].x = torch.from_numpy(batch).float()
                g[t].uri = uris
            else:
                for u in uris:
                    text = self.uri2text.get(u, texts.get(u, ""))
                    if not text: text = f"{t.capitalize()}: {u.split('/')[-1]}"

                    need_texts.append(text)
                    need_uris.append(u)

        if need_texts:
            feats = self.model.encode(
                need_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=enc_batch,
                show_progress_bar=False,
                trust_remote_code=True,
            )
            for u, feat in zip(need_uris, feats):
                self.store.add(u, feat)

        for t, uris in type2uris.items():
            batch, keep = self.store.gets(uris)
            if len(keep) == 0: continue

            g[t].x = torch.from_numpy(batch).float()
            g[t].uri = keep


        graph_type2uri2idx = {
            t: {u: i for i, u in enumerate(getattr(g[t], "uri", []))}
            for t in g.node_types
        }

        # structural edges (still 'parent_of' to stay fast)
        for parent, childs in kids.items():
            parent_type = types.get(parent, "unknown")
            if parent_type not in g.node_types: continue

            parent_idx = graph_type2uri2idx.get(parent_type, {}).get(parent, None)
            if parent_idx is None: continue

            for child in childs:
                child_type = types.get(child, "unknown")
                child_idx = graph_type2uri2idx.get(child_type, {}).get(child, None)
                if child_idx is None: continue

                edge_type = (parent_type, "parent_of", child_type)
                if edge_type not in g.edge_types:
                    g[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

                edge = torch.tensor([[parent_idx], [child_idx]], dtype=torch.long)
                g[edge_type].edge_index = torch.cat(
                    [g[edge_type].edge_index, edge], dim=1
                )

        # cross-doc edges (doc_root -> related_doc_root)
        if (
            self.add_cross_edges
            and cross_pairs
            and ("document" in g.node_types)
        ):
            idoc = graph_type2uri2idx.get("document", {})
            for u, v, pred in cross_pairs:
                if not (u in idoc and v in idoc): continue

                rel = pred if self.cross_keep_names else "cross_doc"
                edge_type = ("document", rel, "document")
                if edge_type not in g.edge_types:
                    g[edge_type].edge_index = torch.empty(
                        (2, 0), dtype=torch.long
                    )
                edge = torch.tensor([[idoc[u]], [idoc[v]]], dtype=torch.long)
                g[edge_type].edge_index = torch.cat(
                    [g[edge_type].edge_index, edge], dim=1
                )

        return g


def add_root_edge(
    g: HeteroData,
    uri: str,
    types: dict[str, str],
    root_rel="root_to",
) -> tuple[str | None, int | None]:
    """add root → edges for this candidate"""

    root_type = types.get(uri, "unknown")
    if root_type not in g.node_types: return None, None
    if not hasattr(g[root_type], "uri") or uri not in g[root_type].uri:
        return root_type, None
    root_idx = g[root_type].uri.index(uri)

    for t in g.node_types:
        N = g[t].x.size(0) if hasattr(g[t], "x") else 0
        if N == 0: continue

        key = (root_type, root_rel, t)
        if not hasattr(g[key], "edge_index"):
            g[key].edge_index = torch.empty((2, 0), dtype=torch.long)

        src = torch.full((N,), root_idx, dtype=torch.long)
        dst = torch.arange(N, dtype=torch.long)
        g[key].edge_index = torch.cat(
            [g[key].edge_index, torch.stack([src, dst], 0)], dim=1
        )

    return root_type, root_idx


@dataclass
class GlobalGraphBuilder:
    local_graphs: list[HeteroData]
    drop_relations: set[str] | None = None

    def build(self):
        """
        URI-deduplicated union:
          - One node per (node_type, URI) across all locals.
          - Edges are the union of remapped local edges (no new edges invented).
          - Duplicate edges removed per edge type.
          - If drop_relations is provided (set of relation names), those edge types are excluded.
        """

        local_graphs = self.local_graphs
        if not local_graphs:
            return None

        node_types = set(nt for g in local_graphs for nt in g.node_types)
        edge_types = set(et for g in local_graphs for et in g.edge_types)

        G = HeteroData()
        node_type2uri2idx: dict[str, dict[str, int]] = {nt: {} for nt in node_types}
        node_type2uris: dict[str, list[str]] = {nt: [] for nt in node_types}
        node_type2xs: dict[str, list[torch.Tensor]] = {nt: [] for nt in node_types}
        edge_type2edges: dict[EdgeType, list[torch.Tensor]] = {
            et: [] for et in edge_types
        }

        # merge nodes, remember local->global maps
        for g in local_graphs:
            local2global: dict[str, torch.Tensor] = {}

            for nt in g.node_types:
                uris_local: list[str] = getattr(g[nt], 'uri', [])
                if not uris_local: continue

                x_local = g[nt].x  # [N, d]
                map_idx = torch.empty((x_local.size(0),), dtype=torch.long)
                for i, u in enumerate(uris_local):
                    if u not in node_type2uri2idx[nt]:
                        node_type2uri2idx[nt][u] = len(node_type2uris[nt])
                        node_type2uris[nt].append(u)
                        node_type2xs[nt].append(
                            x_local[i : i + 1]  # keep first occurrence
                        )
                    map_idx[i] = node_type2uri2idx[nt][u]
                local2global[nt] = map_idx

            # remap edges for this local graph
            for et in g.edge_types:
                if not hasattr(g[et], "edge_index"):
                    continue

                src, rel, dst = et
                if self.drop_relations and rel in self.drop_relations:
                    continue  # strip e.g. 'root_to' from global

                ei = g[et].edge_index
                if ei.numel() == 0: continue

                if src not in local2global or dst not in local2global:
                    continue

                src = local2global[src][ei[0]]
                dst = local2global[dst][ei[1]]
                edge_type2edges.setdefault(et, []).append(
                    torch.stack([src, dst], dim=0)
                )

        # finalize nodes
        feature_dim = self._get_feature_dim()
        for nt in node_types:
            if node_type2xs[nt]:
                G[nt].x = torch.cat(node_type2xs[nt], dim=0)
                G[nt].uri = node_type2uris[nt]
            else:
                G[nt].x = torch.empty((0, feature_dim), dtype=torch.float)
                G[nt].uri = []

        # finalize edges (concat then dedup columns)
        for et in edge_types:
            if edge_type2edges.get(et):
                E = torch.cat(edge_type2edges[et], dim=1)
                if E.numel() > 0:
                    E = torch.unique(E, dim=1)  # dedup
                G[et].edge_index = E
            else:
                G[et].edge_index = torch.empty((2, 0), dtype=torch.long)

        return G


    def _get_feature_dim(self) -> int:
        for g in self.local_graphs:
            for nt in g.node_types:
                if hasattr(g[nt], "x") and g[nt].x.numel() > 0:
                    d = g[nt].x.size(1)
                    return d
        return 768


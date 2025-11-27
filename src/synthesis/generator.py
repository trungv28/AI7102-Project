import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import typer
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from rdflib import Graph, Node
from tqdm import tqdm

from ..helper import T
from .classifier import (
    RAW_TEXT,
    ClassifierOutput,
    NodeClassifier,
    POINTER_RELS,
    _v,
)
from .data_loader import load_knowledge_graph
from .prompt import (
    CROSS_DOC_SYNTHESIS_PROMPT,
    IR_CONTENT_NODE_PROMPT,
    STRUCTURAL_SUMMARY_PROMPT,
    STRUCTURAL_SYNTHESIS_PROMPT,
)


def _short(u, limit=42):
    u = str(u)
    base = u.split("/")[-1]
    return base[:limit]


def _take(xs: list[T], k: int) -> list[T]:
    return random.sample(xs, min(k, len(xs)))


def _merge_texts(texts: list[str | None], max_chars=1200):
    s = " ".join([
        text.strip() for text in texts
        if text and len(text.strip()) > 0
    ])
    return s[:max_chars]


def _chat_once(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens=256,
):
    msgs: list[ChatCompletionMessageParam] = []
    msgs.append({"role": "user", "content": prompt})
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})

    r = client.chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content


def _raw(g: Graph, node: Node):
    for _, _, o in g.triples((node, RAW_TEXT, None)):
        return str(o)

@dataclass
class QuestionGenerator:
    g: Graph
    roles: ClassifierOutput
    client: OpenAI
    model: str = "qwen2.5:7b"

    def generate_phase_1_content_node(self, n=10):
        nodes = [u for u, r in self.roles.items() if r == "Content Node"]
        out = []
        for node in tqdm(_take(nodes, n), desc="P1", leave=True, position=0):
            context = _raw(self.g, node)
            if context is None: continue

            prompt = IR_CONTENT_NODE_PROMPT.format(context_text=context)
            q = _chat_once(self.client, self.model, prompt)
            out.append(
                {
                    "phase": "P1_content",
                    "node": str(node),
                    "doc": str(
                        self.roles.get("doc_of", "")
                    ),  # leaving your original field as-is
                    "question": q,
                    "context": context,
                    "relevant_node_uris": [str(node)],
                }
            )
        return out

    def generate_phase_2_structural_node(self, n=10):
        doc2arts: dict[Node, list[Node]] = {}
        for doc, _, art in self.g.triples((None, _v("hasArticle"), None)):
            doc2arts.setdefault(doc, []).append(art)

        arts = [u for u, r in self.roles.items() if r == "Structural Node"]
        chosen = _take(arts, n)
        out = []

        for art in tqdm(chosen, desc="P2", position=0, leave=True):
            doc = None
            for d, lst in doc2arts.items():
                if art in lst:
                    doc = d
                    break

            if doc is None: continue

            siblings = doc2arts.get(doc, [])
            texts = [_raw(self.g, u) for u in siblings]
            ctx = _merge_texts(texts, 1600)

            if random.random() < 0.5:
                # summary over ALL articles in the doc
                prompt = STRUCTURAL_SUMMARY_PROMPT.format(context_text=ctx)
                phase = "P2_struct_summary"
                rel_uris = [str(u) for u in siblings]
                saved_ctx = ctx
            else:
                # synthesis over a sampled subset
                sibs2 = _take(siblings, min(4, len(siblings)))
                texts2 = [_raw(self.g, u) for u in sibs2]
                ctx2 = _merge_texts(texts2, 1200)
                prompt = STRUCTURAL_SYNTHESIS_PROMPT.format(context_text=ctx2)
                phase = "P2_struct_synthesis"
                rel_uris = [str(u) for u in sibs2]
                saved_ctx = ctx2

            q = _chat_once(self.client, self.model, prompt)
            out.append(
                {
                    "phase": phase,
                    "doc": str(doc),
                    "article": str(art),
                    "question": q,
                    "context": saved_ctx,
                    "relevant_node_uris": rel_uris,
                }
            )
        return out

    def generate_phase_3_cross_doc(self, n=10):
        rels = {_v(r) for r in POINTER_RELS}
        pairs = set(
            (s, o)
            for rel in rels
            for s, _, o in self.g.triples((None, rel, None))
        )
        pairs = list(pairs)
        pairs = _take(list(pairs), n)
        out = []

        for s, o in tqdm(pairs, desc="P3", leave=True, position=0):
            s_arts = list(self.g.objects(s, _v("hasArticle")))
            o_arts = list(self.g.objects(o, _v("hasArticle")))
            parts = _take(s_arts, 3) + _take(o_arts, 3)

            texts = [_raw(self.g, u) for u in parts]
            ctx = "\n\n".join([t for t in texts if t])
            if not ctx: continue

            prompt = CROSS_DOC_SYNTHESIS_PROMPT.format(context=ctx)
            q = _chat_once(self.client, self.model, prompt)
            out.append(
                {
                    "phase": "P3_cross_doc",
                    "doc_left": str(s),
                    "doc_right": str(o),
                    "question": q,
                    "context": ctx,
                    "relevant_node_uris": [str(u) for u in parts],
                }
            )
        return out


def main(
    data_root: str,
    outfile: str = "dataset.jsonl",
    per_phase: int = 10,
    model: str = "qwen2.5-7b-instruct",
    api_key: str = os.getenv("OPENROUTER_API_KEY", ""),
    api_base: str = os.getenv(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    ),
):
    print("config:")
    print(" data_root:", data_root)
    print(" per_phase:", per_phase)
    print(" out:", outfile)
    print(" model:", model)
    print(" base:", api_base)

    client = OpenAI(api_key=api_key, base_url=api_base)

    root = Path(data_root)
    g = load_knowledge_graph(root)

    clf = NodeClassifier(g)
    roles = clf.classify_nodes()

    gen = QuestionGenerator(g, roles, client, model=model)

    all_rows = []
    all_rows += gen.generate_phase_1_content_node(per_phase)
    all_rows += gen.generate_phase_2_structural_node(per_phase)
    all_rows += gen.generate_phase_3_cross_doc(per_phase)

    print("\nTotal samples:", len(all_rows))
    print("writing:", outfile)
    with open(outfile, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done.")

if __name__ == "__main__":
    typer.run(main)

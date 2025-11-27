from collections import defaultdict, deque
from dataclasses import dataclass

import rdflib
from rdflib.namespace import RDF

VLEGAL = "https://aseados.ucd.ie/vlegal/"
RAW_TEXT = VLEGAL + "rawText"

HAS_PREDS = {VLEGAL + s for s in [
    "hasArticle","hasClause","hasPoint","hasSection","hasChapter","hasPart","hasItem","hasSubsection",
    "hasSubclause","hasParagraph","hasSubparagraph","hasSubpoint","hasTitle","hasSubtitle","hasPreamble",
    "hasAnnex","hasAppendix","hasSchedule","hasDefinition","hasProvision","hasException","hasCondition",
    "hasPenalty","hasRight","hasObligation","hasProhibition"
]}

OF_PREDS = {VLEGAL + s for s in [
    "articleOf","clauseOf","pointOf","sectionOf","chapterOf","partOf","itemOf","subsectionOf",
    "subclauseOf","paragraphOf","subparagraphOf","subpointOf","titleOf","subtitleOf","preambleOf",
    "annexOf","appendixOf","scheduleOf","definitionOf","provisionOf","exceptionOf","conditionOf",
    "penaltyOf","rightOf","obligationOf","prohibitionOf"
]}

CROSS_PREDS = {VLEGAL + s for s in [
    "relatedTo","pursuantsOn","amendedBy","amends","repealedBy","repeals","supersededBy","supersedes",
    "implementedBy","implements","derivedFrom","derives","citedBy","cites","referencedBy","references",
    "basedOn","consolidatedBy","consolidates","codifiedBy","codifies","precededBy","precedes",
    "followedBy","follows","replacedBy","replaces"
]}

@dataclass
class ParseResult:
    parents: dict[str, list[str]]
    kids: dict[str, list[str]]
    types: dict[str, str]
    texts: dict[str, str]
    xdocs: dict[str, list[tuple[str, str]]]

    def get_all(self):
        return self.parents, self.kids, self.types, self.texts, self.xdocs

PARSE_CACHE: dict[str, ParseResult] = {}

def parse(path: str):
    if path in PARSE_CACHE:
        return PARSE_CACHE[path]

    g = rdflib.Graph()
    g.parse(path, format="turtle")

    parents = defaultdict(list)
    kids = defaultdict(list)
    types = {}
    texts = {}
    xdocs = defaultdict(list)  # s -> [(pred_name, o)]

    for s, p, o in g:
        s, p, o = str(s), str(p), str(o)
        if p == str(RDF.type):
            types[s] = name(o)
        if p == RAW_TEXT:
            texts[s] = o
        if p in HAS_PREDS:
            parents[o].append(s)
            kids[s].append(o)
        if p in OF_PREDS:
            parents[s].append(o)
            kids[o].append(s)
        if p in CROSS_PREDS:
            xdocs[s].append((name(p), o))  # keep the predicate label

    PARSE_CACHE[path] = ParseResult(
        dict(parents),
        dict(kids),
        types,
        texts,
        dict(xdocs),
    )
    return PARSE_CACHE[path]

def name(uri: str):
    if '#' in uri: return uri.split('#')[-1].lower()
    return uri.rstrip('/').split('/')[-1].lower()


def find_file(
    uri: str,
    uri2file: dict[str, str],
    doc2file: dict[str, str],
):
    if uri in uri2file: return uri2file[uri]
    doc = uri.split('/')[-1].split('#')[0].split('.')[0]
    if doc in doc2file: return doc2file[doc]
    return None

def find_root(uri: str, parents: dict[str, list[str]], max_depth=50):
    cur = uri
    seen = set()
    d = 0
    while d < max_depth and cur not in seen:
        seen.add(cur)
        p = parents.get(cur, [])
        if not p: break
        cur = p[0]
        d += 1

    return cur

def tree(
    root: str,
    kids: dict[str, list[str]],
    types: dict[str, str],
    max_depth=100,
):
    """BFS over hierarchical edges"""
    nodes: set[str] = set()
    queue = deque([(root, 0)])
    seen = set()
    skip = {"organization", "location", "person", "date", "number"}

    while queue:
        u, d = queue.popleft()
        if u in seen or d > max_depth: continue
        seen.add(u)

        if types.get(u, "unknown") in skip: continue
        nodes.add(u)

        for v in kids.get(u, []):
            if v not in seen: queue.append((v, d+1))

    return nodes


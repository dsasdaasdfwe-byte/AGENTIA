#!/usr/bin/env python3
import json
import re

from ollama_client import chat
from grounding_prompts import atomic_claim_extract, verifier_a, verifier_b, atomic_repair

CITE_RE = re.compile(r"\[L(\d{4})(?:-L(\d{4}))?\]")
MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)
DATE_WORD_RE = re.compile(
    rf"\b(?:[0-3]?\d)\s+(?:{MONTHS})\s+(?:19|20)\d{{2}}\b", re.I
)
DATE_NUM_RE = re.compile(
    r"\b(?:[0-3]?\d)[./-](?:[01]?\d)[./-](?:19|20)\d{2}\b"
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
VERIFIABLE = {"SOURCE_FACT", "LEGAL_SOURCE", "INFERENCE"}
EXEMPT = {"RESEARCH_NEEDED", "HYPOTHESIS"}
TARGET_SUPPORT_RATIO = 0.95


def parse_json(raw):
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("verification model did not return JSON")
    return json.loads(raw[start:end + 1])


def citation_lines(citations, source_lines):
    seen = set()
    chunks = []
    for token in citations:
        m = CITE_RE.fullmatch(str(token).strip())
        if not m:
            raise RuntimeError(f"invalid atomic citation: {token}")
        a = int(m.group(1))
        b = int(m.group(2) or m.group(1))
        if a < 1 or b < a or b > len(source_lines):
            raise RuntimeError(f"atomic citation out of range: {token}")
        for n in range(a, b + 1):
            if n not in seen:
                seen.add(n)
                chunks.append(f"[L{n:04d}] {source_lines[n-1]}")
    return "\n".join(chunks)


def extract_atomic_claims(model, report):
    system, user = atomic_claim_extract(report)
    raw, meta = chat(
        model, system, user,
        num_predict=3200, num_ctx=32768, temperature=0.0,
        seed=31001, timeout=1800, keep_alive="10m",
    )
    data = parse_json(raw)
    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        raise RuntimeError("atomic extractor returned no claims")
    valid_kinds = VERIFIABLE | EXEMPT
    ids = set()
    for item in claims:
        cid = item.get("id")
        if not isinstance(cid, int) or cid in ids:
            raise RuntimeError(f"invalid/duplicate atomic claim id: {cid}")
        ids.add(cid)
        if item.get("kind") not in valid_kinds:
            raise RuntimeError(f"invalid atomic claim kind: {item.get('kind')}")
        cites = item.get("citations")
        if not isinstance(cites, list):
            raise RuntimeError(f"claim {cid} citations are not a list")
        if item["kind"] in VERIFIABLE and not cites:
            raise RuntimeError(f"verifiable claim {cid} has no citation")
        for cite in cites:
            if cite not in report:
                raise RuntimeError(f"atomic extractor invented citation {cite}")
    return claims, meta

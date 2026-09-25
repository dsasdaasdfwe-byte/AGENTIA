#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents"))

from fact_ledger import (
    anchor_quotes,
    generate_validated_block,
    section_schema,
    validate_section,
)
from semantic_gate import deterministic_violations


class GroundingGateTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            "Le Conseil alpha rend une décision le 3 mars 2025.",
            "Mme Beta recourt contre cette décision le 10 mars 2025.",
            "La Cour annule la décision et renvoie la cause au Conseil alpha.",
        ]
        self.ledger = {
            "facts": [
                {"subject": "Conseil alpha"},
                {"subject": "Mme Beta"},
            ],
            "procedure_edges": [
                {"actor": "La Cour", "target": "Conseil alpha"},
            ],
            "issues": [],
        }

    def test_supported_date_and_actor(self):
        claims = [{
            "id": 1,
            "claim": "Le Conseil alpha rend une décision le 3 mars 2025.",
            "citations": ["[L0001]"],
            "kind": "SOURCE_FACT",
        }]
        self.assertEqual(
            deterministic_violations(claims, self.source, self.ledger),
            [],
        )

    def test_wrong_date_is_rejected(self):
        claims = [{
            "id": 1,
            "claim": "Le Conseil alpha rend une décision le 10 mars 2025.",
            "citations": ["[L0001]"],
            "kind": "SOURCE_FACT",
        }]
        violations = deterministic_violations(claims, self.source, self.ledger)
        self.assertTrue(any(v["status"] == "UNSUPPORTED_DATE" for v in violations))

    def test_wrong_actor_is_rejected(self):
        claims = [{
            "id": 1,
            "claim": "Mme Beta rend une décision le 3 mars 2025.",
            "citations": ["[L0001]"],
            "kind": "SOURCE_FACT",
        }]
        violations = deterministic_violations(claims, self.source, self.ledger)
        self.assertTrue(any(v["status"] == "UNSUPPORTED_ENTITY" for v in violations))


class FactLedgerSplitTests(unittest.TestCase):
    def setUp(self):
        self.source = [
            "Le Conseil alpha rend une décision le 3 mars 2025.",
            "Mme Beta recourt contre cette décision le 10 mars 2025.",
            "La Cour annule la décision et renvoie la cause au Conseil alpha.",
            "Le Conseil alpha confirme la réception du dossier.",
            "Mme Beta est partie à la procédure.",
        ]
        self.facts_block = {
            "facts": [
                {
                    "id": "F001", "subject": "Conseil alpha",
                    "predicate": "rend", "object": "une décision",
                    "date_text": "3 mars 2025", "source_lines": [1],
                    "quote": "Le Conseil alpha rend une décision le 3 mars 2025.",
                },
                {
                    "id": "F002", "subject": "Mme Beta",
                    "predicate": "recourt", "object": "contre cette décision",
                    "date_text": "10 mars 2025", "source_lines": [2],
                    "quote": "Mme Beta recourt contre cette décision le 10 mars 2025.",
                },
                {
                    "id": "F003", "subject": "La Cour",
                    "predicate": "annule", "object": "la décision",
                    "date_text": None, "source_lines": [3],
                    "quote": "La Cour annule la décision et renvoie la cause au Conseil alpha.",
                },
                {
                    "id": "F004", "subject": "Conseil alpha",
                    "predicate": "confirme", "object": "la réception du dossier",
                    "date_text": None, "source_lines": [4],
                    "quote": "Le Conseil alpha confirme la réception du dossier.",
                },
                {
                    "id": "F005", "subject": "Mme Beta",
                    "predicate": "est", "object": "partie à la procédure",
                    "date_text": None, "source_lines": [5],
                    "quote": "Mme Beta est partie à la procédure.",
                },
            ]
        }

    def test_split_schema_uses_bounded_fact_limits(self):
        normal = section_schema("facts")
        compressed = section_schema("facts", compressed=True)
        self.assertEqual(normal["required"], ["facts"])
        self.assertEqual(normal["properties"]["facts"]["maxItems"], 15)
        self.assertEqual(compressed["properties"]["facts"]["maxItems"], 12)

    def test_section_validation_rejects_non_source_quote(self):
        broken = json.loads(json.dumps(self.facts_block))
        broken["facts"][0]["quote"] = "Citation inventée qui n'existe pas dans la source."
        with self.assertRaisesRegex(RuntimeError, "quote is not contained"):
            validate_section(broken, self.source, "facts")

    def test_quote_anchoring_uses_exact_cited_source(self):
        repaired = json.loads(json.dumps(self.facts_block))
        repaired["facts"][0]["quote"] = "Le Conseil alpha a rendu sa décision."
        rewrites = anchor_quotes(repaired, self.source, "facts")
        self.assertEqual(rewrites, 1)
        self.assertEqual(repaired["facts"][0]["quote"], self.source[0])
        validate_section(repaired, self.source, "facts")

    @patch("fact_ledger._call_section_model")
    def test_length_truncation_retries_in_compression_mode(self, mocked_call):
        mocked_call.side_effect = [
            ("{", {"done_reason": "length", "eval_count": 2000}),
            (json.dumps(self.facts_block), {"done_reason": "stop", "eval_count": 700}),
        ]
        entries, attempts = generate_validated_block(
            "qwen3.5:9b",
            "\n".join(self.source),
            self.source,
            "facts",
            stage="draft",
        )
        self.assertEqual(len(entries), 5)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["compressed"])
        self.assertTrue(attempts[1]["compressed"])
        self.assertTrue(mocked_call.call_args_list[1].kwargs["compressed"])


if __name__ == "__main__":
    unittest.main()

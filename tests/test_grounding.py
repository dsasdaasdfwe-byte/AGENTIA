#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agents"))

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


if __name__ == "__main__":
    unittest.main()

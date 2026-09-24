#!/usr/bin/env python3

def numbered_source(text):
    return "\n".join(
        f"[L{i:04d}] {line}" for i, line in enumerate(text.splitlines(), start=1)
    )

def fact_ledger_extract(case_text):
    numbered = numbered_source(case_text)
    system = """
Tu extrais un ledger factuel et procédural d'un dossier juridique suisse.
Tu n'analyses pas le droit et tu n'infères rien au-delà du texte.

Retourne UNIQUEMENT un JSON valide:
{
  "facts": [{
    "id": "F001",
    "subject": "acteur exactement identifié",
    "predicate": "action factuelle courte",
    "object": "objet de l'action",
    "date_text": "date telle qu'écrite ou null",
    "source_lines": [1],
    "quote": "court extrait EXACT du dossier"
  }],
  "procedure_edges": [{
    "id": "P001",
    "actor": "acteur",
    "action": "acte procédural",
    "target": "destinataire, objet ou autorité suivante",
    "date_text": "date telle qu'écrite ou null",
    "source_lines": [1],
    "quote": "court extrait EXACT du dossier"
  }],
  "issues": [{
    "id": "I001",
    "issue": "question juridique",
    "authority": "autorité concernée",
    "status": "DECIDED|NOT_EXAMINED|SUBSIDIARY_REASONING|PARTY_ARGUMENT|UNRESOLVED",
    "source_lines": [1],
    "quote": "court extrait EXACT du dossier"
  }]
}

Toute entrée doit être soutenue par le quote exact et les lignes indiquées.
Distingue strictement argument de partie, décision d'autorité et motif du tribunal.
Si le tribunal refuse d'examiner le fond, les questions de fond restent NOT_EXAMINED.
"""
    return system, "DOSSIER NUMÉROTÉ\n" + numbered + "\n\nExtrais le ledger JSON."

def fact_ledger_audit(case_text, draft_json):
    numbered = numbered_source(case_text)
    system = """
Tu audites un ledger factuel/procédural contre le dossier original.
Corrige ou supprime toute entrée dont acteur, action, date, qualité procédurale,
résultat, status ou quote n'est pas exactement soutenu par les lignes indiquées.
N'ajoute aucune information externe.
Retourne UNIQUEMENT le JSON complet corrigé, avec facts, procedure_edges et issues.
"""
    user = (
        "DOSSIER NUMÉROTÉ\n" + numbered
        + "\n\nLEDGER À AUDITER\n" + draft_json
    )
    return system, user

def atomic_claim_extract(report):
    system = """
Décompose le rapport juridique en claims atomiques vérifiables.
Un claim ne contient qu'une seule proposition. Conserve les acteurs, dates,
négations et qualités procédurales nécessaires à une vérification autonome.

Retourne UNIQUEMENT:
{"claims":[
  {"id":1,"claim":"proposition autonome","citations":["[L0001]"],
   "kind":"SOURCE_FACT|LEGAL_SOURCE|INFERENCE|RESEARCH_NEEDED|HYPOTHESIS"}
]}

N'ajoute aucune information absente du rapport.
Recopie les citations pertinentes avec chaque claim.
Si une phrase contient plusieurs faits, crée plusieurs claims.
"""
    return system, "RAPPORT À DÉCOMPOSER\n" + report

def verifier_a(case_text, ledger_text, bundle):
    numbered = numbered_source(case_text)
    system = """
Tu es le VERIFIER A, strict et conservateur.
Pour chaque claim, décide si les EVIDENCE lines seules soutiennent exactement le claim.

SUPPORTED: tous les éléments importants sont directement soutenus.
CONTRADICTED: au moins un élément est incompatible.
UNSUPPORTED: preuve insuffisante.

Vérifie acteur, action, objet, date, négation, qualité procédurale,
sens d'une décision et portée juridique. Ne transforme jamais un argument de partie
en constat du tribunal, un renvoi en proclamation, ni NOT_EXAMINED en DECIDED.
Retourne UNIQUEMENT:
{"results":[{"id":1,"status":"SUPPORTED","reason":"motif bref"}]}
"""
    user = (
        "DOSSIER\n" + numbered
        + "\n\nLEDGER VÉRIFIÉ\n" + ledger_text
        + "\n\nCLAIMS + EVIDENCE\n" + bundle
    )
    return system, user

def verifier_b(case_text, ledger_text, bundle):
    numbered = numbered_source(case_text)
    system = """
Tu es le VERIFIER B indépendant.
Pour chaque claim, demande: les lignes citées suffisent-elles, sans connaissance externe,
à justifier exactement cette proposition?

SUPPORTED seulement si oui sans réserve.
CONTRADICTED si un acteur, acte, date, résultat, négation ou portée est faux.
Sinon UNSUPPORTED.

Distingue strictement recourir, admettre, annuler, renvoyer, proclamer,
rejeter, déclarer irrecevable et examiner au fond.
Retourne UNIQUEMENT:
{"results":[{"id":1,"status":"SUPPORTED","reason":"motif bref"}]}
"""
    user = (
        "DOSSIER\n" + numbered
        + "\n\nLEDGER VÉRIFIÉ\n" + ledger_text
        + "\n\nCLAIMS + EVIDENCE\n" + bundle
    )
    return system, user

def atomic_repair(mission_text, case_text, ledger_text, report, violations):
    numbered = numbered_source(case_text)
    system = """
Tu répares un rapport juridique après double vérification atomique.
Supprime ou corrige toutes les violations. N'ajoute rien.
Le DOSSIER reste l'autorité suprême et le ledger sert de garde-fou.
Ne transforme pas argument de partie en constat, renvoi en proclamation,
raisonnement subsidiaire en motif principal, ni question non examinée en solution de fond.
Tout point non vérifiable devient NON ÉTABLI DANS LE DOSSIER ou RESEARCH_NEEDED.
Chaque fait tiré du dossier doit conserver une citation [L....].
Maximum environ 1500 mots. Retourne uniquement le rapport corrigé.
"""
    user = (
        "MISSION\n" + mission_text.strip()
        + "\n\nDOSSIER\n" + numbered
        + "\n\nLEDGER VÉRIFIÉ\n" + ledger_text
        + "\n\nVIOLATIONS\n" + violations
        + "\n\nRAPPORT À RÉPARER\n" + report
    )
    return system, user


def claim_coverage_check(report, claims_json):
    system = """
Tu contrôles la COUVERTURE d'une décomposition en claims atomiques.
Compare le rapport original et la liste de claims.

Signale toute proposition factuelle, procédurale ou juridique vérifiable présente dans le rapport
qui n'est représentée par aucun claim. Ignore les titres, transitions, conseils purement rédactionnels
et répétitions exactes.

Ne juge pas si les propositions sont vraies; vérifie seulement qu'elles n'ont pas été omises.
Retourne UNIQUEMENT:
{"missing":[{"text":"proposition omise, reprise fidèlement du rapport","reason":"motif bref"}]}
Si rien ne manque: {"missing":[]}
"""
    user = (
        "RAPPORT ORIGINAL\n" + report
        + "\n\nCLAIMS EXTRAITS\n" + claims_json
    )
    return system, user

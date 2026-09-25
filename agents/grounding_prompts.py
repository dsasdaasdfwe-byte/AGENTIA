#!/usr/bin/env python3

SECTION_LIMITS = {
    "facts": {"max": 15, "compressed_max": 12},
    "procedure_edges": {"max": 12, "compressed_max": 9},
    "issues": {"max": 8, "compressed_max": 6},
}


def numbered_source(text):
    return "\n".join(
        f"[L{i:04d}] {line}" for i, line in enumerate(text.splitlines(), start=1)
    )


def _section_limit(section, compressed):
    if section not in SECTION_LIMITS:
        raise ValueError(f"unknown ledger section: {section}")
    key = "compressed_max" if compressed else "max"
    return SECTION_LIMITS[section][key]


def fact_ledger_extract_section(case_text, section, compressed=False):
    numbered = numbered_source(case_text)
    max_items = _section_limit(section, compressed)
    compression = """
MODE COMPRESSION STRICTE:
- garde uniquement les éléments indispensables;
- formulations très courtes, aucune répétition;
- n'ajoute aucun détail secondaire.
""" if compressed else ""

    if section == "facts":
        system = f"""
Tu extrais UNIQUEMENT les faits matériels d'un dossier juridique suisse.
Tu n'analyses pas le droit et tu n'infères rien au-delà du texte.

Retourne UNIQUEMENT un JSON valide sous cette forme:
{{"facts":[{{
  "id":"F001",
  "subject":"acteur exactement identifié",
  "predicate":"action factuelle courte",
  "object":"objet de l'action",
  "date_text":"date telle qu'écrite ou null",
  "source_lines":[1]
}}]}}

CONTRAINTES:
- maximum {max_items} faits matériellement importants;
- couvre en priorité chronologie, identités, actes, résultats et dates déterminants;
- source_lines doit viser les lignes qui soutiennent directement toute l'entrée;
- ne produis PAS de champ quote: Python reconstruira l'extrait exact depuis source_lines;
- aucun doublon, aucune paraphrase répétée, aucune information externe;
- JSON compact, aucune explication hors de l'objet.
{compression}
"""
    elif section == "procedure_edges":
        system = f"""
Tu extrais UNIQUEMENT le graphe procédural d'un dossier juridique suisse.
Tu n'analyses pas le droit et tu n'infères rien au-delà du texte.

Retourne UNIQUEMENT un JSON valide sous cette forme:
{{"procedure_edges":[{{
  "id":"P001",
  "actor":"acteur exactement identifié",
  "action":"acte procédural précis",
  "target":"destinataire, objet ou autorité suivante",
  "date_text":"date telle qu'écrite ou null",
  "source_lines":[1]
}}]}}

CONTRAINTES:
- maximum {max_items} transitions procédurales importantes;
- distingue strictement décider, recourir, admettre, annuler, renvoyer, proclamer,
  rejeter et déclarer irrecevable;
- source_lines doit viser les lignes qui soutiennent directement toute l'entrée;
- ne produis PAS de champ quote: Python reconstruira l'extrait exact depuis source_lines;
- aucun doublon, aucune information externe;
- JSON compact, aucune explication hors de l'objet.
{compression}
"""
    else:
        system = f"""
Tu extrais UNIQUEMENT les états des questions juridiques d'un dossier suisse.
Tu dois distinguer ce qui a été décidé, non examiné, traité subsidiairement,
seulement soutenu par une partie, ou laissé non résolu.

Retourne UNIQUEMENT un JSON valide sous cette forme:
{{"issues":[{{
  "id":"I001",
  "issue":"question juridique précise",
  "authority":"autorité concernée",
  "status":"DECIDED|NOT_EXAMINED|SUBSIDIARY_REASONING|PARTY_ARGUMENT|UNRESOLVED",
  "source_lines":[1]
}}]}}

CONTRAINTES:
- maximum {max_items} questions matériellement importantes;
- ne transforme jamais un argument de partie en constat de l'autorité;
- si le tribunal refuse d'examiner le fond, la question de fond reste NOT_EXAMINED;
- un raisonnement expressément subsidiaire reste SUBSIDIARY_REASONING;
- source_lines doit viser les lignes qui soutiennent directement toute l'entrée;
- ne produis PAS de champ quote: Python reconstruira l'extrait exact depuis source_lines;
- aucun doublon, aucune information externe;
- JSON compact, aucune explication hors de l'objet.
{compression}
"""

    return system, (
        "DOSSIER NUMÉROTÉ\n" + numbered
        + f"\n\nExtrais uniquement la section {section}."
    )


def fact_ledger_audit_section(case_text, section, draft_json, compressed=False):
    numbered = numbered_source(case_text)
    max_items = _section_limit(section, compressed)
    compression = """
MODE COMPRESSION STRICTE:
- réduis encore la section aux éléments indispensables;
- supprime doublons et détails secondaires;
- garde des formulations très courtes.
""" if compressed else ""

    section_rules = {
        "facts": (
            "Vérifie spécialement acteur, action, objet et date. "
            "Ne conserve aucun fait qui dépasse le texte cité."
        ),
        "procedure_edges": (
            "Vérifie spécialement acteur, acte procédural, cible, résultat et date. "
            "Ne confonds jamais renvoi et proclamation, ni irrecevabilité et décision au fond."
        ),
        "issues": (
            "Vérifie spécialement le statut DECIDED / NOT_EXAMINED / "
            "SUBSIDIARY_REASONING / PARTY_ARGUMENT / UNRESOLVED. "
            "Ne transforme jamais l'argument d'une partie en constat d'une autorité."
        ),
    }
    if section not in section_rules:
        raise ValueError(f"unknown ledger section: {section}")

    system = f"""
Tu audites UNIQUEMENT la section {section} d'un ledger juridique contre le dossier original.
Corrige ou supprime toute entrée dont acteur, action, date, qualité procédurale,
résultat, statut ou source_lines n'est pas exactement soutenu.
N'ajoute aucune information externe et n'élargis pas la portée des formulations.
{section_rules[section]}
Ne produis PAS de champ quote: Python reconstruira l'extrait exact depuis source_lines.
Conserve au maximum {max_items} entrées et retourne UNIQUEMENT l'objet JSON complet
pour cette section, sans explication.
{compression}
"""
    user = (
        "DOSSIER NUMÉROTÉ\n" + numbered
        + f"\n\nSECTION {section} À AUDITER\n" + draft_json
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

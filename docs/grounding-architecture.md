# AGENTIA — architecture de grounding factuel

## Objectif

Réduire les hallucinations de rapports juridiques longs en séparant strictement:

1. extraction des faits;
2. analyse;
3. synthèse;
4. décomposition en claims atomiques;
5. vérification indépendante;
6. réparation;
7. publication seulement si les gates passent.

Le seuil de 95% dans AGENTIA est un **score interne de support atomique**:
au moins 95% des claims vérifiables doivent être jugés SUPPORTED par les deux
vérificateurs, avec zéro contradiction et zéro violation déterministe résiduelle.
Ce n'est pas, à lui seul, une preuve empirique de 95% d'exactitude juridique réelle.
Cette dernière exige un benchmark annoté indépendamment.

## Pourquoi cette architecture

### Décomposer avant de vérifier

SAFE / LongFact (Wei et al., 2024) et VeriScore (Song et al., 2024) évaluent
la factualité longue en décomposant les réponses en faits ou claims atomiques,
puis en vérifiant chaque unité.

- https://arxiv.org/abs/2403.18802
- https://aclanthology.org/2024.findings-emnlp.552/

Claimify (Metropolitansky & Larson, ACL 2025) montre que la qualité de
l'extraction des claims elle-même est critique, notamment l'atomicité,
la couverture et la décontextualisation.

- https://aclanthology.org/2025.acl-long.348/

Lu et al. (ACL 2025) montrent aussi que la politique de décomposition et le
verifier doivent être alignés; une mauvaise atomicité dégrade la vérification.

- https://aclanthology.org/2025.acl-long.254/

### Vérifier indépendamment de la génération

Chain-of-Verification (Dhuliawala et al., 2023) réduit les hallucinations en
séparant le brouillon des questions/réponses de vérification, afin d'éviter que
le modèle ne se contente de rationaliser sa première réponse.

- https://arxiv.org/abs/2309.11495

AGENTIA adapte ce principe avec deux passes:
- verifier A: entailment strict claim -> evidence;
- verifier B: lecture indépendante vrai/faux fondée seulement sur les lignes citées.

Les deux utilisent des prompts et seeds distincts. Ils restent le même modèle local:
ce sont donc deux **passes indépendantes**, pas deux modèles statistiquement indépendants.

### Réviser au lieu de simplement signaler

RARR (Gao et al., 2022) combine attribution et révision des passages non soutenus.
AGENTIA applique la même idée localement: une violation déclenche une réécriture,
puis une nouvelle décomposition et une nouvelle double vérification.

- https://arxiv.org/abs/2210.08726

### Le RAG/citation seul ne suffit pas en droit

L'évaluation Stanford de systèmes de recherche juridique assistée par IA a montré
que des systèmes spécialisés et fondés sur la recherche peuvent encore produire
des erreurs substantielles. Une citation existante ne garantit donc pas que la
proposition est réellement autorisée par la source.

- https://law.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools/

Le concept de claim-authority warrant (Taranukhin & Shwartz, 2026) formalise
précisément ce problème: une proposition juridique conséquente doit être soutenue
par une autorité qui existe, s'applique, a le statut représenté et autorise
effectivement la proposition formulée.

- https://arxiv.org/abs/2609.17546

## Pipeline AGENTIA V8

SOURCE privée
  -> fact_ledger.py
     -> extraction structurée
     -> audit indépendant
     -> validation déterministe des quotes/lignes/dates
     -> fact-ledger.json
     -> graphe procédural
     -> états DECIDED / NOT_EXAMINED / SUBSIDIARY_REASONING / PARTY_ARGUMENT
  -> 20 agents locaux
  -> 4 panel reviewers locaux
  -> senior reviewer local
  -> décomposition atomique du rapport final
  -> contrôles déterministes dates + acteurs + provenance des citations
  -> verifier A
  -> verifier B
  -> réparation si nécessaire
  -> nouvelle décomposition + nouvelle double vérification
  -> fail closed si:
       * support double < 0.95
       * une contradiction subsiste
       * une violation déterministe subsiste
  -> final-report.md + fidelity-metrics.json

## Principes de sécurité factuelle

- Le dossier original reste toujours supérieur au ledger et aux rapports d'agents.
- Le ledger n'est pas une nouvelle source; il est une structure de navigation auditée.
- Aucun consensus entre agents ne transforme une proposition en fait.
- Toute date factuelle doit apparaître dans les lignes citées.
- Tout acteur connu utilisé dans un claim doit être compatible avec les lignes citées.
- Une citation atomique ne peut utiliser que des lignes déjà citées par le rapport.
- Un renvoi n'est pas une proclamation.
- Une déclaration d'irrecevabilité n'est pas une décision de fond.
- Un argument d'une partie n'est pas un constat du tribunal.
- Les règles externes absentes du dossier restent RESEARCH_NEEDED.

## Mesure future de "95% réel"

Pour revendiquer une précision réelle de 95%, il faut constituer un corpus de dossiers
avec claims annotés humainement, puis mesurer au minimum:

- précision des claims supportés;
- rappel des erreurs importantes;
- exactitude acteur/action/date/objet;
- exactitude du statut procédural;
- exactitude du claim-authority warrant;
- taux d'abstention correcte;
- résultats par classe de risque.

Le score produit par fidelity-metrics.json sert de gate opérationnel interne,
pas de remplacement à cette évaluation indépendante.

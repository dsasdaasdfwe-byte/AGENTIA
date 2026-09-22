# AGENTIA

Moteur multi-agent juridique local exécuté sur GitHub Actions.

## Architecture

- 20 agents IA en parallèle.
- 1 job GitHub Actions = 1 VM = 1 agent.
- Modèle agents : qwen3.5:4b.
- Senior Reviewer : qwen3.5:9b.
- Ollama local sur chaque runner.
- Aucun dossier juridique n'est stocké dans le dépôt.

## Déclenchement

Le workflow est manuel : Actions → Run 20 AI Agents → Run workflow.

Avant le premier run, créer dans Settings → Secrets and variables → Actions :

- AGENTIA_CASE : dossier / faits / procédure / sources à analyser.
- AGENTIA_MISSION : mission optionnelle. Si absent, une mission juridique générique est utilisée.

Le contenu de ces secrets n'est pas committé.

## Topologie

Le job warm-model-cache prépare le modèle explorateur une seule fois. Ensuite les 20 jobs agents sont lancés avec max-parallel: 20. Après leur terminaison, un job séparé télécharge les 20 rapports et exécute le Senior Reviewer.

Les artefacts de sortie sont conservés 3 jours.

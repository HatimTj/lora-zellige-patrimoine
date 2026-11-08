# Fiche des donnees

## Resume

| Element | Valeur |
|---|---|
| Concept | Zellige marocain / motifs geometriques |
| Source principale | Wikimedia Commons (API, licences recuperees automatiquement) |
| Sources complementaires | Photographies de l'equipe |
| Nombre d'images retenues | (a completer - cible 45) |
| Dont reservees au test de generalisation | 8 (jamais vues a l'entrainement) |
| Resolution finale | 512 x 512 |
| Poids total | ~15 Mo (non versionne dans Git) |

## Comment regenerer le jeu

```bash
python src/collect_wikimedia.py --config configs/base.yaml   # telechargement + licences
python src/prepare_data.py --config configs/base.yaml        # recadrage, dedoublonnage, legendes
```

Aucune image n'est stockee dans Git. Seuls `licenses.csv` et `metadata.jsonl`
sont versionnes, ce qui suffit a reconstruire le jeu a l'identique.

## Criteres d'inclusion

- Un panneau de zellige, ou un motif geometrique traditionnel, nettement visible
- Prise de vue frontale ou quasi frontale
- Licence compatible : CC0, CC BY, CC BY-SA, ou domaine public

## Criteres d'exclusion

- Personnes identifiables
- Filigrane, logo, texte incruste
- Flou, forte perspective, faible luminosite
- Doublon ou quasi-doublon (detection par hachage perceptuel)
- Licence absente, inconnue, ou non commerciale ambigue

## Tracabilite des licences

`licenses.csv` contient une ligne par fichier **telecharge**, y compris les
fichiers rejetes (colonnes `conserve` et `raison_rejet`), avec auteur, URL
source et licence.

## Legendes

Format `metadata.jsonl`, une ligne par image :

```json
{"file_name": "zlj_001.jpg", "text": "a zlj_style zellige tile panel, blue and white, eight-pointed star pattern, flat frontal view"}
```

Brouillon genere automatiquement (BLIP), puis **relu et corrige a la main**.
Toutes les legendes contiennent le token `zlj_style`.

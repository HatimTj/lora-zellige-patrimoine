# Fiche des donnees

## Resume

| Element | Valeur |
|---|---|
| Concept | Zellige marocain, panneaux en couleur vus de face |
| Source principale | Wikimedia Commons (API, licences recuperees automatiquement) |
| Candidats examines | 367 |
| Images telechargees | 120 |
| **Retenues apres tri manuel** | **37** |
| Dont reservees au test de generalisation | 8 (jamais vues a l'entrainement) |
| Resolution finale | 512 x 512 |

### Licences des images retenues

| Licence | Nombre |
|---|---|
| CC BY-SA 4.0 | 21 |
| CC BY-SA 3.0 | 6 |
| CC BY 2.0 | 5 |
| CC BY 4.0 | 3 |
| CC BY-SA 2.5 | 1 |
| Public domain | 1 |

Toutes sont des licences libres (CC0, CC BY, CC BY-SA). Le detail par image,
auteur et URL compris, est dans `licenses.csv`.

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

# Feuille de route et taches

Equipe : **Hatim Tajimi**, **Houssam El Moutaqui**, **Saad Belekbira**.

> La repartition nominative des taches sera arretee plus tard et presentee a la
> soutenance. Ce document sert pour l'instant de feuille de route : une ligne =
> une tache = une issue GitHub, assignee au moment ou elle est prise.

## Taches par etape

### Etape 0 - Cadrage

| Fait | Tache | Livrable |
|---|---|---|
| x | Cadrage, arborescence, conventions | `README.md`, `configs/base.yaml` |
| | x Socle commun : lecture de config, seeds, mesure VRAM et temps | `src/common.py` |

### Etape 1 - Jeu de donnees

| Fait | Tache | Livrable |
|---|---|---|
| | x Collecte Wikimedia Commons + licences automatiques | `src/collect_wikimedia.py`, `data/licenses.csv` |
| | x Tri manuel des images (inclusion / exclusion) | `data/README.md` complete |
| | x Pretraitement 512x512 + dedoublonnage perceptuel | `src/prepare_data.py` |
| | x Legendes : brouillon BLIP puis relecture manuelle | `data/metadata.jsonl` |
| | x Les 10 prompts d'evaluation, figes avant tout entrainement | `configs/eval_prompts.yaml` |

### Etape 2 - Baseline et metriques

| Fait | Tache | Livrable |
|---|---|---|
| | x Metriques : CLIPScore, CLIP-I, DINOv2, LPIPS | `src/metrics.py` |
| | Baseline : SD 1.5 non modifie sur les 10 prompts | `results/baseline/` |

### Etape 3 - Premiere LoRA

| Fait | Tache | Livrable |
|---|---|---|
| | Script d'entrainement LoRA | `src/train_lora.py` |
| | Premiere LoRA fonctionnelle (rang 8, 1000 pas) | `notebooks/03_lora_train.ipynb` |

### Etape 4 - Experiences

| Fait | Tache | Livrable |
|---|---|---|
| | Balayage des rangs 4 / 8 / 16 / 32 | lignes dans `results/results.csv` |
| | Balayage du nombre d'etapes 250 / 500 / 1000 / 2000 | lignes dans `results/results.csv` |
| | Balayage de l'intensite LoRA (0 a 1,25) | lignes dans `results/results.csv` |
| | Les 4 figures du rapport | `results/figures/` |

### Etape 5 - Surapprentissage

| Fait | Tache | Livrable |
|---|---|---|
| | Test de memorisation (plus proche voisin) | `src/overfit_check.py` |
| | Test de degradation hors domaine | `results/figures/hors_domaine.png` |

### Etape 6 - Demo

| Fait | Tache | Livrable |
|---|---|---|
| | Application Gradio (prompt + slider d'intensite + avant/apres) | `demo/app.py` |
| | Video de demonstration (60-90 s) | `demo/demo.mp4` |

### Etape 7 - Rapport et soutenance

| Fait | Tache | Livrable |
|---|---|---|
| | Fiche du modele | `docs/model_card.md` |
| | Reflexion ethique et verification des licences | `docs/ethics.md` |
| | Rapport 6-8 pages | `report/rapport.pdf` |
| | Support de soutenance | `report/soutenance.pdf` |

## Ordre des dependances

Ce qui bloque quoi, pour ne pas attendre inutilement :

```
Socle commun          ->  tous les autres scripts
Collecte + tri        ->  pretraitement  ->  legendes  ->  entrainement
Metriques + prompts    ->  baseline  ->  comparaisons de l'Etape 4
Balayages (Etape 4)   ->  figures
```

Les metriques et les 10 prompts ne dependent d'**aucune image** : ils peuvent
etre ecrits en parallele de la collecte.

## Decoupage de la soutenance (10 min)

| Temps | Contenu |
|---|---|
| 1 min | Probleme et objectif |
| 1 min | Donnees, licences, legendes |
| 2 min | LoRA : principe, rang, protocole |
| 3 min | Resultats : compromis et figures |
| 1 min | Demo (video) |
| 1 min | Surapprentissage et limites |
| 1 min | Ethique et conclusion |

## Convention de commits

`<etape>: <verbe a l'infinitif> <objet>`

Exemples :
- `etape1: ajouter le script de collecte Wikimedia`
- `etape4: corriger le calcul de CLIPScore sur les prompts hors domaine`

Un commit par tache terminee, au minimum. Le depot doit montrer une activite
hebdomadaire reguliere.

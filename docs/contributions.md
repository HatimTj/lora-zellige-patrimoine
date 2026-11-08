# Repartition des taches

Equipe : **Hatim Tajimi**, **Houssam El Moutaqui**, **Saad Belekbira**.

Regle de travail : une issue GitHub par ligne du tableau ci-dessous, assignee a
une personne. Le code d'une tache est commite par la personne assignee, pour que
la contribution soit visible dans l'historique Git (consigne du sujet).

## Tableau des taches

| # | Tache | Livrable | Assigne a | Etape |
|---|---|---|---|---|
| 1 | Collecte Wikimedia Commons + licences | `src/collect_wikimedia.py`, `data/licenses.csv` | Hatim | 1 |
| 2 | Tri manuel des images (inclusion/exclusion) | `data/README.md` complete | Hatim | 1 |
| 3 | Pretraitement 512x512 + dedoublonnage | `src/prepare_data.py` | Hatim | 1 |
| 4 | Legendes (brouillon BLIP puis relecture) | `data/metadata.jsonl` | Hatim | 1 |
| 5 | Les 10 prompts d'evaluation figes | `configs/eval_prompts.yaml` | Saad | 1 |
| 6 | Implementation des metriques | `src/metrics.py` | Saad | 2 |
| 7 | Baseline SD 1.5 non modifie | `results/baseline/` | Saad | 2 |
| 8 | Script d'entrainement LoRA | `src/train_lora.py` | Houssam | 3 |
| 9 | Premiere LoRA fonctionnelle (r=8) | `notebooks/03_lora_train.ipynb` | Houssam | 3 |
| 10 | Balayage des rangs 4/8/16/32 | `results/results.csv` | Houssam | 4 |
| 11 | Balayage du nombre d'etapes | `results/results.csv` | Houssam | 4 |
| 12 | Balayage de l'intensite LoRA | `results/results.csv` | Saad | 4 |
| 13 | Figures : compromis, grille, intensite | `results/figures/` | Saad | 4 |
| 14 | Test de memorisation (plus proche voisin) | `src/overfit_check.py` | Saad | 5 |
| 15 | Test de degradation hors domaine | `results/figures/` | Houssam | 5 |
| 16 | Application Gradio | `demo/app.py` | Saad | 6 |
| 17 | Video de demonstration (60-90 s) | `demo/demo.mp4` | Houssam | 6 |
| 18 | Fiche du modele | `docs/model_card.md` | Houssam | 7 |
| 19 | Reflexion ethique et licences | `docs/ethics.md` | Hatim | 7 |
| 20 | Rapport 6-8 pages | `report/rapport.pdf` | Les trois | 7 |
| 21 | Support de soutenance (10 min) | `report/soutenance.pdf` | Les trois | 7 |

## Decoupage de la soutenance (10 min)

| Temps | Contenu | Qui |
|---|---|---|
| 1 min | Probleme et objectif | Hatim |
| 1 min | Donnees, licences, legendes | Hatim |
| 2 min | LoRA : principe, rang, protocole | Houssam |
| 3 min | Resultats : compromis et figures | Saad |
| 1 min | Demo (video) | Saad |
| 1 min | Surapprentissage et limites | Houssam |
| 1 min | Ethique et conclusion | Hatim |

## Convention de commits

`<etape>: <verbe a l'infinitif> <objet>`

Exemples :
- `etape1: ajouter le script de collecte Wikimedia`
- `etape4: corriger le calcul de CLIPScore sur les prompts hors domaine`

Un commit par tache terminee, au minimum. Le depot doit montrer une activite
hebdomadaire reguliere, pas un depot massif la derniere semaine.

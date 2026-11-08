# Projet 5 - Specialiser un modele texte-image avec LoRA (zellige)

Adaptation de **Stable Diffusion 1.5** au style du **zellige marocain** par **LoRA**,
a partir de ~45 images sous licence libre, dans un budget de **12 Go de VRAM**
(Google Colab, GPU T4).

Le projet ne cherche pas seulement a produire de belles images : il mesure le
**compromis** entre fidelite au style, respect du prompt, cout de calcul et
surapprentissage, en fonction du **rang LoRA** et du **nombre d'etapes**.

## Questions de recherche

| # | Question | Mesure |
|---|---|---|
| Q1 | Le modele a-t-il appris le style du zellige ? | CLIP-I / DINOv2 vs images de reference |
| Q2 | Obeit-il encore au texte ? | CLIPScore (image <-> prompt) |
| Q3 | Combien cela coute-t-il ? | VRAM pic, temps/pas, taille du fichier LoRA |
| Q4 | A-t-il surappris ? | similarite au plus proche voisin, prompts hors domaine |

## Etat d'avancement

- [x] Etape 0 - Cadrage et depot
- [ ] Etape 1 - Jeu de donnees (collecte, licences, legendes)
- [ ] Etape 2 - Baseline SD 1.5 et metriques
- [ ] Etape 3 - Premiere LoRA fonctionnelle
- [ ] Etape 4 - Balayage rang x etapes x intensite
- [ ] Etape 5 - Analyse du surapprentissage
- [ ] Etape 6 - Demo Gradio + video
- [ ] Etape 7 - Rapport (6-8 p) et soutenance (10 min)

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
# Etape 1 : collecte des images (Wikimedia Commons, avec licences)
python src/collect_wikimedia.py --config configs/base.yaml

# Etape 1 : preparation (recadrage 512x512, dedoublonnage)
python src/prepare_data.py --config configs/base.yaml

# Etape 2 : baseline du modele non modifie
python src/generate.py --config configs/base.yaml --lora none --out results/baseline

# Etape 3/4 : entrainement d'une LoRA
python src/train_lora.py --config configs/lora_r8.yaml

# Etape 4 : evaluation complete -> results/results.csv
python src/evaluate.py --config configs/base.yaml

# Etape 6 : demo
python demo/app.py
```

## Structure

| Dossier | Contenu |
|---|---|
| `notebooks/` | Notebooks numerotes : donnees, baseline, entrainement, evaluation, demo |
| `src/` | Code reutilisable |
| `configs/` | Parametres des experiences (une config par run) |
| `data/` | Script de telechargement et fiche des donnees (aucune donnee lourde) |
| `results/` | Figures, tableaux et exemples generes |
| `demo/` | Application Gradio et video courte |
| `report/` | Rapport final |
| `docs/` | Licences, fiche du modele, reflexion ethique |

## Resultats principaux

> A completer a l'Etape 4.

## Decisions de cadrage

- **Un seul concept** : zellige / motifs geometriques. Melanger plusieurs styles
  empeche la LoRA de converger avec si peu d'images.
- **Token declencheur** : `zlj_style`, present dans toutes les legendes, pour
  appeler le style sans alterer le reste du modele.
- **SD 1.5 et non SDXL** : contrainte de 12 Go de VRAM.
- **Pas de FID** : avec ~45 images de reference, le FID est fortement biaise.
  On utilise CLIP-I + DINOv2 + inspection visuelle. Justifie dans le rapport.

## Equipe

| Membre | Role principal | Etapes pilotees |
|---|---|---|
| Hatim Tajimi | Donnees, licences et legendes | 1, et volet ethique de 7 |
| Houssam El Moutaqui | Entrainement LoRA et experiences | 3, 4 |
| Saad Belekbira | Metriques, evaluation et demo | 2, 5, 6 |

Chaque membre pilote ses etapes (il ouvre l'issue, ecrit le code, commite) mais
les trois relisent et signent le rapport de l'Etape 7. La repartition detaillee
est dans `docs/contributions.md`.

## Licence

Code sous licence MIT (voir `LICENSE`). Les images sources restent sous leurs
licences respectives, documentees dans `data/licenses.csv`.

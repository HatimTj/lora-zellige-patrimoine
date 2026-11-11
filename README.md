# LoRA Zellige — spécialiser un modèle texte-image sur le patrimoine marocain

**Projet 5** — Module IA génératives visuelles, EMSI 5ᵉ année.

Adaptation de **Stable Diffusion 1.5** au style du **zellige marocain** par
**LoRA**, à partir de ~45 images sous licence libre, dans un budget de
**12 Go de VRAM** (Google Colab, GPU T4).

Le projet ne cherche pas seulement à produire de belles images : il mesure le
**compromis** entre fidélité au style, respect du prompt, coût de calcul et
surapprentissage, en fonction du **rang LoRA** et du **nombre d'étapes
d'entraînement**.

---

## Sommaire

1. [Objectif et questions de recherche](#objectif-et-questions-de-recherche)
2. [Méthode](#méthode)
3. [Résultats principaux](#résultats-principaux)
4. [Démarrage rapide](#démarrage-rapide)
5. [Structure du dépôt](#structure-du-dépôt)
6. [Décisions de cadrage](#décisions-de-cadrage)
7. [Avancement](#avancement)
8. [Équipe](#équipe)
9. [Licence et éthique](#licence-et-éthique)

---

## Objectif et questions de recherche

Réentraîner entièrement un modèle texte-image est hors de portée d'un GPU
gratuit. **LoRA** contourne le problème : au lieu de modifier une matrice de
poids `W` de plusieurs millions de paramètres, on apprend deux petites matrices
`A` et `B` et on utilise `W + B·A`. Seules `A` et `B` sont entraînées, soit un
fichier de quelques mégaoctets au lieu de 4 Go.

La question n'est donc pas *« est-ce que ça marche »* mais *« à quel prix »* :

| # | Question | Mesure |
|---|---|---|
| **Q1** | Le modèle a-t-il appris le style du zellige ? | CLIP-I / DINOv2 vs images de référence |
| **Q2** | Obéit-il encore au texte ? | CLIPScore (image ↔ prompt) |
| **Q3** | Combien cela coûte-t-il ? | VRAM pic, temps par pas, taille du fichier LoRA |
| **Q4** | A-t-il surappris ? | similarité au plus proche voisin, prompts hors domaine |

L'hypothèse testée : **rang élevé et entraînement long donnent un style plus
fidèle, mais au prix du respect du prompt et au risque de mémoriser les images
d'entraînement.** Le projet cartographie ce compromis.

## Méthode

```
45 images Wikimedia          10 prompts figés
   (licences tracées)         + 4 seeds fixes
          │                         │
          ▼                         ▼
   recadrage 512²            baseline SD 1.5
   dédoublonnage              (avant LoRA)
   légendes « zlj_style »            │
          │                          │
          ▼                          │
   entraînement LoRA ──────────┐     │
   rangs 4 / 8 / 16 / 32       │     │
   pas  250 → 2000             │     │
          │                     ▼    ▼
          └──────────────►  évaluation
                            CLIPScore · CLIP-I
                            DINOv2 · LPIPS · coûts
                                   │
                                   ▼
                        compromis + démo Gradio
```

Le **token déclencheur** `zlj_style` est une chaîne sans sens en anglais,
présente dans toutes les légendes. Entraîner sur le mot `zellige` écraserait ce
que le modèle en sait déjà et rendrait la comparaison avant/après inexploitable.

### Environnement de référence

| Élément | Valeur |
|---|---|
| Modèle de base | `runwayml/stable-diffusion-v1-5` |
| Résolution | 512 × 512 |
| Précision | fp16, gradient checkpointing, Adam 8-bit |
| GPU | Google Colab T4 (16 Go, budget imposé : 12 Go) |
| Seeds d'évaluation | 1234, 2025, 777, 42 |

## Résultats principaux

Source : `results/results.csv`. À compléter au fil des Étapes 3 et 4.

| Configuration | Style (DINOv2) | Prompt domaine | Prompt hors domaine | Mémorisation | Diversité |
|---|---|---|---|---|---|
| **Baseline SD 1.5** | **0,274** | 34,27 | 33,14 | 0,737 | 0,646 |
| **LoRA r=8, 1000 pas** | **0,339** (+23,5 %) | 34,40 | 32,94 | 0,845 | 0,597 |
| LoRA r=8, 250 pas | 0,285 | 34,73 | 32,99 | 0,787 | 0,616 |
| LoRA r=8, 500 pas | 0,328 | 34,47 | 32,80 | 0,810 | 0,588 |
| LoRA r=8, 750 pas | 0,334 | 35,25 | 32,91 | 0,851 | 0,592 |
| LoRA r=4, 400 pas | | | | | |
| LoRA r=16, 400 pas | | | | | |
| LoRA r=32, 400 pas | | | | | |

Coût de l'entraînement rang 8 : **49 min** sur T4, **4,66 Go** de VRAM,
**1 594 368 paramètres entraînés sur 861 millions (0,19 %)**, pour un fichier
de **6,1 Mo** contre ~4 Go pour le modèle complet.

Deux chiffres de la baseline servent de référence absolue :

- **Style DINOv2 = 0,274.** Le modèle de base ne sait pas faire de zellige.
  C'est la valeur à faire monter.
- **Mémorisation = 0,737.** C'est le niveau du hasard : la ressemblance
  maximale obtenue entre une image générée et une image d'entraînement que le
  modèle n'a jamais vue. Au-delà, après entraînement, il y a suspicion de copie.

L'écart entre `style_clip_moy` (0,706) et `style_dino_moy` (0,274) sur les mêmes
images justifie l'emploi de deux mesures : CLIP sature sur « des carreaux
colorés », DINOv2 discrimine la texture et la structure fine. **DINOv2 est donc
la mesure de style principale du projet.**

Figures attendues dans `results/figures/` :

- `compromis.png` — fidélité au style vs respect du prompt, un point par configuration
- `grille_rangs.png` — même prompt, même seed, un rang par ligne
- `intensite.png` — effet du curseur d'intensité LoRA de 0 à 1,25
- `hors_domaine.png` — dégradation sur des prompts sans rapport avec le zellige

## Démarrage rapide

```bash
git clone https://github.com/HatimTj/lora-zellige-patrimoine.git
cd lora-zellige-patrimoine
pip install -r requirements.txt
```

Puis, dans l'ordre des étapes :

```bash
# Étape 1 — collecte des images et de leurs licences
python src/collect_wikimedia.py --limite 120
python src/planche_contact.py          # planche pour le tri visuel
python src/prepare_data.py

# Étape 1 — légendes
python src/legender.py

# Étape 2 — baseline du modèle non modifié, puis mesures
python src/generate.py --sortie results/baseline
python src/evaluate.py --dossier results/baseline --nom baseline

# Étape 3 et 4 — entraînement d'une LoRA, puis génération et mesure
python src/train_lora.py --config configs/lora_r8.yaml
python src/generate.py --lora runs/r8 --intensite 1.0 --sortie results/r8
python src/evaluate.py --dossier results/r8 --nom lora_r8_1000pas

# Étape 6 — démonstration interactive
python demo/app.py
```

Les notebooks de `notebooks/` reprennent les mêmes étapes, prêts à exécuter sur
Colab.

## Structure du dépôt

| Dossier | Contenu |
|---|---|
| `notebooks/` | Notebooks numérotés : données, baseline, entraînement, évaluation, démo |
| `src/` | Code réutilisable du projet |
| `configs/` | Paramètres des expériences, une config par run |
| `data/` | Script de téléchargement et fiche des données, **sans données lourdes** |
| `results/` | Figures, tableaux et exemples générés |
| `demo/` | Application Gradio et vidéo courte |
| `report/` | Rapport final |
| `docs/` | Licences, fiche du modèle, réflexion éthique, feuille de route |

Les images et les poids de modèles sont exclus de Git (`.gitignore`). Seuls
`data/licenses.csv` et `data/metadata.jsonl` sont versionnés : ils suffisent à
reconstruire le jeu de données à l'identique.

## Décisions de cadrage

Ces choix sont figés dans `configs/base.yaml`. Les modifier invaliderait les
comparaisons déjà faites.

**Un seul concept — le zellige.** Mélanger zellige, portes et calligraphie
empêcherait la LoRA de converger avec seulement 45 images.

**SD 1.5 plutôt que SDXL.** SDXL en 1024 × 1024 dépasse le budget de 12 Go de
VRAM imposé par le sujet. SD 1.5 en 512 tient dans ~8 à 10 Go.

**Prompts et seeds figés avant tout entraînement.** Choisir ses prompts après
avoir vu les résultats rendrait toute comparaison avant/après sans valeur.

**Pas de FID.** Avec ~45 images de référence, le FID est fortement biaisé et peu
fiable. On utilise CLIP-I, DINOv2 et l'inspection visuelle, et ce choix est
justifié dans le rapport.

## Avancement

- [x] **Étape 0** — Cadrage, structure du dépôt, conventions
- [x] **Étape 1** — Jeu de données : 367 candidats → 37 retenus → 29 + 8
- [x] **Étape 2** — Baseline SD 1.5 mesurée : style DINOv2 = 0,274, prompt = 34,3
- [x] **Étape 3** — LoRA rang 8, 1000 pas : style DINOv2 0,274 → 0,339 (+23,5 %)
- [~] **Étape 4** — Balayage du nombre de pas fait ; intensité à refaire (bug du
  curseur corrigé) ; balayage du rang à lancer (notebook 05)
- [ ] **Étape 5** — Analyse du surapprentissage
- [ ] **Étape 6** — Démo Gradio et vidéo
- [ ] **Étape 7** — Rapport (6-8 p) et soutenance (10 min)

Feuille de route détaillée et dépendances entre tâches : `docs/contributions.md`.

## Équipe

- **Hatim Tajimi**
- **Houssam El Moutaqui**
- **Saad Belekbira**

La répartition des tâches sera présentée à la soutenance. La feuille de route
et les dépendances entre tâches sont dans `docs/contributions.md`.

## Licence et éthique

Le code est sous licence MIT (`LICENSE`). Les images sources restent sous leurs
licences respectives, documentées image par image dans `data/licenses.csv`
(auteur, URL, licence).

Le zellige est un savoir-faire artisanal vivant. Les images produites sont des
illustrations synthétiques, explicitement marquées comme telles dans la démo, et
ne prétendent ni reproduire un ouvrage réel ni se substituer au travail d'un
maalem. Voir `docs/ethics.md`.

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
- `pas.png` — effet du nombre de pas d'entraînement
- `intensite.png` — effet du curseur d'intensité LoRA de 0 à 1,25
- `planche_intensite.png` — mêmes prompt et graine, une intensité par colonne
- `rangs.png` et `planche_rangs.png` — effet du rang, à budget identique
- `memorisation_seuil.png` — seuil de recopie établi sur le holdout (Étape 5)
- `plus_proches_voisins.png` — images générées et leur plus proche voisin d'entraînement
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
| `notebooks/` | Notebooks numérotés : données, baseline, entraînement, balayages, surapprentissage, démo |
| `demo/` | Interface Gradio, avec audit de recopie en direct ; voir `demo/README.md` |
| `report/` | Rapport et soutenance : sources HTML, PDF, et `construire_pdf.sh` |
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

## Résultats de l'Étape 4

### Nombre de pas d'entraînement (rang 8, intensité 1,0)

| pas | style DINOv2 | mémorisation max | diversité |
|---|---|---|---|
| 250 | 0,285 | 0,787 | 0,599 |
| 500 | 0,328 | 0,810 | 0,588 |
| 750 | 0,334 | 0,851 | 0,592 |
| 1000 | 0,339 | 0,845 | 0,597 |

Le style gagne +15 % entre 250 et 500 pas, puis +3 % sur les 500 suivants. La
mémorisation, elle, continue de monter. **500 pas suffisent** : au-delà on paie
du temps GPU et du risque de recopie pour un gain marginal.

### Intensité du curseur (rang 8, 1000 pas)

| intensité | 0 (base) | 0,25 | 0,50 | 0,75 | 1,00 | 1,25 |
|---|---|---|---|---|---|---|
| style DINOv2 | 0,274 | 0,275 | 0,294 | 0,274 | **0,339** | 0,346 |
| diversité | 0,646 | 0,633 | 0,616 | 0,611 | 0,597 | 0,572 |

La réponse n'est pas linéaire, c'est **un effet de seuil autour de 1,0**. En
dessous, la LoRA agit déjà — la diversité décroît de façon strictement monotone
— mais elle modifie la palette et la texture sans imposer la géométrie. Sur
`planche_intensite.png` on voit un grand carreau peint jusqu'à 0,75, puis à 1,00
le champ se fragmente en petites tesselles assemblées en étoiles, qui est la
signature du zellige. L'image et la mesure basculent au même endroit.

Attention à la lecture de la planche : le prompt D1 demande du zellige, donc le
modèle de base produit déjà du carrelage bleu et blanc. Une colonne « qui
ressemble à du zellige » ne prouve donc rien à elle seule ; c'est le passage du
carreau peint à la mosaïque assemblée qui distingue la LoRA.

### Rang de la LoRA (400 pas, accumulation 2, intensité 1,0)

Coût d'abord :

| rang | paramètres | % du U-Net | durée | VRAM | fichier | perte finale |
|---|---|---|---|---|---|---|
| 4 | 797 k | 0,09 % | 9,5 min | 4,65 Go | 3,1 Mo | 0,1345 |
| 8 | 1,59 M | 0,19 % | 9,6 min | 4,66 Go | 6,1 Mo | 0,1334 |
| 16 | 3,19 M | 0,37 % | 9,7 min | 4,67 Go | 12,2 Mo | 0,1356 |
| 32 | 6,38 M | 0,74 % | 9,6 min | 4,70 Go | 24,4 Mo | 0,1361 |

**Le rang ne coûte presque rien.** Multiplier les paramètres par huit ne change
ni la durée (9,5 à 9,7 min) ni la mémoire (+0,05 Go). Le calcul est dominé par
le U-Net gelé, que l'on traverse de toute façon ; les matrices LoRA sont du
bruit à côté. Seul le fichier grossit, proportionnellement. La perte finale est
plate (0,133 à 0,136) : aucun rang ne colle mieux aux données.

Qualité ensuite :

| rang | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| style DINOv2 | 0,274 | 0,277 | 0,278 | **0,303** |
| mémorisation | 0,723 | 0,744 | 0,742 | 0,748 |
| diversité | 0,617 | 0,614 | 0,612 | 0,619 |

**Limite à bien lire.** Le budget de ce balayage (400 pas × accumulation 2 =
800 passages) est plus court que le plus petit point du balayage des pas
(250 pas × accumulation 4 = 1000 passages, style 0,285). Les quatre rangs sont
donc mesurés *avant* le seuil où le style apparaît : trois d'entre eux sont au
niveau de la baseline parce qu'aucun n'a eu le temps d'apprendre, pas parce que
le rang serait sans effet.

Ce qu'on peut affirmer : **à budget égal et court, le rang 32 décolle le
premier.** Un rang plus grand converge plus vite par pas d'optimisation. Ce
qu'on ne peut pas affirmer : quel rang gagne à convergence — il faudrait
relancer les quatre à 1000 pas, soit 40 minutes de plus.

Le modèle retenu reste le **rang 8 à 1000 pas** (style 0,339, 6 Mo), au-dessus
de tout ce qui figure ici.

### Vérification du curseur

La section 5 du notebook 04 génère une image à l'intensité 0,0 et la compare
pixel par pixel au modèle de base. Si l'écart dépasse 3/255, le notebook
s'arrête : le réglage n'aurait pas été appliqué et tout le balayage serait
trompeur. Ce contrôle a été ajouté après une fausse alerte où la forme en
palier de la courbe avait été prise pour une panne du réglage. Le balayage a
ensuite été relancé par une autre voie de l'API diffusers (`set_adapters` au
lieu de `cross_attention_kwargs`) et a redonné les mêmes valeurs au centième,
ce qui confirme les deux mesures.

## Résultats de l'Étape 5 — surapprentissage

### Le seuil de recopie n'est pas choisi, il est mesuré

Décréter « au-dessus de 0,85 c'est une copie » n'est pas défendable. Le seuil
vient donc des données : les 8 images du holdout sont de vraies photos de
zellige que le modèle n'a jamais vues, et leur ressemblance à l'entraînement
donne la ressemblance normale entre deux photos indépendantes du même
artisanat.

| similarité DINOv2 au plus proche voisin d'entraînement | min | médiane | moyenne | max |
|---|---|---|---|---|
| holdout (8 vraies photos jamais vues) | 0,457 | 0,733 | 0,722 | **0,908** |
| images générées (28 prompts zellige) | 0,123 | 0,601 | 0,491 | 0,845 |

**Aucune image générée n'atteint le seuil de 0,908**, et la distribution générée
est globalement *plus basse* que celle du holdout : une image produite ressemble
moins à l'entraînement que ne le fait une vraie photo de zellige inconnue du
modèle.

Lecture plus sévère, puisque le maximum sur 8 échantillons est une barre
permissive : en prenant la moyenne du holdout (0,722), **3 images sur 28
dépassent**, soit 10,7 %. Ces trois-là sont les premières de
`plus_proches_voisins.png`, et c'est là que le regard prend le relais.

### Ce que l'œil ajoute au chiffre

`plus_proches_voisins.png` affiche les 8 paires les plus ressemblantes. Les
trois premières (0,845, 0,830, 0,782) partagent avec leur voisin d'entraînement
la **composition** — une rosace centrale, un éclatement en étoile — mais pas la
palette, pas le nombre de branches, pas le détail des tesselles. Une recopie se
reconnaîtrait au contraire aux défauts du carreau et à la découpe exacte.

C'est attendu : la géométrie islamique repose sur un répertoire fini de
symétries. Deux zelliges authentiques et sans rapport partagent déjà beaucoup,
ce que confirme le holdout. Le score élevé mesure ce répertoire commun, pas une
mémorisation.

### Le modèle sait-il encore faire autre chose ?

| | modèle de base | LoRA | écart |
|---|---|---|---|
| CLIPScore hors domaine | 33,14 | 32,94 | **−0,20** |

Rien n'est perdu. `hors_domaine.png` montre un chien, des fraises et une rue
sous la pluie tout à fait normaux, sans trace de carrelage. Le pire écart
(−2,45, sur les fraises) correspond à un recadrage, pas à une dégradation du
contenu. La LoRA reste accrochée au token `zlj_style` au lieu de déborder sur
tous les prompts.

### Conclusion de l'étape

Le modèle a appris un style, pas un jeu d'images. Les trois mesures
indépendantes — seuil issu du holdout, inspection des paires, prompts hors
domaine — vont dans le même sens, et la plus exigeante des trois (10,7 % au-
dessus de la moyenne du holdout) reste explicable par le répertoire géométrique
commun.

Limite à mentionner en soutenance : avec 8 images de holdout, le seuil repose
sur peu d'échantillons. C'est le même ordre de grandeur de données qui interdit
le FID, et cela reste une contrainte assumée du projet.

## Avancement

- [x] **Étape 0** — Cadrage, structure du dépôt, conventions
- [x] **Étape 1** — Jeu de données : 367 candidats → 37 retenus → 29 + 8
- [x] **Étape 2** — Baseline SD 1.5 mesurée : style DINOv2 = 0,274, prompt = 34,3
- [x] **Étape 3** — LoRA rang 8, 1000 pas : style DINOv2 0,274 → 0,339 (+23,5 %)
- [x] **Étape 4** — Balayages du nombre de pas, de l'intensité et du rang
- [x] **Étape 5** — Surapprentissage : 0/28 image au-dessus du seuil de
  recopie, CLIPScore hors domaine −0,20
- [~] **Étape 6** — Démo Gradio (`demo/app.py`, notebook 07) ; vidéo à
  enregistrer et à déposer hors dépôt
- [x] **Étape 7** — Rapport (`report/rapport.pdf`, 8 p + couverture),
  soutenance (`report/soutenance.pdf`, 16 diapositives), fiche du modèle et
  réflexion éthique

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

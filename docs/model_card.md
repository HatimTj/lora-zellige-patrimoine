# Fiche du modèle — LoRA zellige

## Identité

| | |
|---|---|
| **Nom** | `lora-zellige-r8` |
| **Type** | adaptateur LoRA pour Stable Diffusion 1.5 |
| **Modèle de base** | `runwayml/stable-diffusion-v1-5` |
| **Taille de l'adaptateur** | 6,1 Mo — 1 594 368 paramètres, 0,19 % du U-Net |
| **Rang / alpha** | 8 / 8 |
| **Modules adaptés** | les quatre projections de l'attention (`to_q`, `to_k`, `to_v`, `to_out.0`) |
| **Token déclencheur** | `zlj_style` |
| **Résolution** | 512 × 512 |
| **Licence** | même régime que les données sources (voir `data/licenses.csv`) |

## Données d'entraînement

29 photographies de zellige marocain issues de Wikimedia Commons, sélectionnées
parmi 367 candidats, recadrées en 512 × 512 sans déformation. 8 images
supplémentaires sont réservées (*holdout*) et n'ont jamais été vues par le
modèle : elles servent à établir le seuil de recopie.

Licences acceptées : CC0, domaine public, CC BY, CC BY-SA. Écartées : NC et ND,
puisqu'entraîner un modèle constitue une modification. Chaque image est tracée
— auteur, licence, URL — dans `data/licenses.csv`, qui conserve aussi les 330
rejets avec leur motif.

Les légendes ne décrivent que ce qui varie (couleur, composition), jamais le
style lui-même : c'est ce qui permet au style de se concentrer sur le token
déclencheur plutôt que de se répartir sur des mots ordinaires.

## Procédure d'entraînement

| | |
|---|---|
| Pas d'optimisation | 1000 (accumulation 4 → 4000 passages) |
| Taux d'apprentissage | 1 × 10⁻⁴, échauffement 50 pas |
| Optimiseur | AdamW 8 bits (bitsandbytes) |
| Précision | fp16 avec autocast, poids LoRA en fp32 |
| Mémoire au pic | 4,66 Go |
| Durée | 49 minutes sur Google Colab T4 |
| Perte finale | 0,1156 |

## Usage prévu

Génération d'illustrations inspirées des motifs géométriques du zellige, dans un
cadre pédagogique et de valorisation du patrimoine. Le style s'appelle en
écrivant `zlj_style` dans le prompt ; sans ce mot, le modèle se comporte
normalement.

## Usages hors périmètre

- Faire passer une image produite pour une photographie d'un ouvrage réel.
- Reproduire le travail identifiable d'un artisan ou d'un artiste vivant.
- Tout usage commercial sans vérification préalable des licences sources.
- Documenter le zellige : 29 images n'en représentent qu'une facette.

## Métriques

Moyennes sur 40 images — 10 prompts figés × 4 graines figées, établis avant
tout entraînement. Mesures complètes dans `results/results.csv`.

| Mesure | Modèle d'origine | LoRA | Écart |
|---|---|---|---|
| Style — DINOv2 | 0,2741 | **0,3386** | **+23,5 %** |
| Style — CLIP-I | 0,7061 | 0,7310 | +3,5 % |
| Prompt — domaine (CLIPScore) | 34,27 | 34,40 | +0,4 % |
| Prompt — transfert | 30,78 | 30,28 | −1,6 % |
| Prompt — hors domaine | 33,14 | 32,94 | −0,6 % |
| Diversité — LPIPS | 0,6464 | 0,5968 | −7,6 % |
| Mémorisation maximale | 0,7373 | 0,8446 | +14,6 % |

**DINOv2 est la mesure de style principale.** CLIP-I sature sur ce type de
transfert : entraîné à relier images et mots, il perçoit surtout le contenu,
là où DINOv2, qui n'a jamais vu de texte, perçoit la texture et la géométrie.

## Réglages recommandés

| | |
|---|---|
| Intensité | **1,0** — la réponse est un effet de seuil, pas une rampe : en dessous de 1,0 le style ne bascule pas |
| Pas d'entraînement | **500 suffisent** si l'on réentraîne : +15 % de style entre 250 et 500, puis +3 % seulement |
| Rang | 8 est un bon compromis ; le rang ne coûte ni temps ni mémoire, seulement de la place disque |
| Échantillonnage | DPM-Solver++, 30 étapes, guidage 7,5 |

## Limites connues

- **Mémorisation** : aucune des 28 images d'évaluation n'atteint le seuil de
  recopie de 0,908, mesuré sur le holdout. En retenant la moyenne du holdout
  (0,722) comme barre plus sévère, 3 images sur 28 la dépassent ; l'inspection
  visuelle montre qu'elles partagent une composition, non une copie.
- **Dégradation hors domaine** : −0,20 de CLIPScore, soit l'équivalent du bruit.
- **Diversité** : −7,6 %. Un modèle spécialisé se répète davantage.
- **Couverture** : dominante bleu et blanc, panneaux plutôt que fontaines ou
  sols. Les prompts demandant d'autres palettes sont moins bien servis.
- **Seuil fondé sur 8 images** : c'est peu. La même rareté de données interdit
  le FID, biaisé en dessous de quelques milliers d'images de référence.

## Reproduire

```bash
git clone https://github.com/HatimTj/lora-zellige-patrimoine.git
```

Notebooks 01 à 07 : reconstruction du jeu de données, baseline, entraînement,
balayages, surapprentissage, démonstration. Les images et les poids ne sont pas
versionnés ; le dépôt contient de quoi les régénérer à l'identique.

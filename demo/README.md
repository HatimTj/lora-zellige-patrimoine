# Démo interactive

Interface web de la LoRA zellige. Elle montre le modèle en marche et, surtout,
dit à chaque génération si l'image produite recopie le jeu d'entraînement.

```bash
python demo/app.py                 # local,  http://127.0.0.1:7860
python demo/app.py --partager      # lien public, pour Colab
python demo/app.py --sans-modele   # interface seule, sans GPU
```

Notebook de lancement : `notebooks/07_demo.ipynb`.

## Pré-requis

| | pourquoi |
|---|---|
| `runs/r8/pytorch_lora_weights.safetensors` | les poids entraînés, 6,1 Mo |
| `data/processed/train/` | les 29 images que l'audit interroge |
| `results/surapprentissage.json` | le seuil de recopie ; sinon repli sur 0,908 |

Les deux derniers se reconstruisent avec `src/collect_wikimedia.py --reconstruire`
puis `src/prepare_data.py`.

## Trois choix de conception

**Un seul pipeline pour les deux images.** L'Étape 4 a établi qu'une intensité
de 0 reproduit le modèle de base au pixel près — c'est devenu un test
automatique du notebook 04. Le comparatif « avant / après » ne charge donc pas
deux modèles, il déplace un curseur. Économie : 1,6 Go de VRAM et une minute de
chargement.

**L'image de base est mise en cache.** À prompt et graine identiques elle ne
dépend pas de l'intensité. Jouer avec le curseur ne recalcule que l'image de
droite, et la réponse arrive deux fois plus vite — ce qui rend la démonstration
du curseur fluide à l'écran.

**La démo s'audite elle-même.** Chaque image est comparée aux 29 images
d'entraînement par DINOv2, et le score est placé sur l'échelle de l'Étape 5 :
médiane du holdout et seuil de recopie. L'image d'entraînement la plus proche
s'affiche à côté, parce qu'un chiffre seul ne distingue pas une copie d'un motif
géométrique commun.

## Compatibilité Gradio

Gradio 6 a déplacé `theme` et `css` du constructeur `Blocks` vers `launch()`, et
retiré `show_download_button` de `gr.Image`. Le code détecte la version majeure
et s'adapte. Vérifié sur **Gradio 5.50** et **6.32**.

"""
Etape 2 - Les mesures.

Quatre questions, quatre familles de mesures :

  Q1  a-t-il appris le style ?   CLIP-I et DINOv2 entre images generees et
                                 images de reference
  Q2  obeit-il au texte ?        CLIPScore entre image generee et son prompt
  Q3  combien ca coute ?         mesure dans src/generate.py et src/train_lora.py
  Q4  a-t-il surappris ?         similarite au plus proche voisin du jeu
                                 d'entrainement, et diversite LPIPS

Pourquoi deux mesures de style plutot qu'une ?
  - CLIP-I voit surtout le CONTENU semantique ("des carreaux colores")
  - DINOv2 est entraine sans etiquettes et reagit davantage a la TEXTURE et a
    la structure fine, donc au style proprement dit.
Quand les deux sont d'accord, la conclusion est solide. Quand elles divergent,
c'est en soi un resultat a commenter dans le rapport.

Pourquoi pas de FID : il faut des milliers d'images de reference pour qu'il
soit fiable. Avec 29, il mesurerait surtout le bruit d'echantillonnage.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import info  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MODELE_CLIP = "openai/clip-vit-base-patch32"
MODELE_DINO = "facebook/dinov2-base"


def appareil_torch() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def lire_images(dossier: Path, motif: str = "*") -> List[Tuple[str, Image.Image]]:
    """Charge les images d'un dossier, triees par nom."""
    fichiers = sorted(p for p in dossier.glob(motif)
                      if p.suffix.lower() in EXTENSIONS)
    return [(p.name, Image.open(p).convert("RGB")) for p in fichiers]


# --------------------------------------------------------------------------
# CLIP
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _clip():
    from transformers import CLIPModel, CLIPProcessor
    info(f"chargement de {MODELE_CLIP}")
    modele = CLIPModel.from_pretrained(MODELE_CLIP).to(appareil_torch()).eval()
    processeur = CLIPProcessor.from_pretrained(MODELE_CLIP)
    return modele, processeur


@torch.no_grad()
def vecteurs_clip_image(images: Sequence[Image.Image]) -> torch.Tensor:
    modele, processeur = _clip()
    entrees = processeur(images=list(images), return_tensors="pt").to(appareil_torch())
    v = modele.get_image_features(**entrees)
    return torch.nn.functional.normalize(v, dim=-1)


@torch.no_grad()
def vecteurs_clip_texte(textes: Sequence[str]) -> torch.Tensor:
    modele, processeur = _clip()
    entrees = processeur(text=list(textes), return_tensors="pt",
                         padding=True, truncation=True).to(appareil_torch())
    v = modele.get_text_features(**entrees)
    return torch.nn.functional.normalize(v, dim=-1)


def clipscore(images: Sequence[Image.Image], textes: Sequence[str]) -> List[float]:
    """Respect du prompt : cosinus image-texte, x100, par paire.

    Echelle utile : ~20 = sans rapport, ~30 = correct, ~35+ = tres bon.
    """
    vi = vecteurs_clip_image(images)
    vt = vecteurs_clip_texte(textes)
    return [round(float(x) * 100, 2) for x in (vi * vt).sum(dim=-1)]


# --------------------------------------------------------------------------
# DINOv2
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _dino():
    from transformers import AutoImageProcessor, AutoModel
    info(f"chargement de {MODELE_DINO}")
    modele = AutoModel.from_pretrained(MODELE_DINO).to(appareil_torch()).eval()
    processeur = AutoImageProcessor.from_pretrained(MODELE_DINO)
    return modele, processeur


@torch.no_grad()
def vecteurs_dino(images: Sequence[Image.Image]) -> torch.Tensor:
    modele, processeur = _dino()
    entrees = processeur(images=list(images), return_tensors="pt").to(appareil_torch())
    sortie = modele(**entrees).last_hidden_state[:, 0]      # jeton CLS
    return torch.nn.functional.normalize(sortie, dim=-1)


# --------------------------------------------------------------------------
# Similarite a un jeu de reference
# --------------------------------------------------------------------------

def similarite_au_jeu(vecteurs_generes: torch.Tensor,
                      vecteurs_reference: torch.Tensor) -> Dict[str, List[float]]:
    """Pour chaque image generee : similarite MOYENNE et MAXIMALE au jeu.

    La moyenne repond a Q1 (le style global est-il appris ?).
    Le maximum repond a Q4 : s'il est tres eleve, l'image generee est la copie
    d'une image d'entrainement precise.
    """
    sim = vecteurs_generes @ vecteurs_reference.T        # cosinus, tout contre tout
    return {
        "moyenne": [round(float(x), 4) for x in sim.mean(dim=1)],
        "maximum": [round(float(x), 4) for x in sim.max(dim=1).values],
        "indice_plus_proche": [int(i) for i in sim.argmax(dim=1)],
    }


# --------------------------------------------------------------------------
# Diversite
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _lpips():
    import lpips
    info("chargement de LPIPS (AlexNet)")
    return lpips.LPIPS(net="alex").to(appareil_torch()).eval()


def _en_tenseur(image: Image.Image, taille: int = 256) -> torch.Tensor:
    import numpy as np
    arr = np.asarray(image.resize((taille, taille), Image.LANCZOS), dtype="float32")
    t = torch.from_numpy(arr).permute(2, 0, 1) / 127.5 - 1.0      # vers [-1, 1]
    return t.unsqueeze(0).to(appareil_torch())


@torch.no_grad()
def diversite_lpips(images: Sequence[Image.Image]) -> float:
    """Distance perceptuelle moyenne entre toutes les paires d'un groupe.

    Les images d'un groupe partagent le meme prompt et ne different que par la
    graine. Si la valeur s'effondre, le modele produit toujours la meme image :
    c'est l'effondrement de modes, symptome classique de surapprentissage.
    """
    if len(images) < 2:
        return float("nan")
    modele = _lpips()
    tenseurs = [_en_tenseur(im) for im in images]
    distances = []
    for i in range(len(tenseurs)):
        for j in range(i + 1, len(tenseurs)):
            distances.append(float(modele(tenseurs[i], tenseurs[j]).item()))
    return round(sum(distances) / len(distances), 4)


if __name__ == "__main__":
    # Verification legere : les modeles se chargent-ils, et les mesures
    # se comportent-elles comme attendu sur des cas evidents ?
    from PIL import ImageDraw

    rouge = Image.new("RGB", (224, 224), (200, 30, 30))
    bleu = Image.new("RGB", (224, 224), (30, 60, 200))
    damier = Image.new("RGB", (224, 224), "white")
    d = ImageDraw.Draw(damier)
    for x in range(0, 224, 28):
        for y in range(0, 224, 28):
            if (x // 28 + y // 28) % 2 == 0:
                d.rectangle([x, y, x + 28, y + 28], fill="black")

    print("CLIPScore (doit etre eleve sur la diagonale) :")
    for texte in ["a solid red image", "a solid blue image",
                  "a black and white checkerboard"]:
        scores = clipscore([rouge, bleu, damier], [texte] * 3)
        print(f"  {texte:<32} {scores}")

    print("\nDiversite LPIPS :")
    print("  trois images differentes :", diversite_lpips([rouge, bleu, damier]))
    print("  trois fois la meme       :", diversite_lpips([rouge, rouge, rouge]))

"""
Etape 1.4 - Preparation des images.

Trois operations, dans cet ordre :
  1. recadrage carre au centre puis reduction en 512 x 512
  2. suppression des quasi-doublons (hachage perceptuel)
  3. separation entrainement / test de generalisation

Le recadrage ne DEFORME jamais l'image : une etoile a huit branches doit rester
une etoile a huit branches, sinon le modele apprend des motifs aplatis.

Usage :
    python src/prepare_data.py
    python src/prepare_data.py --seuil-doublon 12    # plus strict
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (assurer_dossier, charger_config, chemin_projet,  # noqa: E402
                    ecrire_json, info)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# --------------------------------------------------------------------------
# Recadrage
# --------------------------------------------------------------------------

def carre_centre(image: Image.Image, taille: int) -> Image.Image:
    """Decoupe le plus grand carre centre, puis reduit a `taille`.

    On decoupe au lieu d'etirer : etirer une image 4000x3000 vers un carre
    ecraserait les motifs horizontalement.
    """
    cote = min(image.size)
    gauche = (image.width - cote) // 2
    haut = (image.height - cote) // 2
    image = image.crop((gauche, haut, gauche + cote, haut + cote))
    return image.resize((taille, taille), Image.LANCZOS)


# --------------------------------------------------------------------------
# Hachage perceptuel (dHash)
# --------------------------------------------------------------------------

def empreinte_visuelle(image: Image.Image) -> int:
    """Empreinte dHash sur 64 bits.

    On reduit l'image a 9x8 pixels en niveaux de gris, puis on compare chaque
    pixel a son voisin de droite : 1 s'il est plus clair, 0 sinon. Deux photos
    de la meme scene donnent des empreintes presque identiques, meme si les
    fichiers sont differents.
    """
    petite = image.convert("L").resize((9, 8), Image.LANCZOS)
    pixels = list(petite.tobytes())   # mode 'L' : un octet par pixel, ligne par ligne
    bits = 0
    for ligne in range(8):
        for colonne in range(8):
            gauche = pixels[ligne * 9 + colonne]
            droite = pixels[ligne * 9 + colonne + 1]
            bits = (bits << 1) | int(gauche > droite)
    return bits


def distance(a: int, b: int) -> int:
    """Nombre de bits differents entre deux empreintes (distance de Hamming)."""
    return bin(a ^ b).count("1")


def ecarter_doublons(empreintes: Dict[str, int], seuil: int
                     ) -> Tuple[List[str], List[Tuple[str, str, int]]]:
    """Garde un representant par groupe d'images trop semblables."""
    rejetes: set[str] = set()
    paires: List[Tuple[str, str, int]] = []
    for a, b in combinations(sorted(empreintes), 2):
        if a in rejetes or b in rejetes:
            continue
        d = distance(empreintes[a], empreintes[b])
        if d <= seuil:
            paires.append((a, b, d))
            rejetes.add(b)          # on garde le premier dans l'ordre des noms
    gardes = [n for n in sorted(empreintes) if n not in rejetes]
    return gardes, paires


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", default="configs/base.yaml")
    parseur.add_argument("--seuil-doublon", type=int, default=10,
                         help="distance max entre empreintes pour etre un doublon")
    args = parseur.parse_args()

    cfg = charger_config(args.config)
    d = cfg["donnees"]
    taille = d["taille_image"]
    n_holdout = d["n_images_holdout"]
    graine = cfg["reproductibilite"]["seeds_evaluation"][0]

    source = chemin_projet("data/raw")
    fichier_selection = chemin_projet("data/selection.txt")

    # 1. Quelles images traiter ?
    if fichier_selection.exists():
        noms = {ligne.split("#")[0].strip()
                for ligne in fichier_selection.read_text(encoding="utf-8").splitlines()}
        noms = {n for n in noms if n}
        fichiers = [source / n for n in sorted(noms) if (source / n).exists()]
        info(f"selection manuelle : {len(fichiers)} images")
    else:
        fichiers = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS)
        info(f"pas de selection manuelle, toutes les images : {len(fichiers)}")

    if not fichiers:
        sys.exit("ERREUR : aucune image dans data/raw. Lancez d'abord "
                 "`python src/collect_wikimedia.py --reconstruire`.")

    attendu = len(noms) if fichier_selection.exists() else len(fichiers)
    if len(fichiers) < attendu:
        info(f"ATTENTION : {attendu - len(fichiers)} image(s) de la selection "
             "manquent dans data/raw")

    # 2. Recadrer en memoire et calculer les empreintes
    recadrees: Dict[str, Image.Image] = {}
    empreintes: Dict[str, int] = {}
    for chemin in fichiers:
        try:
            with Image.open(chemin) as img:
                img = img.convert("RGB")
                carre = carre_centre(img, taille)
        except Exception as e:
            info(f"illisible, ignoree : {chemin.name} ({e})")
            continue
        recadrees[chemin.name] = carre
        empreintes[chemin.name] = empreinte_visuelle(carre)

    # 3. Ecarter les quasi-doublons
    gardes, paires = ecarter_doublons(empreintes, args.seuil_doublon)
    if paires:
        info(f"{len(paires)} paire(s) trop semblable(s), seuil = {args.seuil_doublon} :")
        for a, b, dist in paires:
            info(f"   {a}  ~  {b}   (distance {dist}) -> {b} ecartee")
    else:
        info("aucun quasi-doublon detecte")
    plus_proches = sorted(
        ((distance(empreintes[a], empreintes[b]), a, b)
         for a, b in combinations(sorted(empreintes), 2)))[:5]
    info("paires les plus ressemblantes (verification du detecteur) :")
    for dist, a, b in plus_proches:
        info(f"   distance {dist:2d}   {a}  ~  {b}")

    # 4. Separer entrainement et test de generalisation
    melange = list(gardes)
    random.Random(graine).shuffle(melange)
    holdout = sorted(melange[:n_holdout])
    entrainement = sorted(melange[n_holdout:])

    dossier_train = assurer_dossier("data/processed/train")
    dossier_holdout = assurer_dossier("data/processed/holdout")
    for dossier in (dossier_train, dossier_holdout):
        for ancien in dossier.glob("*"):
            if ancien.is_file():
                ancien.unlink()

    for nom in entrainement:
        recadrees[nom].save(dossier_train / nom, quality=95)
    for nom in holdout:
        recadrees[nom].save(dossier_holdout / nom, quality=95)

    # 5. Rapport, pour le rapport ecrit et la reproductibilite
    rapport = {
        "images_en_entree": len(fichiers),
        "illisibles": len(fichiers) - len(recadrees),
        "quasi_doublons_ecartes": [
            {"gardee": a, "ecartee": b, "distance": dist} for a, b, dist in paires
        ],
        "taille": taille,
        "graine_separation": graine,
        "entrainement": entrainement,
        "holdout": holdout,
        "n_entrainement": len(entrainement),
        "n_holdout": len(holdout),
    }
    chemin_rapport = ecrire_json(rapport, "data/preparation.json")

    info(f"{len(entrainement)} images d'entrainement  -> {dossier_train}")
    info(f"{len(holdout)} images de generalisation -> {dossier_holdout}")
    info(f"rapport : {chemin_rapport}")


if __name__ == "__main__":
    principal()

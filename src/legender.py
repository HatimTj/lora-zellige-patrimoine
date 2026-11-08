"""
Etape 1.5 - Legendes des images d'entrainement.

Produit un brouillon de legende par image, a relire a la main.

Principe directeur : on decrit CE QUI VARIE d'une image a l'autre (couleurs,
forme dominante, support), jamais le style lui-meme. Le style est la seule
chose commune a toutes les images : si on le decrit avec des mots ordinaires
("Moroccan", "mosaic", "geometric"), il se repartit entre ces mots. En ne le
decrivant pas, on le force a se concentrer dans le token declencheur.

Le brouillon combine deux sources :
  - les couleurs dominantes, extraites de l'image elle-meme
  - le type d'objet, devine depuis le titre Wikimedia d'origine

Usage :
    python src/legender.py              # ecrit le brouillon
    python src/legender.py --afficher   # relit et affiche les legendes
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import charger_config, chemin_projet, info  # noqa: E402


# --------------------------------------------------------------------------
# Couleurs
# --------------------------------------------------------------------------

# Couleurs de reference et leur nom anglais (le modele est entraine en anglais)
REFERENCES: List[Tuple[str, Tuple[int, int, int]]] = [
    ("white",     (245, 245, 240)),
    ("black",     (25, 25, 25)),
    ("grey",      (140, 140, 140)),
    ("beige",     (214, 196, 160)),
    ("ochre",     (190, 145, 70)),
    ("brown",     (120, 80, 50)),
    ("terracotta", (180, 95, 65)),
    ("red",       (180, 40, 40)),
    ("orange",    (225, 135, 45)),
    ("yellow",    (225, 200, 70)),
    ("green",     (70, 130, 70)),
    ("turquoise", (60, 165, 160)),
    ("blue",      (50, 90, 170)),
    ("navy",      (30, 45, 95)),
]


def nom_couleur(rvb: Tuple[int, int, int]) -> str:
    """Nom de la couleur de reference la plus proche (distance euclidienne)."""
    return min(REFERENCES,
               key=lambda ref: sum((a - b) ** 2 for a, b in zip(rvb, ref[1])))[0]


def saturation(rvb: Tuple[int, int, int]) -> float:
    """0 pour un gris, 1 pour une couleur pure."""
    haut, bas = max(rvb), min(rvb)
    return 0.0 if haut == 0 else (haut - bas) / haut


def couleurs_dominantes(image: Image.Image, combien: int = 3) -> List[str]:
    """Les couleurs principales de l'image, nommees, sans repetition.

    Les zelliges sont faits de tesselles colorees separees par du platre clair.
    Une simple quantification renvoie surtout les tons moyens issus du melange :
    on obtient "grey, beige, brown" pour un panneau rouge et bleu. On classe
    donc les couleurs SATUREES d'abord, et on ne complete avec les neutres que
    s'il n'y en a pas assez.
    """
    reduite = image.convert("RGB").resize((128, 128), Image.LANCZOS)
    palette = reduite.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
    table = palette.getpalette()
    comptes = Counter(palette.tobytes())
    total = sum(comptes.values())

    colorees: List[Tuple[float, str]] = []
    neutres: List[Tuple[float, str]] = []
    for index, compte in comptes.most_common():
        rvb = tuple(table[index * 3: index * 3 + 3])
        part = compte / total
        if part < 0.03:                      # couleur anecdotique
            continue
        nom = nom_couleur(rvb)               # type: ignore[arg-type]
        sat = saturation(rvb)                # type: ignore[arg-type]
        # on pondere la part par la saturation : une couleur vive compte double
        (colorees if sat >= 0.30 else neutres).append((part * (1 + sat), nom))

    noms: List[str] = []
    for _, nom in sorted(colorees, reverse=True) + sorted(neutres, reverse=True):
        if nom not in noms:
            noms.append(nom)
        if len(noms) == combien:
            break
    return noms


def formuler_couleurs(noms: List[str]) -> str:
    if len(noms) == 1:
        return noms[0]
    return ", ".join(noms[:-1]) + " and " + noms[-1]


# --------------------------------------------------------------------------
# Type d'objet, devine depuis le titre Wikimedia
# --------------------------------------------------------------------------

INDICES = [
    (r"fountain|fontaine",        "tiled fountain"),
    (r"ceiling|plafond|dome",     "tiled ceiling"),
    (r"floor|sol\b|pavement",     "tiled floor"),
    (r"column|colonne|pillar",    "tiled column"),
    (r"table|tabletop",           "tiled tabletop"),
    (r"door|porte|gate",          "tiled doorway"),
    (r"niche|mihrab|arch",        "tiled niche"),
    (r"panel|panneau",            "tile panel"),
    (r"wall|mur\b|facade",        "tiled wall"),
]

COMPOSITIONS = [
    (r"rosette|rosace",                    "large central rosette"),
    (r"star|étoile|etoile",                "eight-pointed star pattern"),
    (r"medallion|médaillon",               "central medallion"),
    (r"border|frieze|frise|band|bande",    "horizontal banded border"),
    (r"calligraph|inscription|verset",     "carved inscription band above"),
    (r"check|damier",                      "checkerboard layout"),
]


def deviner(titre: str, regles: List[Tuple[str, str]], defaut: str) -> str:
    for motif, valeur in regles:
        if re.search(motif, titre, re.I):
            return valeur
    return defaut


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def titres_originaux() -> Dict[str, str]:
    """Associe chaque fichier a son titre Wikimedia, depuis licenses.csv."""
    chemin = chemin_projet("data/licenses.csv")
    if not chemin.exists():
        return {}
    with open(chemin, encoding="utf-8") as f:
        return {l["fichier"]: l["titre_original"]
                for l in csv.DictReader(f) if l["fichier"]}


def corrections_manuelles() -> Dict[str, Tuple[str, str]]:
    """Lit data/compositions.csv : objet et composition decrits a la main.

    Le titre Wikimedia ne dit presque jamais comment le motif est compose.
    Ces valeurs, ecrites en regardant les images, priment sur les devinettes.
    """
    chemin = chemin_projet("data/compositions.csv")
    if not chemin.exists():
        return {}
    lignes = [l for l in chemin.read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.lstrip().startswith("#")]
    return {l["fichier"]: (l["objet"], l["composition"])
            for l in csv.DictReader(lignes)}


def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", default="configs/base.yaml")
    parseur.add_argument("--dossier", default="data/processed/train")
    parseur.add_argument("--afficher", action="store_true",
                         help="relit le fichier existant au lieu de le regenerer")
    args = parseur.parse_args()

    cfg = charger_config(args.config)
    token = cfg["projet"]["trigger_token"]
    dossier = chemin_projet(args.dossier)
    sortie = dossier / "metadata.jsonl"

    if args.afficher:
        if not sortie.exists():
            info(f"{sortie} n'existe pas encore")
            return
        for ligne in sortie.read_text(encoding="utf-8").splitlines():
            entree = json.loads(ligne)
            print(f"{entree['file_name']:<16} {entree['text']}")
        return

    titres = titres_originaux()
    manuelles = corrections_manuelles()
    if manuelles:
        info(f"{len(manuelles)} descriptions manuelles chargees")
    entrees = []
    for chemin in sorted(dossier.glob("*.jpg")):
        with Image.open(chemin) as img:
            couleurs = couleurs_dominantes(img)
        titre = titres.get(chemin.name, "")
        if chemin.name in manuelles:
            objet, composition = manuelles[chemin.name]
        else:
            objet = deviner(titre, INDICES, "tile panel")
            composition = deviner(titre, COMPOSITIONS, "repeating geometric pattern")
        texte = f"a {token} {objet}, {formuler_couleurs(couleurs)}, {composition}"
        entrees.append({"file_name": chemin.name, "text": texte})

    if not entrees:
        sys.exit("ERREUR : aucune image dans "
                 f"{dossier}. Lancez d'abord `python src/prepare_data.py`.")

    with open(sortie, "w", encoding="utf-8") as f:
        for entree in entrees:
            f.write(json.dumps(entree, ensure_ascii=False) + "\n")

    info(f"{len(entrees)} legendes brouillon -> {sortie}")
    manquantes = [e["file_name"] for e in entrees if e["file_name"] not in manuelles]
    if manquantes:
        info(f"{len(manquantes)} image(s) sans description manuelle, composition devinee : "
             + ", ".join(manquantes))


if __name__ == "__main__":
    principal()

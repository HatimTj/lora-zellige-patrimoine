"""
Etape 1.3 - Planche de contact pour le tri manuel.

Assemble toutes les images d'un dossier en une seule planche numerotee, afin de
pouvoir juger le jeu de donnees d'un coup d'oeil et noter les numeros a ecarter.

Usage :
    python src/planche_contact.py                          # data/raw
    python src/planche_contact.py --source data/processed   # apres pretraitement
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import assurer_dossier, chemin_projet, info  # noqa: E402

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def lire_selection(chemin: Path) -> set[str]:
    """Lit un fichier de selection : un nom par ligne, '#' pour les commentaires."""
    noms = set()
    for ligne in chemin.read_text(encoding="utf-8").splitlines():
        ligne = ligne.split("#")[0].strip()
        if ligne:
            noms.add(ligne)
    return noms


def planche(source: Path, destination: Path, colonnes: int = 9,
            vignette: int = 220, marge: int = 6,
            selection: Path | None = None) -> int:
    fichiers = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS)
    if selection is not None and selection.exists():
        gardes = lire_selection(selection)
        fichiers = [f for f in fichiers if f.name in gardes]
        info(f"selection appliquee : {len(fichiers)} images sur {len(gardes)} listees")
    if not fichiers:
        info(f"aucune image dans {source}")
        return 0

    lignes = (len(fichiers) + colonnes - 1) // colonnes
    bande = 20                                    # bandeau du numero sous la vignette
    largeur = colonnes * (vignette + marge) + marge
    hauteur = lignes * (vignette + bande + marge) + marge

    planche_img = Image.new("RGB", (largeur, hauteur), (24, 24, 28))
    dessin = ImageDraw.Draw(planche_img)

    for index, fichier in enumerate(fichiers):
        colonne, ligne = index % colonnes, index // colonnes
        x = marge + colonne * (vignette + marge)
        y = marge + ligne * (vignette + bande + marge)

        try:
            with Image.open(fichier) as img:
                img = img.convert("RGB")
                # Recadrage carre centre, pour comparer les cadrages entre eux
                cote = min(img.size)
                gauche = (img.width - cote) // 2
                haut = (img.height - cote) // 2
                img = img.crop((gauche, haut, gauche + cote, haut + cote))
                img = img.resize((vignette, vignette), Image.LANCZOS)
                planche_img.paste(img, (x, y))
        except Exception as e:
            dessin.rectangle([x, y, x + vignette, y + vignette], fill=(80, 20, 20))
            info(f"illisible : {fichier.name} ({e})")

        etiquette = fichier.stem.replace("zlj_", "")
        dessin.text((x + 3, y + vignette + 4), etiquette, fill=(220, 220, 220))

    destination.parent.mkdir(parents=True, exist_ok=True)
    planche_img.save(destination, quality=88)
    info(f"planche : {destination}  ({len(fichiers)} images, {largeur}x{hauteur})")
    return len(fichiers)


def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--source", default="data/raw")
    parseur.add_argument("--sortie", default=None)
    parseur.add_argument("--colonnes", type=int, default=9)
    parseur.add_argument("--selection", default=None,
                         help="fichier listant les images a garder (ex. data/selection.txt)")
    args = parseur.parse_args()

    source = chemin_projet(args.source)
    nom = Path(args.source).name
    sortie = chemin_projet(args.sortie) if args.sortie else \
        assurer_dossier("results/figures") / (
            f"planche_selection.jpg" if args.selection else f"planche_{nom}.jpg")
    planche(source, sortie, colonnes=args.colonnes,
            selection=chemin_projet(args.selection) if args.selection else None)


if __name__ == "__main__":
    principal()

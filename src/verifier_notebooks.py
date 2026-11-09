"""
Verifie la syntaxe Python de chaque cellule de code des notebooks.

Une erreur de syntaxe dans un notebook ne se voit qu'a l'execution, souvent
apres plusieurs minutes de calcul. Ce script la trouve en une seconde, avant
d'ouvrir Colab.

    python src/verifier_notebooks.py notebooks/*.ipynb

Les lignes magiques de Jupyter (`!commande`, `%magie`) ne sont pas du Python :
elles sont neutralisees avant l'analyse. Leurs continuations -- les lignes qui
suivent un `\\` final -- sont simplement supprimees, car une ligne indentee
seule ne serait pas du Python valide.
"""

from __future__ import annotations

import ast
import io
import json
import sys
from pathlib import Path


def neutraliser_magies(source: str) -> str:
    """Remplace les lignes Jupyter par du Python inerte, indentation comprise."""
    lignes, dans_continuation = [], False
    for ligne in source.splitlines():
        depouillee = ligne.strip()
        if dans_continuation:
            lignes.append("")                      # continuation : on la retire
            dans_continuation = depouillee.endswith("\\")
        elif depouillee.startswith(("!", "%")):
            indentation = ligne[:len(ligne) - len(ligne.lstrip())]
            lignes.append(f"{indentation}pass")    # garde la structure des blocs
            dans_continuation = depouillee.endswith("\\")
        else:
            lignes.append(ligne)
    return "\n".join(lignes)


def verifier(chemin: Path) -> list[tuple[int, SyntaxError, str]]:
    nb = json.load(io.open(chemin, encoding="utf-8"))
    problemes = []
    for index, cellule in enumerate(nb["cells"]):
        if cellule["cell_type"] != "code":
            continue
        source = "".join(cellule["source"])
        try:
            ast.parse(neutraliser_magies(source))
        except SyntaxError as erreur:
            problemes.append((index, erreur, source))
    return problemes


def principal() -> int:
    chemins = [Path(a) for a in sys.argv[1:]] or sorted(Path("notebooks").glob("*.ipynb"))
    total = casses = 0
    for chemin in chemins:
        nb = json.load(io.open(chemin, encoding="utf-8"))
        n = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        problemes = verifier(chemin)
        total += n
        casses += len(problemes)
        etat = "toutes valides" if not problemes else f"{len(problemes)} CASSEE(S)"
        print(f"{str(chemin):<40} {n:2d} cellules de code  ->  {etat}")
        for index, erreur, source in problemes:
            print(f"   cellule {index}, ligne {erreur.lineno} : {erreur.msg}")
            for numero, ligne in enumerate(source.splitlines(), 1):
                marque = ">>" if numero == erreur.lineno else "  "
                print(f"   {marque} {numero:2d} | {ligne}")
    print(f"\n{total} cellules verifiees, {casses} cassee(s)")
    return 1 if casses else 0


if __name__ == "__main__":
    sys.exit(principal())

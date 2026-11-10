"""
Etape 4 - Les figures du rapport, toutes construites depuis results.csv.

Une seule source de verite : si une ligne change dans results.csv, les quatre
figures se mettent a jour ensemble. Aucun chiffre n'est recopie a la main.

    python src/figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import assurer_dossier, chemin_projet, info  # noqa: E402

# Palette sobre et lisible en niveaux de gris, pour l'impression du rapport
BLEU, ORANGE, VERT, ROUGE, GRIS = "#2d6cb5", "#e07b39", "#3f8f5c", "#c0392b", "#7b7b85"


def charger() -> pd.DataFrame:
    chemin = chemin_projet("results/results.csv")
    if not chemin.exists():
        sys.exit("ERREUR : results/results.csv absent.")
    df = pd.read_csv(chemin)
    # Les lignes produites avant l'Etape 4 n'ont pas les colonnes rang et pas :
    # on les cree vides plutot que de faire echouer toutes les figures.
    for colonne in ("rang", "pas", "intensite"):
        if colonne not in df.columns:
            df[colonne] = pd.NA
        df[colonne] = pd.to_numeric(df[colonne], errors="coerce")
    if "nom" not in df.columns:
        sys.exit("ERREUR : results.csv ne contient pas de colonne 'nom'.")
    return df


def reference(df: pd.DataFrame, colonne: str) -> float | None:
    ligne = df[df["nom"] == "baseline"]
    return float(ligne.iloc[0][colonne]) if len(ligne) else None


# --------------------------------------------------------------------------
# Figure 1 - le compromis
# --------------------------------------------------------------------------

def figure_compromis(df: pd.DataFrame, sortie: Path) -> None:
    """Fidelite au style contre respect du prompt : le front de Pareto.

    Chaque point est une configuration. Celles situees en haut a droite
    dominent : plus de style ET plus d'obeissance au texte. La figure montre
    s'il existe vraiment un arbitrage, ou si certaines configurations sont
    simplement meilleures que d'autres.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for _, l in df.iterrows():
        base = l["nom"] == "baseline"
        ax.scatter(l["prompt_domaine"], l["style_dino_moy"],
                   s=190 if base else 120,
                   c=GRIS if base else BLEU, marker="s" if base else "o",
                   edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(l["nom"], (l["prompt_domaine"], l["style_dino_moy"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)

    x0, y0 = reference(df, "prompt_domaine"), reference(df, "style_dino_moy")
    if x0 and y0:
        ax.axhline(y0, color=GRIS, ls=":", lw=1)
        ax.axvline(x0, color=GRIS, ls=":", lw=1)
    ax.set_xlabel("respect du prompt — CLIPScore sur les prompts du domaine")
    ax.set_ylabel("fidelite au style — similarite DINOv2")
    ax.set_title("Le compromis : style contre obeissance au texte")
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(sortie / "compromis.png", dpi=140); plt.close(fig)
    info("compromis.png")


# --------------------------------------------------------------------------
# Figure 2 - effet de l'intensite
# --------------------------------------------------------------------------

def figure_intensite(df: pd.DataFrame, sortie: Path) -> None:
    """Le curseur d'intensite, de 0 a 1,25, sur la meme LoRA."""
    serie = df[df["nom"].str.startswith("intensite_", na=False)].sort_values("intensite")
    if len(serie) < 2:
        info("intensite.png ignoree (moins de deux points)")
        return

    fig, (g, d) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    g.plot(serie["intensite"], serie["style_dino_moy"], "o-", color=BLEU)
    y0 = reference(df, "style_dino_moy")
    if y0:
        g.axhline(y0, color=GRIS, ls="--", lw=1, label="baseline")
        g.legend(fontsize=8)
    g.set_ylabel("style (DINOv2)"); g.set_xlabel("intensite de la LoRA")
    g.set_title("Le style monte avec l'intensite"); g.grid(alpha=.25)

    for colonne, couleur, etiquette in [
            ("prompt_domaine", BLEU, "domaine"),
            ("prompt_transfert", ORANGE, "transfert"),
            ("prompt_hors_domaine", VERT, "hors domaine")]:
        d.plot(serie["intensite"], serie[colonne], "o-", color=couleur, label=etiquette)
    d.set_ylabel("respect du prompt (CLIPScore)"); d.set_xlabel("intensite de la LoRA")
    d.set_title("Ce que l'intensite coute au texte"); d.legend(fontsize=8); d.grid(alpha=.25)

    fig.tight_layout(); fig.savefig(sortie / "intensite.png", dpi=140); plt.close(fig)
    info("intensite.png")


# --------------------------------------------------------------------------
# Figure 3 - effet du nombre de pas
# --------------------------------------------------------------------------

def figure_pas(df: pd.DataFrame, sortie: Path) -> None:
    """Style et memorisation en fonction de la duree d'entrainement."""
    # Le filtre porte sur le NOM, pas sur rang == 8 : le balayage d'intensite
    # et le balayage de rang produisent eux aussi des lignes de rang 8, avec
    # une intensite ou un budget differents. Les melanger donnerait une courbe
    # qui n'a aucun sens -- plusieurs points a 1000 pas, et un point a 400 pas
    # issu d'un budget d'entrainement different.
    serie = df[df["nom"].str.fullmatch(r"r8_\d+pas", na=False)].sort_values("pas")
    if len(serie) < 2:
        info("pas.png ignoree (moins de deux points)")
        return

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(serie["pas"], serie["style_dino_moy"], "o-", color=BLEU, label="style (DINOv2)")
    ax.set_xlabel("pas d'entrainement"); ax.set_ylabel("style (DINOv2)", color=BLEU)
    ax.tick_params(axis="y", labelcolor=BLEU); ax.grid(alpha=.25)

    jumeau = ax.twinx()
    jumeau.plot(serie["pas"], serie["memorisation_max_dino"], "s--", color=ROUGE,
                label="memorisation")
    m0 = reference(df, "memorisation_max_dino")
    if m0:
        jumeau.axhline(m0, color=GRIS, ls=":", lw=1)
        jumeau.annotate("niveau du hasard", (serie["pas"].iloc[0], m0), fontsize=8,
                        color=GRIS, textcoords="offset points", xytext=(4, 4))
    jumeau.set_ylabel("memorisation (max DINOv2)", color=ROUGE)
    jumeau.tick_params(axis="y", labelcolor=ROUGE)

    ax.set_title("Entrainer plus longtemps : le style gagne, la memorisation aussi")
    fig.tight_layout(); fig.savefig(sortie / "pas.png", dpi=140); plt.close(fig)
    info("pas.png")


# --------------------------------------------------------------------------
# Figure 4 - effet du rang
# --------------------------------------------------------------------------

def figure_rangs(df: pd.DataFrame, sortie: Path) -> None:
    """Quatre rangs a budget identique : style, texte, memorisation, diversite."""
    serie = df[df["nom"].str.fullmatch(r"lora_r\d+_400pas", na=False)].sort_values("rang")
    if len(serie) < 2:
        info("rangs.png ignoree (moins de deux points)")
        return

    mesures = [("style_dino_moy", "style (DINOv2)", BLEU),
               ("prompt_hors_domaine", "prompt hors domaine", VERT),
               ("memorisation_max_dino", "memorisation", ROUGE),
               ("diversite_lpips", "diversite (LPIPS)", ORANGE)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, (colonne, titre, couleur) in zip(axes, mesures):
        ax.plot(serie["rang"], serie[colonne], "o-", color=couleur)
        valeur = reference(df, colonne)
        if valeur:
            ax.axhline(valeur, color=GRIS, ls="--", lw=1)
        ax.set_xscale("log", base=2); ax.set_xticks(serie["rang"])
        ax.set_xticklabels([int(r) for r in serie["rang"]])
        ax.set_xlabel("rang LoRA"); ax.set_title(titre, fontsize=10); ax.grid(alpha=.25)
    fig.suptitle("Effet du rang, a budget d'entrainement identique "
                 "(pointilles : baseline)", fontsize=12)
    fig.tight_layout(); fig.savefig(sortie / "rangs.png", dpi=140); plt.close(fig)
    info("rangs.png")


def principal() -> None:
    df = charger()
    sortie = assurer_dossier("results/figures")
    info(f"{len(df)} configurations dans results.csv")
    figure_compromis(df, sortie)
    figure_intensite(df, sortie)
    figure_pas(df, sortie)
    figure_rangs(df, sortie)
    info(f"figures -> {sortie}")


if __name__ == "__main__":
    principal()

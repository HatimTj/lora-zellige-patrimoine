#!/usr/bin/env python3
"""Etape 5 - le modele a-t-il appris un style, ou recopie ses images ?

Une LoRA entrainee sur 29 images peut tres bien ne rien avoir appris du tout
et se contenter de restituer ce qu'elle a vu. Le score de style monterait
quand meme, puisqu'une copie ressemble forcement au jeu d'entrainement. Il
faut donc une mesure separee, et surtout un point de comparaison honnete.

Trois questions, dans l'ordre ou elles se posent :

  Q1. A partir de quelle similarite peut-on parler de recopie ?
      On ne le decide pas a la main. Les 8 images du holdout sont de vraies
      photos de zellige que le modele n'a JAMAIS vues. On mesure a quel point
      chacune ressemble a l'image d'entrainement la plus proche : c'est la
      ressemblance normale entre deux photos independantes du meme artisanat.
      Une image generee qui reste dans cette fourchette ne copie pas plus que
      deux carreaux differents ne se ressemblent entre eux.

  Q2. Quelles images generees s'approchent le plus du jeu d'entrainement ?
      On les classe, on affiche les pires paires cote a cote, et on juge a
      l'oeil. Un chiffre seul ne distingue pas une copie d'un motif commun.

  Q3. Le modele a-t-il oublie le reste du monde ?
      On compare la baseline et la LoRA sur les prompts hors domaine
      (un chien, un paysage, un portrait). Si la LoRA les a abimes, elle a
      sur-appris : elle ne sait plus faire que du zellige.

Usage :

    python src/overfit_check.py --lora results/r8_1000pas --base results/baseline

Sorties : results/surapprentissage.json et trois figures dans results/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import assurer_dossier, chemin_projet, ecrire_json, info  # noqa: E402
import metrics as M  # noqa: E402


def lire_evaluation(dossier: Path) -> Dict[str, Any]:
    """Relit le evaluation.json produit par evaluate.py."""
    chemin = dossier / "evaluation.json"
    if not chemin.exists():
        sys.exit(f"ERREUR : {chemin} est absent. Lancez d'abord "
                 f"src/evaluate.py sur {dossier}.")
    return json.loads(chemin.read_text(encoding="utf-8"))


def exiger_images(dossier: Path, attendus: List[str]) -> None:
    """Les images ne sont pas versionnees : on verifie avant de charger."""
    manquants = [n for n in attendus if not (dossier / n).exists()]
    if manquants:
        sys.exit(f"ERREUR : {len(manquants)} des {len(attendus)} images de "
                 f"{dossier} sont absentes (la premiere : {manquants[0]}). "
                 f"Les images ne sont pas dans Git : relancez la generation.")


def moyenne(valeurs: List[float]) -> float:
    return round(sum(valeurs) / len(valeurs), 4) if valeurs else 0.0


# --------------------------------------------------------------------------
# Q1 - le seuil de hasard, mesure sur le holdout
# --------------------------------------------------------------------------

def seuil_par_holdout(dossier_train: Path, dossier_holdout: Path) -> Dict[str, Any]:
    """Ressemblance normale entre deux vraies photos de zellige.

    Chaque image du holdout est comparee a l'entrainement. Le maximum obtenu
    est la barre a ne pas depasser : au-dela, une image generee ressemble a
    une image d'entrainement plus que deux photos distinctes ne se ressemblent.
    """
    train = M.lire_images(dossier_train)
    holdout = M.lire_images(dossier_holdout)
    if not train or not holdout:
        sys.exit(f"ERREUR : {dossier_train} ou {dossier_holdout} est vide. "
                 "Relancez src/prepare_data.py.")

    v_train = M.vecteurs_dino([im for _, im in train])
    v_holdout = M.vecteurs_dino([im for _, im in holdout])
    sim = M.similarite_au_jeu(v_holdout, v_train)

    noms_train = [n for n, _ in train]
    detail = [{"holdout": n,
               "plus_proche_train": noms_train[sim["indice_plus_proche"][i]],
               "similarite_max": sim["maximum"][i]}
              for i, (n, _) in enumerate(holdout)]

    maxima = sim["maximum"]
    return {
        "n_holdout": len(holdout),
        "n_train": len(train),
        "similarite_max_moyenne": moyenne(maxima),
        "similarite_max_minimum": round(min(maxima), 4),
        "seuil": round(max(maxima), 4),
        "detail": detail,
    }


# --------------------------------------------------------------------------
# Q2 - les paires les plus proches
# --------------------------------------------------------------------------

def plus_proches_voisins(dossier_gen: Path, dossier_train: Path,
                         n_paires: int) -> Dict[str, Any]:
    """Classe les images generees par ressemblance maximale a l'entrainement."""
    evaluation = lire_evaluation(dossier_gen)
    par_image = evaluation["par_image"]

    # Le style ne se juge que sur les prompts censes produire du zellige :
    # un chien hors domaine n'a aucune raison de ressembler a un carreau.
    retenues = [e for e in par_image
                if e["categorie"] in ("domaine", "transfert")]
    exiger_images(dossier_gen, [e["fichier"] for e in retenues])

    classees = sorted(retenues, key=lambda e: -e["memorisation_dino_max"])
    return {
        "n_examinees": len(retenues),
        "paires": [{"generee": e["fichier"],
                    "train": e["image_la_plus_proche"],
                    "similarite": e["memorisation_dino_max"]}
                   for e in classees[:n_paires]],
        "toutes": [{"generee": e["fichier"],
                    "train": e["image_la_plus_proche"],
                    "similarite": e["memorisation_dino_max"]}
                   for e in classees],
    }


# --------------------------------------------------------------------------
# Q3 - le modele sait-il encore faire autre chose ?
# --------------------------------------------------------------------------

def degradation_hors_domaine(dossier_lora: Path, dossier_base: Path) -> Dict[str, Any]:
    """Compare la baseline et la LoRA sur les prompts hors domaine."""
    ev_lora = lire_evaluation(dossier_lora)["par_image"]
    ev_base = lire_evaluation(dossier_base)["par_image"]

    base = {e["fichier"]: e for e in ev_base}
    lignes = []
    for e in ev_lora:
        if e["categorie"] != "hors_domaine":
            continue
        ref = base.get(e["fichier"])
        if ref is None:
            continue
        lignes.append({
            "fichier": e["fichier"],
            "prompt_id": e["prompt_id"],
            "clipscore_base": ref["clipscore"],
            "clipscore_lora": e["clipscore"],
            "ecart": round(e["clipscore"] - ref["clipscore"], 4),
        })

    if not lignes:
        sys.exit("ERREUR : aucune image hors domaine commune aux deux dossiers.")

    ecarts = [l["ecart"] for l in lignes]
    return {
        "n_images": len(lignes),
        "clipscore_base": moyenne([l["clipscore_base"] for l in lignes]),
        "clipscore_lora": moyenne([l["clipscore_lora"] for l in lignes]),
        "ecart_moyen": moyenne(ecarts),
        "pire_ecart": round(min(ecarts), 4),
        "detail": lignes,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def figure_seuil(seuil: Dict[str, Any], voisins: Dict[str, Any],
                 sortie: Path) -> None:
    """Les deux distributions face a face : holdout contre images generees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = [d["similarite_max"] for d in seuil["detail"]]
    g = [p["similarite"] for p in voisins["toutes"]]
    barre = seuil["seuil"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.scatter(h, [1] * len(h), s=70, alpha=0.75, color="#2a7ab9",
               label=f"holdout, {len(h)} vraies photos jamais vues", zorder=3)
    ax.scatter(g, [0] * len(g), s=45, alpha=0.55, color="#c8482a",
               label=f"images generees, {len(g)} zellige", zorder=3)
    ax.axvline(barre, color="#444", ls="--", lw=1.2,
               label=f"seuil de recopie = {barre:.3f}")

    depassent = [v for v in g if v > barre]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["genere", "holdout"])
    ax.set_xlabel("similarite DINOv2 a l'image d'entrainement la plus proche")
    ax.set_title("Jusqu'ou peut-on ressembler au jeu d'entrainement sans le copier ?\n"
                 f"{len(depassent)} image(s) generee(s) au-dessus du seuil",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    # Legende sous l'axe : au centre elle recouvrirait les points generes,
    # qui s'etalent sur toute la largeur.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=8.5, frameon=False)
    ax.set_ylim(-0.6, 1.6)
    plt.tight_layout()
    plt.savefig(sortie, dpi=120)
    plt.close()
    info(sortie.name)


def figure_paires(voisins: Dict[str, Any], dossier_gen: Path,
                  dossier_train: Path, seuil: float, sortie: Path) -> None:
    """Les pires paires cote a cote : a l'oeil, est-ce une copie ?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    paires = voisins["paires"]
    n = len(paires)
    fig, axes = plt.subplots(2, n, figsize=(2.3 * n, 5.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, p in enumerate(paires):
        axes[0, j].imshow(Image.open(dossier_gen / p["generee"]))
        couleur = "#c8482a" if p["similarite"] > seuil else "#2a7ab9"
        axes[0, j].set_title(f"{p['similarite']:.3f}", fontsize=10,
                             color=couleur, weight="bold")
        axes[1, j].imshow(Image.open(dossier_train / p["train"]))
        for i in (0, 1):
            axes[i, j].axis("off")

    axes[0, 0].text(-0.08, 0.5, "genere", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=11,
                    weight="bold")
    axes[1, 0].text(-0.08, 0.5, "train le\nplus proche",
                    transform=axes[1, 0].transAxes, rotation=90, va="center",
                    ha="right", fontsize=11, weight="bold")
    plt.suptitle("Les paires les plus ressemblantes, classees\n"
                 f"(rouge : au-dessus du seuil de recopie {seuil:.3f})",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(sortie, dpi=110)
    plt.close()
    info(sortie.name)


def figure_hors_domaine(degradation: Dict[str, Any], dossier_lora: Path,
                        dossier_base: Path, sortie: Path) -> None:
    """Le modele sait-il encore dessiner un chien ?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    lignes = degradation["detail"]
    # Une image par prompt hors domaine, celle dont l'ecart est le pire :
    # on montre le cas le plus defavorable a la LoRA, pas le plus flatteur.
    par_prompt: Dict[str, Dict[str, Any]] = {}
    for l in lignes:
        pid = l["prompt_id"]
        if pid not in par_prompt or l["ecart"] < par_prompt[pid]["ecart"]:
            par_prompt[pid] = l
    choisies = [par_prompt[k] for k in sorted(par_prompt)]

    manquantes = [l["fichier"] for l in choisies
                  if not (dossier_base / l["fichier"]).exists()
                  or not (dossier_lora / l["fichier"]).exists()]
    if manquantes:
        info(f"figure hors_domaine ignoree : {len(manquantes)} image(s) absente(s)")
        return

    n = len(choisies)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.2))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, l in enumerate(choisies):
        axes[0, j].imshow(Image.open(dossier_base / l["fichier"]))
        axes[0, j].set_title(f"{l['prompt_id']}\nbase {l['clipscore_base']:.1f}",
                             fontsize=9.5)
        axes[1, j].imshow(Image.open(dossier_lora / l["fichier"]))
        signe = "+" if l["ecart"] >= 0 else ""
        axes[1, j].set_title(f"LoRA {l['clipscore_lora']:.1f}  "
                             f"({signe}{l['ecart']:.1f})", fontsize=9.5)
        for i in (0, 1):
            axes[i, j].axis("off")

    for i, etiquette in enumerate(("modele\nde base", "LoRA")):
        axes[i, 0].text(-0.06, 0.5, etiquette, transform=axes[i, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=10.5,
                        weight="bold")

    plt.suptitle("Prompts hors domaine : la LoRA a-t-elle fait oublier le reste ?\n"
                 f"CLIPScore moyen {degradation['clipscore_base']:.2f} -> "
                 f"{degradation['clipscore_lora']:.2f}", fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.12)
    plt.savefig(sortie, dpi=110)
    plt.close()
    info(sortie.name)


# --------------------------------------------------------------------------

def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--lora", default="results/r8_1000pas",
                         help="dossier des images produites avec la LoRA")
    parseur.add_argument("--base", default="results/baseline",
                         help="dossier des images du modele de base")
    parseur.add_argument("--train", default="data/processed/train",
                         help="images d'entrainement")
    parseur.add_argument("--holdout", default="data/processed/holdout",
                         help="images reservees, jamais vues a l'entrainement")
    parseur.add_argument("--n-paires", type=int, default=8,
                         help="nombre de paires affichees sur la figure")
    parseur.add_argument("--sortie", default="results/surapprentissage.json")
    args = parseur.parse_args()

    d_lora = chemin_projet(args.lora)
    d_base = chemin_projet(args.base)
    d_train = chemin_projet(args.train)
    d_holdout = chemin_projet(args.holdout)
    figures = assurer_dossier("results/figures")

    info("Q1 : ressemblance normale entre deux vraies photos de zellige")
    seuil = seuil_par_holdout(d_train, d_holdout)
    info(f"     seuil = {seuil['seuil']} "
         f"(moyenne {seuil['similarite_max_moyenne']} sur {seuil['n_holdout']} images)")

    info("Q2 : les images generees les plus proches de l'entrainement")
    voisins = plus_proches_voisins(d_lora, d_train, args.n_paires)
    au_dessus = [p for p in voisins["toutes"] if p["similarite"] > seuil["seuil"]]
    info(f"     {len(au_dessus)} / {voisins['n_examinees']} au-dessus du seuil")

    info("Q3 : le modele sait-il encore faire autre chose que du zellige ?")
    degradation = degradation_hors_domaine(d_lora, d_base)
    info(f"     CLIPScore hors domaine {degradation['clipscore_base']} -> "
         f"{degradation['clipscore_lora']} ({degradation['ecart_moyen']:+})")

    figure_seuil(seuil, voisins, figures / "memorisation_seuil.png")
    figure_paires(voisins, d_lora, d_train, seuil["seuil"],
                  figures / "plus_proches_voisins.png")
    figure_hors_domaine(degradation, d_lora, d_base,
                        figures / "hors_domaine.png")

    resume = {
        "dossier_lora": args.lora,
        "dossier_base": args.base,
        "seuil_de_recopie": seuil,
        "plus_proches_voisins": voisins,
        "degradation_hors_domaine": degradation,
        "verdict": {
            "images_au_dessus_du_seuil": len(au_dessus),
            "images_examinees": voisins["n_examinees"],
            "part_au_dessus_pct": round(100 * len(au_dessus)
                                        / max(voisins["n_examinees"], 1), 1),
            "similarite_max_observee": voisins["toutes"][0]["similarite"]
                                       if voisins["toutes"] else None,
        },
    }
    ecrire_json(resume, chemin_projet(args.sortie))

    v = resume["verdict"]
    print("\n=== Etape 5 - surapprentissage ===")
    print(f"seuil de recopie (holdout)      : {seuil['seuil']}")
    print(f"similarite maximale observee    : {v['similarite_max_observee']}")
    print(f"images au-dessus du seuil       : {v['images_au_dessus_du_seuil']} "
          f"/ {v['images_examinees']}  ({v['part_au_dessus_pct']} %)")
    print(f"CLIPScore hors domaine          : {degradation['clipscore_base']} -> "
          f"{degradation['clipscore_lora']}  ({degradation['ecart_moyen']:+})")
    print(f"\ndetail : {args.sortie}")


if __name__ == "__main__":
    principal()

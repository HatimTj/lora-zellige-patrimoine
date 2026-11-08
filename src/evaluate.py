"""
Etape 2 et 4 - Evaluation d'un dossier d'images generees.

Lit un dossier produit par src/generate.py, calcule toutes les mesures et
ajoute UNE LIGNE PAR CONFIGURATION dans results/results.csv. Ce fichier unique
est la source de toutes les figures du rapport.

    python src/evaluate.py --dossier results/baseline --nom baseline
    python src/evaluate.py --dossier results/r8 --nom "lora_r8_1000pas"

Detail par image dans <dossier>/evaluation.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import chemin_projet, ecrire_json, info  # noqa: E402
import metrics as M  # noqa: E402

COLONNES = [
    "nom", "lora", "intensite", "rang", "pas",
    "style_clip_moy", "style_dino_moy",          # Q1 : a-t-il appris le style
    "prompt_domaine", "prompt_transfert", "prompt_hors_domaine",   # Q2 : texte
    "duree_par_image_s", "vram_pic_go", "taille_lora_mo",          # Q3 : cout
    "memorisation_max_dino", "image_la_plus_copiee", "diversite_lpips",  # Q4
    "n_images",
]


def moyenne(valeurs: List[float]) -> float:
    valeurs = [v for v in valeurs if v == v]          # retire les NaN
    return round(sum(valeurs) / len(valeurs), 4) if valeurs else float("nan")


def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--dossier", required=True)
    parseur.add_argument("--nom", required=True,
                         help="nom de la configuration dans results.csv")
    parseur.add_argument("--reference", default="data/processed/train",
                         help="jeu de reference pour la similarite de style")
    parseur.add_argument("--rang", type=int, default=None,
                         help="rang LoRA, pour tracer les courbes de l'Etape 4")
    parseur.add_argument("--pas", type=int, default=None,
                         help="nombre de pas d'entrainement, meme usage")
    args = parseur.parse_args()

    dossier = chemin_projet(args.dossier)
    manifeste = json.loads((dossier / "manifeste.json").read_text(encoding="utf-8"))

    # 1. Images generees, dans l'ordre du manifeste
    noms = [e["fichier"] for e in manifeste["images"]]
    from PIL import Image
    generees = [Image.open(dossier / n).convert("RGB") for n in noms]
    textes = [e["texte"] for e in manifeste["images"]]
    categories = [e["categorie"] for e in manifeste["images"]]
    prompts_ids = [e["prompt_id"] for e in manifeste["images"]]

    # 2. Images de reference (celles qui ont servi a entrainer)
    reference = M.lire_images(chemin_projet(args.reference))
    noms_ref = [n for n, _ in reference]
    images_ref = [im for _, im in reference]
    info(f"{len(generees)} images generees, {len(images_ref)} images de reference")

    # 3. Q2 - respect du prompt
    scores_prompt = M.clipscore(generees, textes)

    # 4. Q1 et Q4 - style et memorisation
    v_clip_gen = M.vecteurs_clip_image(generees)
    v_clip_ref = M.vecteurs_clip_image(images_ref)
    sim_clip = M.similarite_au_jeu(v_clip_gen, v_clip_ref)

    v_dino_gen = M.vecteurs_dino(generees)
    v_dino_ref = M.vecteurs_dino(images_ref)
    sim_dino = M.similarite_au_jeu(v_dino_gen, v_dino_ref)

    # 5. Q4 - diversite, prompt par prompt
    groupes: Dict[str, List] = defaultdict(list)
    for pid, image in zip(prompts_ids, generees):
        groupes[pid].append(image)
    diversites = {pid: M.diversite_lpips(ims) for pid, ims in groupes.items()}

    # 6. Detail par image
    detail = []
    for i, nom in enumerate(noms):
        detail.append({
            "fichier": nom,
            "prompt_id": prompts_ids[i],
            "categorie": categories[i],
            "clipscore": scores_prompt[i],
            "style_clip_moy": sim_clip["moyenne"][i],
            "style_dino_moy": sim_dino["moyenne"][i],
            "memorisation_clip_max": sim_clip["maximum"][i],
            "memorisation_dino_max": sim_dino["maximum"][i],
            "image_la_plus_proche": noms_ref[sim_dino["indice_plus_proche"][i]],
        })
    ecrire_json({"configuration": args.nom, "par_image": detail,
                 "diversite_par_prompt": diversites},
                dossier / "evaluation.json")

    # 7. Agregation : une ligne pour results.csv
    def par_categorie(cat: str) -> float:
        return moyenne([scores_prompt[i] for i, c in enumerate(categories) if c == cat])

    # Le style ne se mesure que sur les prompts censes produire du zellige
    indices_zellige = [i for i, c in enumerate(categories)
                       if c in ("domaine", "transfert")]
    pire = max(indices_zellige, key=lambda i: sim_dino["maximum"][i])

    ligne = {
        "nom": args.nom,
        "lora": manifeste.get("lora") or "aucune",
        "intensite": manifeste.get("intensite", 0.0),
        "rang": args.rang if args.rang is not None else "",
        "pas": args.pas if args.pas is not None else "",
        "style_clip_moy": moyenne([sim_clip["moyenne"][i] for i in indices_zellige]),
        "style_dino_moy": moyenne([sim_dino["moyenne"][i] for i in indices_zellige]),
        "prompt_domaine": par_categorie("domaine"),
        "prompt_transfert": par_categorie("transfert"),
        "prompt_hors_domaine": par_categorie("hors_domaine"),
        "duree_par_image_s": manifeste.get("duree_par_image_s"),
        "vram_pic_go": manifeste.get("vram_pic_go"),
        "taille_lora_mo": manifeste.get("taille_lora_mo", ""),
        "memorisation_max_dino": round(max(sim_dino["maximum"][i]
                                           for i in indices_zellige), 4),
        "image_la_plus_copiee": f"{noms[pire]} ~ "
                                f"{noms_ref[sim_dino['indice_plus_proche'][pire]]}",
        "diversite_lpips": moyenne(list(diversites.values())),
        "n_images": len(generees),
    }

    chemin_csv = chemin_projet("results/results.csv")
    chemin_csv.parent.mkdir(parents=True, exist_ok=True)
    existantes = []
    if chemin_csv.exists():
        with open(chemin_csv, encoding="utf-8") as f:
            existantes = [l for l in csv.DictReader(f) if l["nom"] != args.nom]
    with open(chemin_csv, "w", encoding="utf-8", newline="") as f:
        redacteur = csv.DictWriter(f, fieldnames=COLONNES, restval="",
                                   extrasaction="ignore")
        redacteur.writeheader()
        redacteur.writerows(existantes + [ligne])

    print(f"\n=== {args.nom} ===")
    for cle, valeur in ligne.items():
        print(f"  {cle:<24} {valeur}")
    info(f"ligne ajoutee a {chemin_csv}")


if __name__ == "__main__":
    principal()

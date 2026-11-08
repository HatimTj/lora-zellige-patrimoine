"""
Etape 2 et 4 - Generation d'images.

Un seul script sert a tout : la baseline (sans LoRA) et chaque configuration
entrainee. C'est voulu -- si la baseline et les LoRA etaient generees par deux
codes differents, une partie de l'ecart mesure viendrait du code, pas du modele.

    python src/generate.py --sortie results/baseline
    python src/generate.py --lora runs/r8 --intensite 0.8 --sortie results/r8

Produit, dans le dossier de sortie :
    <ID_PROMPT>_s<GRAINE>.png   une image par couple prompt x graine
    manifeste.json              prompts, graines, duree, VRAM pic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (assurer_dossier, charger_config, chemin_projet,  # noqa: E402
                    ecrire_json, fixer_graine, info, mesurer, neutraliser_torchao,
                    reinitialiser_vram, taille_fichier_mo, vram_pic_go)


def charger_pipeline(cfg: Dict[str, Any], lora: str | None):
    """Charge Stable Diffusion, et branche une LoRA si on en fournit une."""
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    modele = cfg["modele_base"]["id"]
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    info(f"chargement de {modele} ({dtype})")
    pipe = StableDiffusionPipeline.from_pretrained(
        modele,
        torch_dtype=dtype,
        safety_checker=None,        # motifs geometriques : aucun risque, et on
        requires_safety_checker=False,   # gagne de la VRAM et du temps
    )
    # DPM-Solver++ : qualite equivalente en 30 etapes la ou PNDM en demande 50
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    pipe.set_progress_bar_config(disable=True)

    if lora:
        neutraliser_torchao()      # load_lora_weights passe par les memes dispatchers
        chemin = chemin_projet(lora)
        info(f"chargement de la LoRA : {chemin}")
        pipe.load_lora_weights(str(chemin))

    return pipe


def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", default="configs/base.yaml")
    parseur.add_argument("--prompts", default="configs/eval_prompts.yaml")
    parseur.add_argument("--lora", default=None,
                         help="dossier d'une LoRA entrainee ; absent = baseline")
    parseur.add_argument("--intensite", type=float, default=1.0,
                         help="force de la LoRA a la generation (0 = modele de base)")
    parseur.add_argument("--sortie", required=True)
    parseur.add_argument("--rapide", action="store_true",
                         help="une seule image, pour verifier la chaine")
    args = parseur.parse_args()

    cfg = charger_config(args.config)
    with open(chemin_projet(args.prompts), encoding="utf-8") as f:
        pcfg = yaml.safe_load(f)

    prompts: List[Dict[str, str]] = pcfg["prompts"]
    graines: List[int] = cfg["reproductibilite"]["seeds_evaluation"]
    if args.rapide:
        prompts, graines = prompts[:1], graines[:1]
        info("mode rapide : 1 prompt, 1 graine")
    gen = pcfg["generation"]
    sortie = assurer_dossier(args.sortie)

    pipe = charger_pipeline(cfg, args.lora)
    kwargs: Dict[str, Any] = {}
    if args.lora:
        # diffusers applique l'intensite via cross_attention_kwargs
        kwargs["cross_attention_kwargs"] = {"scale": args.intensite}

    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    reinitialiser_vram()
    entrees: List[Dict[str, Any]] = []
    journal: Dict[str, Any] = {}

    with mesurer("generation complete", journal):
        for prompt in prompts:
            for graine in graines:
                fixer_graine(graine)
                generateur = torch.Generator(device=appareil).manual_seed(graine)
                image = pipe(
                    prompt["texte"],
                    negative_prompt=pcfg.get("negatif") or None,
                    num_inference_steps=gen["etapes_debruitage"],
                    guidance_scale=gen["guidance"],
                    height=gen["taille"], width=gen["taille"],
                    generator=generateur,
                    **kwargs,
                ).images[0]

                nom = f"{prompt['id']}_s{graine}.png"
                image.save(sortie / nom)
                entrees.append({
                    "fichier": nom,
                    "prompt_id": prompt["id"],
                    "categorie": prompt["categorie"],
                    "texte": prompt["texte"],
                    "graine": graine,
                })
            info(f"  {prompt['id']} : {len(graines)} images")

    duree = journal["generation complete"]["duree_s"]
    manifeste = {
        "lora": args.lora,
        "intensite": args.intensite if args.lora else 0.0,
        "modele_base": cfg["modele_base"]["id"],
        "etapes_debruitage": gen["etapes_debruitage"],
        "guidance": gen["guidance"],
        "taille": gen["taille"],
        "n_images": len(entrees),
        "duree_totale_s": duree,
        "duree_par_image_s": round(duree / max(len(entrees), 1), 3),
        "vram_pic_go": vram_pic_go(),
        "images": entrees,
    }
    if args.lora:
        poids = list(chemin_projet(args.lora).glob("*.safetensors"))
        if poids:
            manifeste["taille_lora_mo"] = taille_fichier_mo(poids[0])

    ecrire_json(manifeste, Path(sortie) / "manifeste.json")
    info(f"{len(entrees)} images -> {sortie}")
    info(f"{manifeste['duree_par_image_s']} s par image, VRAM pic "
         f"{manifeste['vram_pic_go']} Go")


if __name__ == "__main__":
    principal()

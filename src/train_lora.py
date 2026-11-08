"""
Etape 3 et 4 - Entrainement d'une LoRA sur Stable Diffusion 1.5.

Ce que fait concretement l'entrainement :
  1. on prend une image du jeu, on la compresse en latent avec le VAE
  2. on y ajoute du bruit, d'une intensite tiree au hasard
  3. on demande au U-Net de predire le bruit ajoute, en lui donnant la legende
  4. on compare sa prediction au vrai bruit, et on corrige

Tout le modele est GELE. Seules les matrices LoRA, branchees sur les quatre
projections de l'attention, recoivent des gradients : environ 1,5 million de
parametres entrainables contre 860 millions, soit 0,2 %.

    python src/train_lora.py --config configs/lora_r8.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (assurer_dossier, charger_config, chemin_projet,  # noqa: E402
                    ecrire_json, fixer_graine, info, reinitialiser_vram,
                    taille_fichier_mo, vram_pic_go)


# --------------------------------------------------------------------------
# Jeu de donnees
# --------------------------------------------------------------------------

class JeuZellige(Dataset):
    """Images 512x512 et leurs legendes, lues depuis metadata.jsonl."""

    def __init__(self, dossier: Path, tokenizer, retournement: bool = True):
        self.dossier = dossier
        self.retournement = retournement
        fichier = dossier / "metadata.jsonl"
        if not fichier.exists():
            sys.exit(f"ERREUR : {fichier} absent. Lancez `python src/legender.py`.")
        self.entrees = [json.loads(l) for l in
                        fichier.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.entrees:
            sys.exit(f"ERREUR : {fichier} est vide.")
        self.tokenizer = tokenizer
        info(f"{len(self.entrees)} images d'entrainement")

    def __len__(self) -> int:
        return len(self.entrees)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        entree = self.entrees[index]
        image = Image.open(self.dossier / entree["file_name"]).convert("RGB")

        # Les motifs de zellige sont symetriques : le retournement horizontal
        # produit une variante parfaitement plausible, et double gratuitement
        # la taille apparente du jeu -- precieux avec seulement 29 images.
        if self.retournement and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        import numpy as np
        tableau = np.asarray(image, dtype="float32") / 127.5 - 1.0   # vers [-1, 1]
        pixels = torch.from_numpy(tableau).permute(2, 0, 1)

        jetons = self.tokenizer(
            entree["text"], padding="max_length", truncation=True,
            max_length=self.tokenizer.model_max_length, return_tensors="pt",
        )
        return {"pixels": pixels, "ids": jetons.input_ids[0]}


# --------------------------------------------------------------------------
# Sauvegarde
# --------------------------------------------------------------------------

def sauver_lora(unet, dossier: Path) -> Path:
    """Ecrit les seuls poids LoRA, au format lisible par load_lora_weights."""
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    dossier.mkdir(parents=True, exist_ok=True)
    etat = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(
        save_directory=str(dossier), unet_lora_layers=etat, safe_serialization=True)
    return dossier


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", required=True)
    parseur.add_argument("--base", default="configs/base.yaml")
    args = parseur.parse_args()

    base = charger_config(args.base)
    with open(chemin_projet(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clora, cent = cfg["lora"], cfg["entrainement"]
    fixer_graine(cent["graine"])

    if not torch.cuda.is_available():
        sys.exit("ERREUR : aucun GPU. Lancez ce script sur Colab ou Kaggle.")
    appareil = "cuda"

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from peft import LoraConfig
    from transformers import CLIPTextModel, CLIPTokenizer

    modele = base["modele_base"]["id"]
    info(f"chargement de {modele}")
    tokenizer = CLIPTokenizer.from_pretrained(modele, subfolder="tokenizer")
    encodeur_texte = CLIPTextModel.from_pretrained(modele, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(modele, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(modele, subfolder="unet")
    bruiteur = DDPMScheduler.from_pretrained(modele, subfolder="scheduler")

    # 1. Tout geler
    for module in (encodeur_texte, vae, unet):
        module.requires_grad_(False)
        module.to(appareil)
    encodeur_texte.eval()
    vae.eval()

    # 2. Brancher la LoRA sur l'attention du U-Net
    unet.add_adapter(LoraConfig(
        r=clora["rang"], lora_alpha=clora["alpha"],
        lora_dropout=clora.get("dropout", 0.0),
        target_modules=clora["modules_cibles"], init_lora_weights="gaussian",
    ))
    unet.enable_gradient_checkpointing()      # moins de VRAM, un peu plus lent

    parametres = [p for p in unet.parameters() if p.requires_grad]
    for p in parametres:
        p.data = p.data.float()               # les poids entraines restent en fp32
    n_entrainables = sum(p.numel() for p in parametres)
    n_total = sum(p.numel() for p in unet.parameters())
    info(f"parametres entrainables : {n_entrainables:,} sur {n_total:,} "
         f"({100 * n_entrainables / n_total:.2f} %)")

    # 3. Donnees
    jeu = JeuZellige(chemin_projet(cfg["donnees"]["dossier"]), tokenizer,
                     cent.get("retournement_horizontal", True))
    chargeur = DataLoader(jeu, batch_size=cent["taille_lot"], shuffle=True,
                          num_workers=2, drop_last=True)

    # 4. Optimiseur : Adam 8 bits si disponible, sinon AdamW classique
    try:
        import bitsandbytes as bnb
        optimiseur = bnb.optim.AdamW8bit(parametres, lr=cent["taux_apprentissage"])
        info("optimiseur : AdamW 8 bits")
    except Exception:
        optimiseur = torch.optim.AdamW(parametres, lr=cent["taux_apprentissage"])
        info("optimiseur : AdamW (bitsandbytes indisponible)")

    from diffusers.optimization import get_scheduler
    planificateur = get_scheduler(
        cent.get("planificateur", "cosine"), optimizer=optimiseur,
        num_warmup_steps=cent.get("echauffement", 0),
        num_training_steps=cent["pas"])

    amplificateur = torch.amp.GradScaler("cuda")
    sortie = assurer_dossier(cfg["sortie"])
    reinitialiser_vram()

    # 5. Boucle d'entrainement
    import time
    debut = time.perf_counter()
    pas, pertes, journal = 0, [], []
    accumulation = cent.get("accumulation", 1)
    unet.train()

    while pas < cent["pas"]:
        for lot in chargeur:
            if pas >= cent["pas"]:
                break
            pixels = lot["pixels"].to(appareil, non_blocking=True)
            ids = lot["ids"].to(appareil, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                contexte = encodeur_texte(ids)[0]

            bruit = torch.randn_like(latents)
            etapes = torch.randint(0, bruiteur.config.num_train_timesteps,
                                   (latents.shape[0],), device=appareil).long()
            bruites = bruiteur.add_noise(latents, bruit, etapes)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                prediction = unet(bruites, etapes, contexte).sample
                perte = F.mse_loss(prediction.float(), bruit.float())

            amplificateur.scale(perte / accumulation).backward()
            pertes.append(perte.item())

            if (len(pertes)) % accumulation == 0:
                amplificateur.unscale_(optimiseur)
                torch.nn.utils.clip_grad_norm_(parametres, 1.0)
                amplificateur.step(optimiseur)
                amplificateur.update()
                planificateur.step()
                optimiseur.zero_grad(set_to_none=True)
                pas += 1

                if pas % 50 == 0:
                    moyenne = sum(pertes[-200:]) / len(pertes[-200:])
                    journal.append({"pas": pas, "perte": round(moyenne, 5),
                                    "taux": planificateur.get_last_lr()[0]})
                    info(f"pas {pas}/{cent['pas']}  perte {moyenne:.4f}  "
                         f"VRAM {vram_pic_go():.2f} Go")

                if pas % cent.get("checkpoint_tous_les", 250) == 0:
                    sauver_lora(unet, Path(sortie) / f"checkpoint-{pas}")
                    info(f"  checkpoint-{pas} sauvegarde")

    duree = time.perf_counter() - debut
    sauver_lora(unet, Path(sortie))

    poids = list(Path(sortie).glob("*.safetensors"))
    rapport = {
        "nom": cfg["nom"],
        "rang": clora["rang"], "alpha": clora["alpha"],
        "pas": cent["pas"], "taux_apprentissage": cent["taux_apprentissage"],
        "parametres_entrainables": n_entrainables,
        "part_entrainable_pct": round(100 * n_entrainables / n_total, 3),
        "duree_s": round(duree, 1),
        "duree_par_pas_s": round(duree / max(cent["pas"], 1), 3),
        "vram_pic_go": vram_pic_go(),
        "taille_lora_mo": taille_fichier_mo(poids[0]) if poids else None,
        "perte_finale": round(sum(pertes[-200:]) / len(pertes[-200:]), 5),
        "courbe": journal,
    }
    ecrire_json(rapport, Path(sortie) / "entrainement.json")

    info(f"termine en {duree / 60:.1f} min, VRAM pic {rapport['vram_pic_go']:.2f} Go")
    info(f"LoRA : {rapport['taille_lora_mo']} Mo -> {sortie}")


if __name__ == "__main__":
    principal()

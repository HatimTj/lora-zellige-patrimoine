"""
Etape 3 et 4 - Entrainement d'une LoRA sur Stable Diffusion 1.5.

Ce que fait concretement un pas d'entrainement :
  1. le VAE compresse l'image en latent (512x512x3 -> 64x64x4)
  2. on y ajoute du bruit, d'une intensite tiree au hasard
  3. le U-Net predit le bruit ajoute, en lisant la legende
  4. on compare sa prediction au vrai bruit et on corrige les matrices LoRA

Tout le modele est GELE. Seules les matrices LoRA recoivent des gradients.

Le script est volontairement defensif : les API de diffusers et de peft ont
bouge d'une version a l'autre, et Colab n'installe pas toujours la meme. Chaque
operation sensible a donc une solution de repli, et un controle verifie qu'elle
a bien fonctionne au lieu de laisser l'erreur apparaitre trois etapes plus loin.

    python src/train_lora.py --config configs/lora_r8.yaml
    python src/train_lora.py --config configs/lora_r8.yaml --pas 20 --sortie runs/test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (assurer_dossier, charger_config, chemin_projet,  # noqa: E402
                    ecrire_json, fixer_graine, info, neutraliser_torchao,
                    reinitialiser_vram, taille_fichier_mo, vram_pic_go)


# --------------------------------------------------------------------------
# Jeu de donnees
# --------------------------------------------------------------------------

class JeuZellige(Dataset):
    """Images 512x512 et leurs legendes, lues depuis metadata.jsonl."""

    def __init__(self, dossier: Path, tokenizer, retournement: bool = True):
        self.dossier, self.retournement, self.tokenizer = dossier, retournement, tokenizer
        fichier = dossier / "metadata.jsonl"
        if not fichier.exists():
            sys.exit(f"ERREUR : {fichier} absent. Lancez `python src/legender.py`.")
        self.entrees = [json.loads(l) for l in
                        fichier.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not self.entrees:
            sys.exit(f"ERREUR : {fichier} est vide.")
        info(f"{len(self.entrees)} images d'entrainement")

    def __len__(self) -> int:
        return len(self.entrees)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        import numpy as np
        entree = self.entrees[index]
        image = Image.open(self.dossier / entree["file_name"]).convert("RGB")

        # Les motifs de zellige sont symetriques : le retournement horizontal
        # donne une variante plausible et double le jeu, precieux avec 29 images.
        if self.retournement and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        tableau = np.asarray(image, dtype="float32") / 127.5 - 1.0      # vers [-1, 1]
        pixels = torch.from_numpy(tableau).permute(2, 0, 1)
        jetons = self.tokenizer(entree["text"], padding="max_length", truncation=True,
                                max_length=self.tokenizer.model_max_length,
                                return_tensors="pt")
        return {"pixels": pixels, "ids": jetons.input_ids[0]}


# --------------------------------------------------------------------------
# Branchement de la LoRA, avec repli
# --------------------------------------------------------------------------

def brancher_lora(unet, clora: Dict[str, Any]) -> List[torch.nn.Parameter]:
    """Ajoute les matrices LoRA au U-Net et renvoie les parametres entrainables."""
    neutraliser_torchao()
    from peft import LoraConfig

    config = LoraConfig(
        r=clora["rang"], lora_alpha=clora["alpha"],
        lora_dropout=clora.get("dropout", 0.0),
        target_modules=clora["modules_cibles"], init_lora_weights="gaussian",
    )

    if hasattr(unet, "add_adapter"):
        unet.add_adapter(config)                 # voie normale, diffusers >= 0.25
    else:
        from peft import inject_adapter_in_model
        inject_adapter_in_model(config, unet)    # repli si la methode n'existe pas
        info("LoRA branchee via peft.inject_adapter_in_model")

    parametres = [p for p in unet.parameters() if p.requires_grad]
    if not parametres:
        sys.exit("ERREUR : aucun parametre entrainable apres le branchement de la "
                 "LoRA. Les modules cibles de la config ne correspondent a rien "
                 "dans ce U-Net.")
    for p in parametres:
        p.data = p.data.float()                  # les poids entraines restent en fp32
    return parametres


def verifier_gradients(unet, vae, encodeur, bruiteur, parametres, appareil) -> bool:
    """Un aller-retour a blanc : les gradients arrivent-ils jusqu'a la LoRA ?

    Le gradient checkpointing economise beaucoup de VRAM, mais quand tout le
    modele est gele il arrive qu'il coupe le graphe et que les matrices LoRA ne
    recoivent plus rien. L'entrainement tourne alors 20 minutes pour rien. On
    verifie donc en une seconde, avant de commencer.
    """
    pixels = torch.zeros(1, 3, 512, 512, device=appareil)
    ids = torch.zeros(1, 77, dtype=torch.long, device=appareil)
    with torch.no_grad():
        latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
        contexte = sortie_texte(encodeur, ids)
    bruit = torch.randn_like(latents)
    etapes = torch.zeros(1, dtype=torch.long, device=appareil)
    bruites = bruiteur.add_noise(latents, bruit, etapes)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        perte = F.mse_loss(unet(bruites, etapes, contexte).sample.float(), bruit.float())
    perte.backward()
    recus = sum(1 for p in parametres if p.grad is not None and p.grad.abs().sum() > 0)
    for p in parametres:
        p.grad = None
    return recus > 0


def sortie_texte(encodeur, ids):
    """Etats caches du texte, quelle que soit la version de transformers."""
    sortie = encodeur(ids)
    etats = getattr(sortie, "last_hidden_state", None)
    return etats if etats is not None else sortie[0]


# --------------------------------------------------------------------------
# Sauvegarde, avec repli
# --------------------------------------------------------------------------

def sauver_lora(unet, dossier: Path) -> Path:
    """Ecrit les seuls poids LoRA, au format lisible par load_lora_weights."""
    from peft.utils import get_peft_model_state_dict

    dossier.mkdir(parents=True, exist_ok=True)
    etat = get_peft_model_state_dict(unet)
    try:
        from diffusers.utils import convert_state_dict_to_diffusers
        etat = convert_state_dict_to_diffusers(etat)
    except Exception as e:
        info(f"conversion diffusers indisponible ({e}), etat peft conserve")

    try:
        from diffusers import StableDiffusionPipeline
        StableDiffusionPipeline.save_lora_weights(
            save_directory=str(dossier), unet_lora_layers=etat, safe_serialization=True)
    except Exception as e:
        info(f"save_lora_weights a echoue ({e}), ecriture directe en safetensors")
        from safetensors.torch import save_file
        save_file({k: v.detach().cpu().contiguous() for k, v in etat.items()},
                  str(dossier / "pytorch_lora_weights.safetensors"))

    poids = list(dossier.glob("*.safetensors"))
    if not poids:
        sys.exit(f"ERREUR : aucun fichier de poids ecrit dans {dossier}")
    return poids[0]


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", required=True)
    parseur.add_argument("--base", default="configs/base.yaml")
    parseur.add_argument("--pas", type=int, default=None,
                         help="surcharge le nombre de pas (test rapide)")
    parseur.add_argument("--sortie", default=None, help="surcharge le dossier de sortie")
    args = parseur.parse_args()

    base = charger_config(args.base)
    with open(chemin_projet(args.config), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    clora, cent = cfg["lora"], cfg["entrainement"]
    if args.pas:
        cent["pas"] = args.pas
        cent["checkpoint_tous_les"] = max(args.pas, 1)      # pas de checkpoint inutile
    dossier_sortie = args.sortie or cfg["sortie"]
    fixer_graine(cent["graine"])

    if not torch.cuda.is_available():
        sys.exit("ERREUR : aucun GPU detecte. Lancez ce script sur Colab ou Kaggle.")
    appareil = "cuda"

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    modele = base["modele_base"]["id"]
    info(f"chargement de {modele}")
    tokenizer = CLIPTokenizer.from_pretrained(modele, subfolder="tokenizer")
    encodeur = CLIPTextModel.from_pretrained(modele, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(modele, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(modele, subfolder="unet")
    bruiteur = DDPMScheduler.from_pretrained(modele, subfolder="scheduler")

    for module in (encodeur, vae, unet):
        module.requires_grad_(False)
        module.to(appareil)
    encodeur.eval()
    vae.eval()

    parametres = brancher_lora(unet, clora)
    n_entrainables = sum(p.numel() for p in parametres)
    n_total = sum(p.numel() for p in unet.parameters())
    info(f"parametres entrainables : {n_entrainables:,} sur {n_total:,} "
         f"({100 * n_entrainables / n_total:.2f} %)")

    # Gradient checkpointing, mais seulement s'il ne casse pas le graphe
    unet.train()
    unet.enable_gradient_checkpointing()
    if verifier_gradients(unet, vae, encodeur, bruiteur, parametres, appareil):
        info("gradient checkpointing actif (gradients verifies)")
    else:
        unet.disable_gradient_checkpointing()
        if not verifier_gradients(unet, vae, encodeur, bruiteur, parametres, appareil):
            sys.exit("ERREUR : les matrices LoRA ne recoivent aucun gradient.")
        info("gradient checkpointing DESACTIVE : il coupait le graphe. "
             "Consommation VRAM plus elevee.")

    jeu = JeuZellige(chemin_projet(cfg["donnees"]["dossier"]), tokenizer,
                     cent.get("retournement_horizontal", True))
    chargeur = DataLoader(jeu, batch_size=cent["taille_lot"], shuffle=True,
                          num_workers=2, drop_last=True)

    try:
        import bitsandbytes as bnb
        optimiseur = bnb.optim.AdamW8bit(parametres, lr=cent["taux_apprentissage"])
        info("optimiseur : AdamW 8 bits")
    except Exception:
        optimiseur = torch.optim.AdamW(parametres, lr=cent["taux_apprentissage"])
        info("optimiseur : AdamW (bitsandbytes indisponible)")

    from diffusers.optimization import get_scheduler
    planificateur = get_scheduler(cent.get("planificateur", "cosine"),
                                  optimizer=optimiseur,
                                  num_warmup_steps=cent.get("echauffement", 0),
                                  num_training_steps=cent["pas"])

    try:
        amplificateur = torch.amp.GradScaler("cuda")
    except TypeError:
        amplificateur = torch.cuda.amp.GradScaler()         # torch plus ancien

    sortie = assurer_dossier(dossier_sortie)
    reinitialiser_vram()
    debut = time.perf_counter()
    pas, pertes, journal, micro = 0, [], [], 0
    accumulation = cent.get("accumulation", 1)

    while pas < cent["pas"]:
        for lot in chargeur:
            if pas >= cent["pas"]:
                break
            pixels = lot["pixels"].to(appareil, non_blocking=True)
            ids = lot["ids"].to(appareil, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(pixels).latent_dist.sample() * vae.config.scaling_factor
                contexte = sortie_texte(encodeur, ids)

            bruit = torch.randn_like(latents)
            etapes = torch.randint(0, bruiteur.config.num_train_timesteps,
                                   (latents.shape[0],), device=appareil).long()
            bruites = bruiteur.add_noise(latents, bruit, etapes)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                perte = F.mse_loss(unet(bruites, etapes, contexte).sample.float(),
                                   bruit.float())

            amplificateur.scale(perte / accumulation).backward()
            pertes.append(perte.item())
            micro += 1

            if micro % accumulation == 0:
                amplificateur.unscale_(optimiseur)
                torch.nn.utils.clip_grad_norm_(parametres, 1.0)
                amplificateur.step(optimiseur)
                amplificateur.update()
                planificateur.step()
                optimiseur.zero_grad(set_to_none=True)
                pas += 1

                if pas % max(cent["pas"] // 20, 1) == 0 or pas == cent["pas"]:
                    moyenne = sum(pertes[-200:]) / len(pertes[-200:])
                    journal.append({"pas": pas, "perte": round(moyenne, 5),
                                    "taux": planificateur.get_last_lr()[0]})
                    info(f"pas {pas}/{cent['pas']}  perte {moyenne:.4f}  "
                         f"VRAM {vram_pic_go():.2f} Go")

                intervalle = cent.get("checkpoint_tous_les", 250)
                if pas % intervalle == 0 and pas < cent["pas"]:
                    sauver_lora(unet, Path(sortie) / f"checkpoint-{pas}")
                    info(f"  checkpoint-{pas} sauvegarde")

    duree = time.perf_counter() - debut
    fichier_poids = sauver_lora(unet, Path(sortie))

    rapport = {
        "nom": cfg["nom"], "rang": clora["rang"], "alpha": clora["alpha"],
        "pas": cent["pas"], "taux_apprentissage": cent["taux_apprentissage"],
        "parametres_entrainables": n_entrainables,
        "part_entrainable_pct": round(100 * n_entrainables / n_total, 3),
        "duree_s": round(duree, 1),
        "duree_par_pas_s": round(duree / max(cent["pas"], 1), 3),
        "vram_pic_go": vram_pic_go(),
        "taille_lora_mo": taille_fichier_mo(fichier_poids),
        "perte_finale": round(sum(pertes[-200:]) / len(pertes[-200:]), 5),
        "courbe": journal,
    }
    ecrire_json(rapport, Path(sortie) / "entrainement.json")
    info(f"termine en {duree / 60:.1f} min, VRAM pic {rapport['vram_pic_go']:.2f} Go")
    info(f"LoRA : {rapport['taille_lora_mo']} Mo -> {fichier_poids}")


if __name__ == "__main__":
    principal()

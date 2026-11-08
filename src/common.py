"""
Socle commun du projet.

Tous les scripts passent par ici pour :
  - lire la configuration YAML (une seule source de verite)
  - fixer les graines aleatoires (reproductibilite)
  - mesurer le temps et la VRAM (reponse a la question Q3 du README)

Rien dans ce fichier ne depend de diffusers ni de torch au moment de l'import :
il reste utilisable sur une machine sans GPU, pour la partie collecte.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

# Racine du projet = dossier parent de src/
RACINE = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

def charger_config(chemin: str | Path = "configs/base.yaml") -> Dict[str, Any]:
    """Lit un fichier YAML de configuration et renvoie un dictionnaire.

    Les chemins relatifs sont resolus depuis la racine du projet, pour que les
    scripts fonctionnent qu'on les lance depuis la racine ou depuis un notebook
    dans notebooks/.
    """
    chemin = Path(chemin)
    if not chemin.is_absolute():
        chemin = RACINE / chemin
    if not chemin.exists():
        raise FileNotFoundError(f"Configuration introuvable : {chemin}")
    with open(chemin, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_chemin_config"] = str(chemin)
    return config


def chemin_projet(*parties: str) -> Path:
    """Construit un chemin absolu depuis la racine du projet."""
    return RACINE.joinpath(*parties)


def assurer_dossier(chemin: str | Path) -> Path:
    """Cree le dossier s'il n'existe pas et renvoie son chemin absolu."""
    p = Path(chemin)
    if not p.is_absolute():
        p = RACINE / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Reproductibilite
# --------------------------------------------------------------------------

def fixer_graine(graine: int, deterministe: bool = False) -> None:
    """Fixe toutes les sources d'alea utilisees par le projet.

    `deterministe=True` force les algorithmes CUDA deterministes : les
    resultats deviennent rigoureusement reproductibles mais l'entrainement
    ralentit. On l'active pour l'evaluation, pas pour l'entrainement.
    """
    random.seed(graine)
    os.environ["PYTHONHASHSEED"] = str(graine)

    try:
        import numpy as np
        np.random.seed(graine)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(graine)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(graine)
        if deterministe:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def appareil() -> str:
    """Renvoie 'cuda', 'mps' ou 'cpu' selon ce qui est disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# --------------------------------------------------------------------------
# Mesure du cout (question Q3)
# --------------------------------------------------------------------------

def vram_pic_go() -> Optional[float]:
    """VRAM maximale allouee depuis la derniere remise a zero, en Go."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except ImportError:
        pass
    return None


def reinitialiser_vram() -> None:
    """Remet le compteur de VRAM pic a zero avant une mesure."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


@contextmanager
def mesurer(nom: str, journal: Optional[Dict[str, Any]] = None) -> Iterator[Dict[str, Any]]:
    """Mesure le temps ecoule et la VRAM pic d'un bloc de code.

        with mesurer("entrainement") as m:
            entrainer(...)
        print(m["duree_s"], m["vram_pic_go"])

    Si `journal` est fourni, les mesures y sont aussi enregistrees sous `nom`,
    ce qui permet de tout verser ensuite dans results/results.csv.
    """
    reinitialiser_vram()
    mesures: Dict[str, Any] = {"nom": nom}
    debut = time.perf_counter()
    try:
        yield mesures
    finally:
        mesures["duree_s"] = round(time.perf_counter() - debut, 3)
        mesures["vram_pic_go"] = (
            round(v, 3) if (v := vram_pic_go()) is not None else None
        )
        if journal is not None:
            journal[nom] = dict(mesures)
        print(
            f"[mesure] {nom} : {mesures['duree_s']} s"
            + (f", VRAM pic {mesures['vram_pic_go']} Go" if mesures["vram_pic_go"] else "")
        )


def taille_fichier_mo(chemin: str | Path) -> float:
    """Taille d'un fichier en Mo, utile pour le poids des adaptateurs LoRA."""
    return round(Path(chemin).stat().st_size / (1024 ** 2), 3)


# --------------------------------------------------------------------------
# Compatibilite des bibliotheques
# --------------------------------------------------------------------------

def neutraliser_torchao() -> bool:
    """Desamorce l'incompatibilite entre peft et le torchao preinstalle sur Colab.

    Pour brancher une LoRA, peft essaie successivement plusieurs "dispatchers"
    selon le type de couche rencontree. L'un d'eux interroge torchao. Quand la
    version installee est plus ancienne que ce que peft exige, cette fonction
    leve une ImportError au lieu de repondre simplement "non disponible", et
    tout le branchement echoue :

        ImportError: Found an incompatible version of torchao.
        Found version 0.10.0, but only versions above 0.16.0 are supported

    Le projet n'utilise pas torchao -- c'est une bibliotheque de quantification
    sans rapport avec notre entrainement. On force donc la reponse a "non
    disponible", ce qui fait passer peft au dispatcher suivant, le bon.

    Renvoie True si une neutralisation a ete necessaire.
    """
    import importlib.util
    if importlib.util.find_spec("torchao") is None:
        return False                       # absent : peft repondra False de lui-meme
    try:
        import peft.import_utils as utils
    except ImportError:
        return False
    try:
        utils.is_torchao_available()
        return False                       # versions compatibles, rien a faire
    except ImportError:
        pass

    utils.is_torchao_available = lambda: False
    # Le dispatcher a importe la fonction dans son propre espace de noms :
    # corriger uniquement peft.import_utils ne suffirait pas.
    try:
        import peft.tuners.lora.torchao as dispatcher
        dispatcher.is_torchao_available = lambda: False
    except ImportError:
        pass
    info("torchao incompatible avec peft : verification neutralisee")
    return True


# --------------------------------------------------------------------------
# Journalisation des runs
# --------------------------------------------------------------------------

def ecrire_json(objet: Any, chemin: str | Path) -> Path:
    """Ecrit un objet en JSON lisible, en creant le dossier au besoin."""
    p = Path(chemin)
    if not p.is_absolute():
        p = RACINE / p
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(objet, f, ensure_ascii=False, indent=2)
    return p


def info(*messages: Any) -> None:
    """Affichage uniforme, prefixe, envoye sur stderr pour ne pas polluer stdout."""
    print("[info]", *messages, file=sys.stderr)


if __name__ == "__main__":
    # Verification rapide du socle : python src/common.py
    cfg = charger_config()
    info("configuration lue :", cfg["_chemin_config"])
    info("concept :", cfg["projet"]["concept"], "| token :", cfg["projet"]["trigger_token"])
    info("modele de base :", cfg["modele_base"]["id"])
    info("appareil detecte :", appareil())
    fixer_graine(cfg["reproductibilite"]["seeds_evaluation"][0])
    with mesurer("test du socle"):
        time.sleep(0.1)
    info("socle operationnel.")

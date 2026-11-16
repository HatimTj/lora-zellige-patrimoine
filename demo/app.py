#!/usr/bin/env python3
"""Etape 6 - demonstration interactive de la LoRA zellige.

Trois choix de conception, et leur raison :

1. UN SEUL pipeline pour la baseline et la LoRA. L'Etape 4 a etabli qu'une
   intensite de 0 reproduit le modele de base au pixel pres -- c'est meme
   devenu un test automatique du notebook 04. On s'en sert ici : le bouton
   « avant / apres » ne charge pas deux modeles, il deplace un curseur.
   Economie : 1,6 Go de VRAM et une minute de chargement.

2. L'image de base est mise en cache. A prompt et graine identiques elle ne
   depend pas de l'intensite, donc on ne la recalcule pas quand l'utilisateur
   joue avec le curseur. Le second appel est deux fois plus rapide.

3. La demo s'audite elle-meme. Chaque image produite est comparee aux 29
   images d'entrainement, et le resultat est place sur l'echelle etablie a
   l'Etape 5 : le seuil de recopie mesure sur le holdout. Montrer un modele
   generatif sans dire s'il recopie serait incomplet ; ici la reponse est
   affichee a chaque generation, y compris quand elle derange.

Usage :

    python demo/app.py                 # local, http://127.0.0.1:7860
    python demo/app.py --partager      # lien public, pour Colab
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

import gradio as gr  # noqa: E402
import yaml  # noqa: E402

from common import charger_config, chemin_projet, fixer_graine, info  # noqa: E402

# Gradio 6 a deplace le theme et la feuille de style du constructeur Blocks
# vers launch(), et a retire show_download_button de gr.Image. Colab installe
# la version du jour : on s'adapte au lieu de parier dessus.
try:
    VERSION_GRADIO = int(gr.__version__.split(".")[0])
except (AttributeError, ValueError):       # version exotique : on vise le neuf
    VERSION_GRADIO = 6

# --------------------------------------------------------------------------
# Constantes de reference, issues des etapes precedentes
# --------------------------------------------------------------------------

# Valeurs de repli si results/surapprentissage.json est absent : elles
# viennent du meme fichier, mesurees le 16/11/2026 sur 8 images de holdout.
SEUIL_DEFAUT = 0.908
MEDIANE_HOLDOUT_DEFAUT = 0.733

ETAT: Dict[str, Any] = {"pipe": None, "train": None, "seuil": SEUIL_DEFAUT,
                        "mediane": MEDIANE_HOLDOUT_DEFAUT, "cache": {}}

PALETTE = {
    "cobalt": "#2D4FA2",
    "turquoise": "#1F8A8C",
    "terre": "#C2603A",
    "safran": "#D9A441",
    "encre": "#1B1A18",
}


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------

def lire_seuils() -> Tuple[float, float]:
    """Relit le seuil de recopie etabli a l'Etape 5."""
    chemin = chemin_projet("results/surapprentissage.json")
    if not chemin.exists():
        info("surapprentissage.json absent : seuils de repli utilises")
        return SEUIL_DEFAUT, MEDIANE_HOLDOUT_DEFAUT
    donnees = json.loads(chemin.read_text(encoding="utf-8"))
    bloc = donnees["seuil_de_recopie"]
    maxima = sorted(d["similarite_max"] for d in bloc["detail"])
    milieu = len(maxima) // 2
    mediane = (maxima[milieu] if len(maxima) % 2
               else (maxima[milieu - 1] + maxima[milieu]) / 2)
    return float(bloc["seuil"]), float(mediane)


def charger_tout(lora: str) -> None:
    """Charge le pipeline, la LoRA, et les vecteurs du jeu d'entrainement."""
    import generate as G
    import metrics as M

    chemin_lora = chemin_projet(lora)
    if not (chemin_lora / "pytorch_lora_weights.safetensors").exists():
        sys.exit(f"ERREUR : poids LoRA introuvables dans {chemin_lora}. "
                 "Decompressez runs_r8.zip a la racine du projet.")

    cfg = charger_config()
    info("chargement du pipeline, patientez une minute...")
    ETAT["pipe"] = G.charger_pipeline(cfg, lora)
    ETAT["appliquer"] = G.appliquer_intensite

    dossier_train = chemin_projet("data/processed/train")
    images = M.lire_images(dossier_train)
    if not images:
        sys.exit(f"ERREUR : {dossier_train} est vide. Lancez "
                 "src/collect_wikimedia.py --reconstruire puis src/prepare_data.py.")
    info(f"vectorisation des {len(images)} images d'entrainement (DINOv2)")
    ETAT["train"] = {"noms": [n for n, _ in images],
                     "images": [im for _, im in images],
                     "vecteurs": M.vecteurs_dino([im for _, im in images])}

    ETAT["seuil"], ETAT["mediane"] = lire_seuils()
    info(f"seuil de recopie : {ETAT['seuil']}  |  mediane holdout : "
         f"{ETAT['mediane']:.3f}")
    info("pret")


def charger_prompts() -> List[Dict[str, str]]:
    with open(chemin_projet("configs/eval_prompts.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)["prompts"]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _une_image(prompt: str, graine: int, etapes: int, guidance: float,
               intensite: float):
    """Produit une image a l'intensite demandee. 0 = modele de base."""
    import torch

    pipe = ETAT["pipe"]
    ETAT["appliquer"](pipe, float(intensite))
    fixer_graine(int(graine))
    generateur = torch.Generator(
        device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(int(graine))
    sortie = pipe(prompt, num_inference_steps=int(etapes),
                  guidance_scale=float(guidance), generator=generateur,
                  height=512, width=512)
    return sortie.images[0]


def auditer(image) -> Tuple[Any, str]:
    """Compare l'image produite aux 29 images d'entrainement."""
    import metrics as M

    vecteur = M.vecteurs_dino([image])
    sim = M.similarite_au_jeu(vecteur, ETAT["train"]["vecteurs"])
    score = sim["maximum"][0]
    indice = sim["indice_plus_proche"][0]
    voisine = ETAT["train"]["images"][indice]
    nom = ETAT["train"]["noms"][indice]
    return voisine, barre_audit(score, nom)


def barre_audit(score: float, nom_voisin: str) -> str:
    """Place le score sur l'echelle etablie a l'Etape 5."""
    seuil, mediane = ETAT["seuil"], ETAT["mediane"]
    pct = max(0.0, min(1.0, score)) * 100
    p_seuil, p_mediane = seuil * 100, mediane * 100

    if score > seuil:
        teinte, verdict = PALETTE["terre"], "au-dessus du seuil : à inspecter à l'œil"
    elif score > mediane:
        teinte, verdict = PALETTE["safran"], "au-dessus de la médiane du holdout"
    else:
        teinte, verdict = PALETTE["turquoise"], "dans la normale"

    return f"""
<div class="zlj-audit">
  <div class="zlj-audit-top">
    <span class="zlj-score" style="color:{teinte}">{score:.3f}</span>
    <span class="zlj-verdict">{verdict}</span>
  </div>
  <div class="zlj-bar">
    <div class="zlj-fill" style="width:{pct:.1f}%;background:{teinte}"></div>
    <div class="zlj-mark" style="left:{p_mediane:.1f}%"></div>
    <div class="zlj-mark zlj-mark-seuil" style="left:{p_seuil:.1f}%"></div>
  </div>
  <div class="zlj-axis">
    <span>0 — aucun rapport</span>
    <span>médiane holdout {mediane:.3f} · seuil {seuil:.3f}</span>
    <span>1</span>
  </div>
  <p class="zlj-note">
    Similarité DINOv2 à l'image d'entraînement la plus proche
    (<code>{nom_voisin}</code>). Le seuil n'est pas choisi : c'est la
    ressemblance maximale entre une vraie photo de zellige jamais vue par le
    modèle et le jeu d'entraînement. En dessous, l'image ne ressemble pas plus
    à l'entraînement que deux carreaux authentiques ne se ressemblent.
  </p>
</div>"""


def generer(prompt: str, ajouter_token: bool, intensite: float, graine: float,
            etapes: float, guidance: float, auditer_actif: bool):
    """Appelee par le bouton. Produit base, LoRA, et l'audit."""
    if ETAT["pipe"] is None:
        raise gr.Error("Le modèle n'est pas chargé.")
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Écrivez un prompt, par exemple « a tile panel, blue and white ».")

    if ajouter_token and "zlj_style" not in prompt:
        prompt_lora = f"zlj_style, {prompt}"
    else:
        prompt_lora = prompt
    # Le modele de base ne connait pas le token : le lui donner ajouterait du
    # bruit dans la comparaison, puisqu'il le decouperait en sous-mots.
    prompt_base = prompt.replace("zlj_style,", "").replace("zlj_style", "").strip(" ,")
    if not prompt_base:
        # L'utilisateur n'a ecrit que le declencheur : le retirer laisserait un
        # prompt vide, et la comparaison n'aurait plus de sens.
        raise gr.Error("Ce prompt ne contient que le déclencheur. Ajoutez ce que "
                       "vous voulez voir, par exemple « a tile panel, blue and white ».")

    graine, etapes = int(graine), int(etapes)
    depart = time.time()

    cle = (prompt_base, graine, etapes, round(float(guidance), 2))
    if cle in ETAT["cache"]:
        image_base = ETAT["cache"][cle]
        recalcul = False
    else:
        image_base = _une_image(prompt_base, graine, etapes, guidance, 0.0)
        ETAT["cache"][cle] = image_base
        recalcul = True
        if len(ETAT["cache"]) > 24:           # on ne garde pas tout en VRAM
            ETAT["cache"].pop(next(iter(ETAT["cache"])))

    image_lora = _une_image(prompt_lora, graine, etapes, guidance, float(intensite))
    duree = time.time() - depart

    if auditer_actif:
        voisine, audit = auditer(image_lora)
    else:
        voisine, audit = None, ("<div class='zlj-audit'><p class='zlj-note'>"
                                "Audit désactivé.</p></div>")

    detail = (f"<div class='zlj-meta'><b>{duree:.1f} s</b> · graine {graine} · "
              f"intensité {intensite:.2f} · {etapes} étapes · "
              f"{'base recalculée' if recalcul else 'base reprise du cache'}"
              f"<br><code>{prompt_lora}</code></div>")

    return image_base, image_lora, audit, voisine, detail


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------

CSS = """
.gradio-container { max-width: 1180px !important; }

#zlj-entete {
  border-radius: 14px;
  padding: 22px 26px;
  margin-bottom: 6px;
  color: #FBF7F0;
  background:
    repeating-linear-gradient(45deg, rgba(255,255,255,.055) 0 10px,
                              transparent 10px 20px),
    repeating-linear-gradient(-45deg, rgba(255,255,255,.055) 0 10px,
                              transparent 10px 20px),
    linear-gradient(118deg, #2D4FA2 0%, #1F8A8C 58%, #1B6E72 100%);
}
#zlj-entete h1 { margin: 0 0 6px 0; font-size: 1.5rem; letter-spacing: .2px; }
#zlj-entete p  { margin: 0; opacity: .93; font-size: .93rem; line-height: 1.5; }
#zlj-entete code {
  background: rgba(255,255,255,.17); padding: 1px 7px;
  border-radius: 5px; font-weight: 600;
}

.zlj-audit { padding: 4px 2px 2px 2px; }
.zlj-audit-top { display: flex; align-items: baseline; gap: 12px; margin-bottom: 9px; }
.zlj-score { font-size: 1.85rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.zlj-verdict { font-size: .9rem; opacity: .78; }

.zlj-bar {
  position: relative; height: 13px; border-radius: 7px;
  background: var(--neutral-200, #e6e3de); overflow: hidden;
}
.dark .zlj-bar { background: var(--neutral-700, #3a3a3a); }
.zlj-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 7px; }
.zlj-mark {
  position: absolute; top: -3px; bottom: -3px; width: 2px;
  background: var(--body-text-color, #1B1A18); opacity: .5;
}
.zlj-mark-seuil { width: 3px; opacity: .95; }

.zlj-axis {
  display: flex; justify-content: space-between;
  font-size: .72rem; opacity: .62; margin-top: 5px;
}
.zlj-note { font-size: .79rem; opacity: .72; line-height: 1.55; margin: 9px 0 0 0; }
.zlj-note code { font-size: .74rem; }

.zlj-meta {
  font-size: .78rem; opacity: .72; line-height: 1.6;
  padding: 7px 10px; border-radius: 8px;
  border-left: 3px solid #D9A441;
  background: var(--neutral-100, #f6f4f1);
}
.dark .zlj-meta { background: var(--neutral-800, #27272a); }
.zlj-meta code { font-size: .74rem; }
"""


def theme_zellige():
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.stone,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )


def construire(lora: str) -> gr.Blocks:
    prompts = charger_prompts()

    reglages = {"title": "LoRA zellige", "analytics_enabled": False}
    if VERSION_GRADIO < 6:
        reglages["theme"] = theme_zellige()
        reglages["css"] = CSS

    with gr.Blocks(**reglages) as demo:

        gr.HTML("""
<div id="zlj-entete">
  <h1>Zellige — Stable Diffusion spécialisé par LoRA</h1>
  <p>
    29 photos, 6 Mo de poids, 0,19 % du modèle réentraîné. Le mot déclencheur
    <code>zlj_style</code> appelle le style ; sans lui le modèle reste intact.
    Chaque image produite est confrontée en direct au seuil de recopie mesuré
    sur les images que le modèle n'a jamais vues.
  </p>
</div>""")

        with gr.Row():
            # ---------------- colonne gauche : les reglages ----------------
            with gr.Column(scale=4):
                prompt = gr.Textbox(
                    label="Que voulez-vous voir ?",
                    value="a tile panel, deep blue and white, one large rosette filling the frame",
                    lines=3,
                    info="En anglais : le modèle de base a été entraîné dessus.",
                )
                ajouter_token = gr.Checkbox(
                    value=True, label="Ajouter le déclencheur zlj_style",
                    info="Décochez pour voir ce que la LoRA fait sans son mot-clé.",
                )
                intensite = gr.Slider(
                    0.0, 1.5, value=1.0, step=0.05, label="Intensité de la LoRA",
                    info="0 = modèle d'origine. Le style bascule vers 1,0 : en "
                         "dessous le motif reste un carreau peint, au-dessus il "
                         "se fragmente en tesselles.",
                )
                with gr.Row():
                    graine = gr.Number(value=1234, precision=0, label="Graine",
                                       scale=3)
                    tirer = gr.Button("Tirer au sort", scale=2)

                with gr.Accordion("Réglages avancés", open=False):
                    etapes = gr.Slider(15, 50, value=30, step=1,
                                       label="Étapes de débruitage",
                                       info="30 avec DPM-Solver++ vaut 50 avec PNDM.")
                    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,
                                         label="Guidage (CFG)")
                    auditer_actif = gr.Checkbox(
                        value=True, label="Auditer la recopie",
                        info="Ajoute environ une seconde par image.")

                lancer = gr.Button("Générer", variant="primary", size="lg")

                gr.Markdown("### Les prompts figés de l'évaluation")
                gr.Examples(
                    examples=[[p["texte"].replace("zlj_style, ", ""),
                               p["categorie"] != "hors_domaine"]
                              for p in prompts],
                    inputs=[prompt, ajouter_token],
                    label="Choisis avant tout entraînement : domaine, "
                          "transfert, puis hors domaine.",
                    examples_per_page=10,
                )

            # ---------------- colonne droite : les resultats ----------------
            with gr.Column(scale=6):
                with gr.Row():
                    sortie_base = gr.Image(label="Modèle d'origine", type="pil",
                                           height=360)
                    sortie_lora = gr.Image(label="Avec la LoRA zellige", type="pil",
                                           height=360)
                meta = gr.HTML()

                gr.Markdown("### Cette image est-elle une copie ?")
                with gr.Row():
                    with gr.Column(scale=7):
                        audit = gr.HTML()
                    with gr.Column(scale=3):
                        voisine = gr.Image(label="Image d'entraînement la plus proche",
                                           type="pil", height=190)

        with gr.Accordion("Méthode, licences et limites", open=False):
            gr.Markdown(f"""
**Modèle.** Stable Diffusion 1.5, U-Net gelé. Une LoRA de rang 8 est branchée
sur les quatre projections de l'attention : 1,59 M de paramètres entraînés,
0,19 % du U-Net, 6,1 Mo sur le disque.

**Données.** 37 photographies de zellige issues de Wikimedia Commons, 29 pour
l'entraînement et 8 réservées. Licences CC0, domaine public, CC BY et CC BY-SA
uniquement ; NC et ND écartées, parce qu'entraîner un modèle est une
modification. Chaque image est tracée dans `data/licenses.csv` avec son auteur,
sa licence et son URL.

**Le seuil de recopie ({ETAT['seuil']:.3f}).** Il n'est pas choisi à la main. Les
8 images réservées sont de vraies photos de zellige que le modèle n'a jamais
vues ; leur ressemblance maximale au jeu d'entraînement donne la ressemblance
normale entre deux photos indépendantes du même artisanat. Sur les 28 images
d'évaluation, aucune n'atteint ce seuil.

**Limites.** Le seuil repose sur 8 images : c'est peu. La même rareté de données
interdit le FID, biaisé en dessous de quelques milliers d'images de référence —
d'où CLIPScore, DINOv2, LPIPS et l'inspection visuelle. Enfin, 29 photographies
ne représentent qu'une fraction du répertoire du zellige : ce modèle en imite
une facette, il ne le documente pas.

**Usage.** Outil pédagogique et de création. Il ne remplace pas l'artisan, et
les motifs produits ne sont pas des relevés fidèles de panneaux existants.
""")

        tirer.click(lambda: random.randint(0, 2_147_483_647), outputs=graine)
        entrees = [prompt, ajouter_token, intensite, graine, etapes, guidance,
                   auditer_actif]
        sorties = [sortie_base, sortie_lora, audit, voisine, meta]
        lancer.click(generer, inputs=entrees, outputs=sorties,
                     concurrency_limit=1)
        prompt.submit(generer, inputs=entrees, outputs=sorties,
                      concurrency_limit=1)

    return demo


def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parseur.add_argument("--lora", default="runs/r8")
    parseur.add_argument("--partager", action="store_true",
                         help="lien public, necessaire sur Colab")
    parseur.add_argument("--port", type=int, default=7860)
    parseur.add_argument("--sans-modele", action="store_true",
                         help="construit l'interface sans charger le modele "
                              "(verification de l'interface seule)")
    args = parseur.parse_args()

    if not args.sans_modele:
        charger_tout(args.lora)
    demo = construire(args.lora)

    lancement = {"share": args.partager, "server_port": args.port,
                 "show_error": True}
    if VERSION_GRADIO >= 6:
        lancement["theme"] = theme_zellige()
        lancement["css"] = CSS
    demo.queue(max_size=12).launch(**lancement)


if __name__ == "__main__":
    principal()

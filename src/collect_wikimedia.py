"""
Etape 1.2 - Collecte d'images sur Wikimedia Commons, avec leurs licences.

Le script interroge l'API Commons, filtre les fichiers inutilisables, telecharge
les images retenues dans data/raw/ et ecrit UNE LIGNE PAR FICHIER EXAMINE dans
data/licenses.csv -- y compris les fichiers rejetes, avec le motif du rejet.

C'est volontaire : le sujet demande de conserver auteur, URL et licence de chaque
image, et pouvoir montrer ce qui a ete ecarte et pourquoi est ce qui rend le jeu
de donnees defendable en soutenance.

Usage :
    python src/collect_wikimedia.py                     # collecte standard
    python src/collect_wikimedia.py --limite 20         # s'arreter a 20 images
    python src/collect_wikimedia.py --simulation        # n'ecrit rien, montre le tri
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import assurer_dossier, charger_config, chemin_projet, info  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"

# Wikimedia demande un User-Agent identifiant le client. On donne un nom
# generique : aucune donnee personnelle n'est envoyee.
ENTETES = {
    "User-Agent": "lora-zellige-patrimoine/0.1 (projet pedagogique EMSI)",
}

MIMES_ACCEPTES = {"image/jpeg", "image/png", "image/webp"}

# Pause entre deux appels API : Commons renvoie HTTP 429 si on va trop vite.
PAUSE_API = 1.5


def _contexte_ssl() -> "ssl.SSLContext":
    """Contexte HTTPS fiable sur toutes les plateformes.

    Les installations Python de python.org sur macOS n'utilisent pas le trousseau
    du systeme : sans cela, l'appel echoue avec CERTIFICATE_VERIFY_FAILED. On
    s'appuie sur les certificats de `certifi`, present via requirements.txt.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CONTEXTE_SSL = _contexte_ssl()

COLONNES_CSV = [
    "fichier", "titre_original", "auteur", "url_source",
    "licence", "date_telechargement", "conserve", "raison_rejet",
]


# --------------------------------------------------------------------------
# Appels API
# --------------------------------------------------------------------------

def appeler_api(parametres: Dict[str, str], essais: int = 6) -> Dict[str, Any]:
    """Appelle l'API Commons et renvoie le JSON decode.

    Commons limite le debit : en cas de HTTP 429, on attend de plus en plus
    longtemps (1 s, 2 s, 4 s...) avant de reessayer, au lieu d'abandonner.
    """
    parametres = {**parametres, "format": "json", "formatversion": "2"}
    url = f"{API}?{urllib.parse.urlencode(parametres)}"
    attente = 5.0
    for essai in range(essais):
        requete = urllib.request.Request(url, headers=ENTETES)
        try:
            with urllib.request.urlopen(requete, timeout=30, context=CONTEXTE_SSL) as reponse:
                return json.loads(reponse.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and essai < essais - 1:
                # Commons indique parfois lui-meme combien de temps attendre
                conseil = e.headers.get("Retry-After") if e.headers else None
                pause = float(conseil) if conseil and conseil.isdigit() else attente
                info(f"API saturee (HTTP {e.code}), nouvelle tentative dans {pause:.0f} s "
                     f"(essai {essai + 1}/{essais})")
                time.sleep(pause)
                attente = min(attente * 2, 120)
                continue
            raise
    raise RuntimeError("API Commons injoignable apres plusieurs tentatives")


PROPS_IMAGE = {
    "prop": "imageinfo",
    "iiprop": "url|extmetadata|size|mime",
}


def chercher(requete: str, par_page: int = 50, pages_max: int = 2,
             largeur: int = 1024) -> List[Dict[str, Any]]:
    """Recherche plein texte dans l'espace de noms Fichier (namespace 6)."""
    resultats, suite = [], {}
    for _ in range(pages_max):
        params = {
            "action": "query", "generator": "search",
            "gsrsearch": requete, "gsrnamespace": "6",
            "gsrlimit": str(par_page), "iiurlwidth": str(largeur),
            **PROPS_IMAGE, **suite,
        }
        try:
            data = appeler_api(params)
        except Exception as e:
            info(f"recherche '{requete}' interrompue : {e}")
            return resultats          # on garde ce qui a deja ete trouve
        resultats.extend((data.get("query") or {}).get("pages") or [])
        if "continue" not in data:
            break
        suite = data["continue"]
        time.sleep(PAUSE_API)
    return resultats


def resoudre_categories(terme: str, limite: int = 3) -> List[str]:
    """Trouve les vrais noms de categories Commons correspondant a un terme.

    Les noms de categories sont imprevisibles ('Zellij' ? 'Zellige' ?
    'Tilework in Morocco' ?). Plutot que de les deviner dans la config, on
    interroge l'espace de noms Categorie (14) et on utilise ce qui existe
    reellement.
    """
    try:
        data = appeler_api({
            "action": "query", "list": "search", "srsearch": terme,
            "srnamespace": "14", "srlimit": str(limite),
        })
    except Exception as e:
        info(f"resolution de categorie '{terme}' impossible : {e}")
        return []
    titres = [h["title"] for h in (data.get("query") or {}).get("search", [])]
    time.sleep(PAUSE_API)
    return [t.replace("Category:", "") for t in titres]


def lister_categorie(categorie: str, par_page: int = 100, pages_max: int = 2,
                     largeur: int = 1024) -> List[Dict[str, Any]]:
    """Liste les fichiers d'une categorie Commons."""
    resultats, suite = [], {}
    for _ in range(pages_max):
        params = {
            "action": "query", "generator": "categorymembers",
            "gcmtitle": f"Category:{categorie}", "gcmtype": "file",
            "gcmlimit": str(par_page), "iiurlwidth": str(largeur),
            **PROPS_IMAGE, **suite,
        }
        try:
            data = appeler_api(params)
        except Exception as e:
            info(f"categorie '{categorie}' ignoree : {e}")
            return resultats
        resultats.extend((data.get("query") or {}).get("pages") or [])
        if "continue" not in data:
            break
        suite = data["continue"]
        time.sleep(PAUSE_API)
    return resultats


# --------------------------------------------------------------------------
# Nettoyage des metadonnees
# --------------------------------------------------------------------------

def sans_html(texte: Optional[str]) -> str:
    """L'API renvoie l'auteur en HTML : on extrait le texte lisible."""
    if not texte:
        return "inconnu"
    texte = re.sub(r"<[^>]+>", " ", texte)
    texte = urllib.parse.unquote(texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte or "inconnu"


def champ_meta(info_image: Dict[str, Any], cle: str, defaut: str = "") -> str:
    return ((info_image.get("extmetadata") or {}).get(cle) or {}).get("value", defaut)


# --------------------------------------------------------------------------
# Score de pertinence
# --------------------------------------------------------------------------

# Ce qu'on veut : un panneau de zellige vu de face, qui remplit le cadre.
# Ce qu'on ne veut pas : des panoramas, des scenes de rue, des portraits
# d'artisans. Classer par resolution decroissante faisait remonter exactement
# ces derniers, d'ou ce score.

MOTS_FORTS = re.compile(r"zellig|zellij|zillij", re.I)
MOTS_UTILES = re.compile(r"mosaic|mosaïque|mosaique|tile|tilework|carrelage|"
                         r"azulejo|pattern|motif|ornament", re.I)
# Bonus : ce qui annonce un panneau plat vu de face, notre cible exacte
MOTS_CIBLE = re.compile(r"panel|panneau|wall|mur\b|rosette|star|étoile|etoile|"
                        r"geometric|géométrique|geometrique|detail|close-?up", re.I)
# Malus : d'abord les scenes, ensuite les AUTRES artisanats, qui sont le piege
# principal -- le stuc sculpte et le bois incruste ressemblent au zellige dans
# les resultats de recherche mais n'ont ni ses couleurs ni sa geometrie.
MOTS_A_EVITER = re.compile(r"artisan|craftsm|worker|potter|girl|boy|child|man\b|"
                           r"woman|people|portrait|wedding|seated|family|tourist|"
                           r"panoram|aerial|street|skyline|apartment|hotel|"
                           r"restaurant|market|souk|garden|jardin|park\b|pond|"
                           r"lily|lilies|nénuphar|fountain in|courtyard|cour\b|"
                           r"fragment|broken|shard|tesson|inlaid|incrust|box\b|"
                           r"stucco|stuc\b|plaster|plâtre|platre|carved|sculpt|"
                           r"gate\b|porte de la ville|minaret|skyline", re.I)
# Malus fort : images sans couleur, inutiles pour apprendre une palette
MOTS_MONOCHROME = re.compile(r"black and white|noir et blanc|grayscale|"
                             r"greyscale|sepia|ca\.1[89]\d\d|19[0-2]\d", re.I)


def score_pertinence(page: Dict[str, Any]) -> float:
    """Note un fichier : plus c'est haut, plus c'est un bon candidat."""
    infos = (page.get("imageinfo") or [{}])[0]
    titre = page.get("title", "")
    description = sans_html(champ_meta(infos, "ImageDescription"))[:300]
    texte = f"{titre} {description}"

    score = 0.0

    # Pertinence du sujet
    if MOTS_FORTS.search(texte):
        score += 3.0
    if MOTS_UTILES.search(texte):
        score += 1.0
    if MOTS_CIBLE.search(texte):
        score += 1.5
    if MOTS_A_EVITER.search(texte):
        score -= 3.0
    if MOTS_MONOCHROME.search(texte):
        score -= 4.0

    # Cadrage : un panneau vu de face est proche du carre
    largeur = infos.get("width") or 1
    hauteur = infos.get("height") or 1
    rapport = largeur / hauteur
    if 0.8 <= rapport <= 1.25:
        score += 2.0
    elif 0.6 <= rapport <= 1.6:
        score += 1.0
    elif rapport > 2.2 or rapport < 0.45:
        score -= 2.0          # panorama : inutilisable en 512x512

    # Resolution : un leger bonus, plafonne, pour ne pas dominer le reste
    score += min(largeur, hauteur) / 4000.0

    return score


# --------------------------------------------------------------------------
# Filtrage
# --------------------------------------------------------------------------

def motif_de_rejet(page: Dict[str, Any], cfg_donnees: Dict[str, Any]) -> Optional[str]:
    """Renvoie la raison du rejet, ou None si le fichier est acceptable."""
    infos = (page.get("imageinfo") or [{}])[0]
    if not infos:
        return "metadonnees absentes"

    mime = infos.get("mime", "")
    if mime not in MIMES_ACCEPTES:
        return f"format non utilisable ({mime})"

    largeur = infos.get("width") or 0
    hauteur = infos.get("height") or 0
    taille_min = cfg_donnees.get("taille_min_source", 400)
    if min(largeur, hauteur) < taille_min:
        return f"image trop petite ({largeur}x{hauteur})"

    licence = champ_meta(infos, "LicenseShortName") or champ_meta(infos, "UsageTerms")
    licence = sans_html(licence)
    if licence == "inconnu":
        return "licence absente"

    majuscules = licence.upper()
    for interdite in cfg_donnees.get("licences_interdites", []):
        # On cible les mentions NC / ND dans un code de licence, pas un mot
        if re.search(rf"\b(BY-)?{re.escape(interdite)}\b", majuscules):
            return f"licence non libre ({licence})"

    acceptees = cfg_donnees.get("licences_acceptees", [])
    if acceptees and not any(a.upper() in majuscules for a in acceptees):
        return f"licence hors liste ({licence})"

    return None


# --------------------------------------------------------------------------
# Telechargement
# --------------------------------------------------------------------------

def telecharger(url: str) -> Optional[bytes]:
    """Telecharge une image et renvoie ses octets, ou None en cas d'echec."""
    try:
        requete = urllib.request.Request(url, headers=ENTETES)
        with urllib.request.urlopen(requete, timeout=60, context=CONTEXTE_SSL) as reponse:
            contenu = reponse.read()
        if len(contenu) < 5000:          # fichier suspect (page d'erreur, etc.)
            return None
        return contenu
    except Exception as e:
        info(f"telechargement echoue : {e}")
        return None


# Les URL de vignettes Commons portent des parametres de requete
# (?...utm_campaign=...), donc l'extension ne peut pas etre deduite de l'URL.
# On lit la signature des premiers octets, qui ne mentent pas.
SIGNATURES = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
)


def extension_reelle(contenu: bytes) -> Optional[str]:
    """Extension deduite de la signature binaire du fichier."""
    for signature, extension in SIGNATURES:
        if contenu.startswith(signature):
            return extension
    if contenu[:4] == b"RIFF" and contenu[8:12] == b"WEBP":
        return ".webp"
    return None


# --------------------------------------------------------------------------
# Reconstruction a l'identique
# --------------------------------------------------------------------------

def reconstruire(cfg_donnees: Dict[str, Any], dossier_brut: Path,
                 seulement_selection: bool = True) -> int:
    """Re-telecharge exactement les fichiers deja enregistres dans licenses.csv.

    Pourquoi ce mode existe : en mode recherche, le nom zlj_NNN.jpg depend de
    l'ORDRE des resultats et des telechargements reussis. Si une requete echoue
    (API saturee) ou si Commons publie de nouvelles images, la numerotation se
    decale -- et data/selection.txt, qui designe les images par leur numero,
    ne pointe plus sur les bonnes images.

    Ici on part de licenses.csv, qui associe chaque nom de fichier au titre
    Commons d'origine. On interroge l'API par titres exacts, donc le resultat
    est identique a chaque execution, quel que soit l'etat de Commons.
    En prime c'est beaucoup plus rapide : une poignee d'appels au lieu de
    plusieurs dizaines, donc bien moins de risque d'etre limite.
    """
    chemin_csv = chemin_projet(cfg_donnees["fichier_licences"])
    if not chemin_csv.exists():
        info(f"{chemin_csv} absent : impossible de reconstruire")
        return 0

    with open(chemin_csv, encoding="utf-8") as f:
        voulus = {l["fichier"]: l["titre_original"]
                  for l in csv.DictReader(f)
                  if l["conserve"] == "oui" and l["fichier"]}

    chemin_selection = chemin_projet("data/selection.txt")
    if seulement_selection and chemin_selection.exists():
        gardes = {l.split("#")[0].strip()
                  for l in chemin_selection.read_text(encoding="utf-8").splitlines()}
        voulus = {k: v for k, v in voulus.items() if k in gardes}
        info(f"reconstruction limitee a la selection : {len(voulus)} fichiers")
    else:
        info(f"reconstruction de {len(voulus)} fichiers")

    largeur = cfg_donnees.get("largeur_telechargement", 1024)
    par_titre = {f"File:{titre}": nom for nom, titre in voulus.items()}
    titres = sorted(par_titre)
    obtenus = 0

    # L'API accepte jusqu'a 50 titres par appel
    for debut in range(0, len(titres), 50):
        lot = titres[debut:debut + 50]
        data = appeler_api({
            "action": "query", "titles": "|".join(lot),
            "iiurlwidth": str(largeur), **PROPS_IMAGE,
        })
        for page in (data.get("query") or {}).get("pages") or []:
            nom = par_titre.get(page.get("title", ""))
            infos = (page.get("imageinfo") or [{}])[0]
            if not nom or not infos:
                info(f"introuvable sur Commons : {page.get('title')}")
                continue
            destination = dossier_brut / nom
            if destination.exists():
                obtenus += 1
                continue
            contenu = telecharger(infos.get("thumburl") or infos.get("url"))
            if contenu is None:
                continue
            destination.write_bytes(contenu)
            obtenus += 1
            if obtenus % 10 == 0:
                info(f"{obtenus}/{len(voulus)} fichiers")
            time.sleep(0.4)
        time.sleep(PAUSE_API)

    manquants = [n for n in voulus if not (dossier_brut / n).exists()]
    info(f"{obtenus}/{len(voulus)} fichiers reconstruits dans {dossier_brut}")
    if manquants:
        info(f"MANQUANTS ({len(manquants)}) : " + ", ".join(manquants[:10]))
    return obtenus


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def principal() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--config", default="configs/base.yaml")
    parseur.add_argument("--limite", type=int, default=None,
                         help="nombre d'images a conserver (defaut : config)")
    parseur.add_argument("--simulation", action="store_true",
                         help="affiche le tri sans rien telecharger")
    parseur.add_argument("--reconstruire", action="store_true",
                         help="re-telecharge a l'identique les fichiers de "
                              "licenses.csv, sans refaire de recherche")
    args = parseur.parse_args()

    cfg = charger_config(args.config)
    d = cfg["donnees"]
    cible = args.limite or d["n_images_cible"]
    largeur = d.get("largeur_telechargement", 1024)

    dossier_brut = assurer_dossier("data/raw")
    chemin_csv = chemin_projet(d["fichier_licences"])

    if args.reconstruire:
        obtenus = reconstruire(d, dossier_brut)
        if obtenus == 0:
            sys.exit("echec de la reconstruction : aucun fichier obtenu")
        return

    # 1. Rassembler les candidats (recherches + categories), sans doublon de titre
    candidats: Dict[str, Dict[str, Any]] = {}
    for requete in d.get("requetes", []):
        trouves = chercher(requete, largeur=largeur)
        info(f"recherche '{requete}' : {len(trouves)} fichiers")
        for page in trouves:
            candidats.setdefault(page["title"], page)
    categories_reelles: List[str] = []
    for terme in d.get("categories", []):
        categories_reelles.extend(resoudre_categories(terme))
    for categorie in dict.fromkeys(categories_reelles):      # sans doublon, ordre garde
        trouves = lister_categorie(categorie, largeur=largeur)
        info(f"categorie '{categorie}' : {len(trouves)} fichiers")
        for page in trouves:
            candidats.setdefault(page["title"], page)

    info(f"total de candidats uniques : {len(candidats)}")

    # 2. Trier : acceptables d'abord, les plus grandes images en tete
    acceptables, rejetes = [], []
    for titre, page in candidats.items():
        motif = motif_de_rejet(page, d)
        (rejetes if motif else acceptables).append((page, motif))
    acceptables.sort(key=lambda t: -score_pertinence(t[0]))

    info(f"acceptables : {len(acceptables)} | rejetes : {len(rejetes)}")
    if args.simulation:
        print("\n--- 15 premiers acceptables ---")
        for page, _ in acceptables[:15]:
            ii = page["imageinfo"][0]
            print(f"  score {score_pertinence(page):5.2f}  {ii['width']}x{ii['height']:<5}"
                  f"  {sans_html(champ_meta(ii,'LicenseShortName')):<14} {page['title'][:70]}")
        print("\n--- motifs de rejet ---")
        compte: Dict[str, int] = {}
        for _, motif in rejetes:
            cle = re.sub(r"\(.*\)", "(...)", motif or "")
            compte[cle] = compte.get(cle, 0) + 1
        for motif, n in sorted(compte.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {motif}")
        return

    # 3. Telecharger et journaliser
    lignes: List[Dict[str, str]] = []
    empreintes_vues: set[str] = set()
    conserves = 0
    aujourdhui = date.today().isoformat()

    for page, _ in acceptables:
        if conserves >= cible:
            break
        infos = page["imageinfo"][0]
        url = infos.get("thumburl") or infos.get("url")

        contenu = telecharger(url)
        if contenu is None:
            continue

        extension = extension_reelle(contenu)
        if extension is None:
            info(f"format binaire non reconnu, ignore : {page['title'][:60]}")
            continue

        # Doublon strict : le meme fichier remonte souvent via deux requetes
        emp = hashlib.sha256(contenu).hexdigest()
        if emp in empreintes_vues:
            continue
        empreintes_vues.add(emp)

        nom = f"zlj_{conserves + 1:03d}{extension}"
        destination = dossier_brut / nom
        destination.write_bytes(contenu)

        lignes.append({
            "fichier": nom,
            "titre_original": page["title"].replace("File:", ""),
            "auteur": sans_html(champ_meta(infos, "Artist")),
            "url_source": infos.get("descriptionurl", ""),
            "licence": sans_html(champ_meta(infos, "LicenseShortName")
                                 or champ_meta(infos, "UsageTerms")),
            "date_telechargement": aujourdhui,
            "conserve": "oui",
            "raison_rejet": "",
        })
        conserves += 1
        if conserves % 10 == 0:
            info(f"{conserves}/{cible} images telechargees")
        time.sleep(0.4)          # on reste poli avec l'API

    # Les rejetes sont journalises aussi : tracabilite du tri
    for page, motif in rejetes:
        infos = (page.get("imageinfo") or [{}])[0]
        lignes.append({
            "fichier": "",
            "titre_original": page["title"].replace("File:", ""),
            "auteur": sans_html(champ_meta(infos, "Artist")),
            "url_source": infos.get("descriptionurl", ""),
            "licence": sans_html(champ_meta(infos, "LicenseShortName")
                                 or champ_meta(infos, "UsageTerms")),
            "date_telechargement": aujourdhui,
            "conserve": "non",
            "raison_rejet": motif or "",
        })

    with open(chemin_csv, "w", encoding="utf-8", newline="") as f:
        redacteur = csv.DictWriter(f, fieldnames=COLONNES_CSV)
        redacteur.writeheader()
        redacteur.writerows(lignes)

    info(f"{conserves} images dans {dossier_brut}")
    info(f"{len(lignes)} lignes de tracabilite dans {chemin_csv}")
    if conserves < cible:
        info(f"ATTENTION : cible de {cible} non atteinte. "
             "Ajoutez des requetes dans configs/base.yaml, "
             "ou completez avec vos propres photographies.")


if __name__ == "__main__":
    principal()

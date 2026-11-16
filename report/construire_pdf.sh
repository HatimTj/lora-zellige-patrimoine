#!/usr/bin/env bash
# Fabrique les PDF du rapport et de la soutenance a partir de leurs sources
# HTML. On passe par Chrome sans interface plutot que par un convertisseur
# dedie : c'est le moteur qui sert deja a relire les documents a l'ecran, donc
# le PDF est exactement ce qu'on voit.
#
#   bash report/construire_pdf.sh              # les deux
#   bash report/construire_pdf.sh rapport      # seulement le rapport
#   bash report/construire_pdf.sh soutenance   # seulement la soutenance
#
# Le format de page est fixe dans chaque fichier par une regle @page : A4
# portrait pour le rapport, 16:9 pour la soutenance.
#
# Sans Chrome, ou sous Windows : ouvrir le .html dans le navigateur,
# Imprimer, Enregistrer en PDF, marges « par defaut », en cochant
# « Graphiques d'arriere-plan ».

set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
if [ ! -x "$CHROME" ]; then
  echo "Chrome introuvable. Definissez CHROME=/chemin/vers/chrome, ou"
  echo "imprimez le .html depuis votre navigateur."
  exit 1
fi

construire() {
  local nom="$1"
  local src="$PWD/report/$nom.html"
  local out="$PWD/report/$nom.pdf"
  local profil; profil="$(mktemp -d)"

  [ -f "$src" ] || { echo "source absente : $src"; exit 1; }
  rm -f "$out"

  # Chrome ne rend pas toujours la main apres --print-to-pdf : on le lance en
  # arriere-plan, on attend que le fichier cesse de grossir, puis on l'arrete.
  "$CHROME" --headless --disable-gpu --no-sandbox --no-first-run \
    --no-pdf-header-footer --user-data-dir="$profil" \
    --virtual-time-budget=15000 --print-to-pdf="$out" "file://$src" \
    >/dev/null 2>&1 &
  local pid=$!

  for _ in $(seq 1 60); do
    sleep 1
    if [ -s "$out" ]; then
      local t1 t2
      t1=$(wc -c < "$out"); sleep 1; t2=$(wc -c < "$out")
      [ "$t1" = "$t2" ] && break
    fi
  done
  kill $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  rm -rf "$profil"

  [ -s "$out" ] || { echo "echec : $nom"; exit 1; }
  echo "report/$nom.pdf  ($(( $(wc -c < "$out") / 1024 )) Ko)"
}

case "${1:-tout}" in
  rapport)    construire rapport ;;
  soutenance) construire soutenance ;;
  tout)       construire rapport; construire soutenance ;;
  *) echo "usage : $0 [rapport|soutenance|tout]"; exit 1 ;;
esac

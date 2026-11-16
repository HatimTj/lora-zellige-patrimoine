#!/usr/bin/env bash
# Fabrique report/rapport.pdf a partir de report/rapport.html.
#
# On passe par Chrome sans interface plutot que par un convertisseur dedie :
# c'est le moteur qui sert deja a relire le rapport a l'ecran, donc le PDF est
# identique a ce qu'on voit. Chrome ne rend pas toujours la main apres
# --print-to-pdf : on attend que le fichier cesse de grossir, puis on l'arrete.
#
#   bash report/construire_pdf.sh
#
# Sous Windows ou Linux, ou si Chrome est ailleurs :
#   ouvrir report/rapport.html dans le navigateur, Imprimer, Enregistrer en PDF,
#   format A4, marges « par defaut », cocher « Graphiques d'arriere-plan ».

set -euo pipefail
cd "$(dirname "$0")/.."

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
SRC="$PWD/report/rapport.html"
OUT="$PWD/report/rapport.pdf"
PROFIL="$(mktemp -d)"

if [ ! -x "$CHROME" ]; then
  echo "Chrome introuvable. Definissez CHROME=/chemin/vers/chrome, ou"
  echo "imprimez report/rapport.html depuis votre navigateur."
  exit 1
fi

rm -f "$OUT"
"$CHROME" --headless --disable-gpu --no-sandbox --no-first-run \
  --no-pdf-header-footer --user-data-dir="$PROFIL" \
  --virtual-time-budget=15000 --print-to-pdf="$OUT" "file://$SRC" \
  >/dev/null 2>&1 &
PID=$!

for _ in $(seq 1 60); do
  sleep 1
  if [ -s "$OUT" ]; then
    T1=$(wc -c < "$OUT"); sleep 1; T2=$(wc -c < "$OUT")
    [ "$T1" = "$T2" ] && break
  fi
done
kill $PID 2>/dev/null || true
wait $PID 2>/dev/null || true
rm -rf "$PROFIL"

[ -s "$OUT" ] && echo "rapport.pdf : $(wc -c < "$OUT") octets" || { echo "echec"; exit 1; }

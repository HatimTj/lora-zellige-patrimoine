# Reflexion ethique

## 1. Patrimoine culturel et appropriation

Le zellige est un savoir-faire artisanal vivant, transmis par des maalems. Un
modele qui en imite la surface ne reproduit ni la geometrie sous-jacente exacte,
ni la contrainte du materiau. Le projet est presente comme un outil
d'exploration visuelle, jamais comme un substitut au travail artisanal.

Mesure prise : toute image generee est marquee comme synthetique dans la demo.

## 2. Licences des images sources

Chaque image d'entrainement est documentee (auteur, URL, licence) dans
`data/licenses.csv`. Aucune image sans licence identifiee n'est utilisee. Les
oeuvres d'artistes vivants nommes sont exclues, conformement a la consigne.

## 3. Memorisation

Avec seulement ~45 images, une LoRA de rang eleve peut memoriser des exemples.
L'Etape 5 mesure la similarite au plus proche voisin du jeu d'entrainement et
reporte les cas suspects, plutot que de les ecarter.

## 4. Attribution et transparence

Les images generees ne sont pas presentees comme authentiques. La demo affiche
une mention visible, et le rapport precise le modele de base, les donnees et les
seeds, afin que les resultats soient verifiables.

## 5. Empreinte de calcul

(a completer a l'Etape 4 : temps GPU cumule des runs)

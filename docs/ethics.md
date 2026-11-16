# Reflexion ethique

## 1. Patrimoine culturel et appropriation

Le zellige est un savoir-faire artisanal vivant, transmis par des maalems. Un
modele qui en imite la surface ne reproduit ni la geometrie sous-jacente exacte,
ni la contrainte du materiau. Le projet est presente comme un outil
d'exploration visuelle, jamais comme un substitut au travail artisanal.

Mesures prises : la demonstration rappelle en en-tete que les images sont
produites par un modele, et elle affiche a chaque generation la similarite a
l'image d'entrainement la plus proche ainsi que cette image, afin que personne
ne puisse prendre une production pour un releve d'ouvrage existant.

## 2. Licences des images sources

Chaque image d'entrainement est documentee (auteur, URL, licence) dans
`data/licenses.csv`. Aucune image sans licence identifiee n'est utilisee. Les
oeuvres d'artistes vivants nommes sont exclues, conformement a la consigne.

## 3. Memorisation

Avec 29 images, une LoRA peut memoriser des exemples plutot qu'apprendre un
style. Le score de style monterait tout autant, puisqu'une copie ressemble
necessairement au jeu d'entrainement : il fallait donc une mesure separee.

Le seuil au-dela duquel on parle de recopie n'est pas choisi a la main. Les 8
images du holdout sont de vraies photographies de zellige que le modele n'a
jamais vues ; leur ressemblance maximale au jeu d'entrainement donne la
ressemblance normale entre deux photographies independantes du meme artisanat.

| | min | mediane | moyenne | max |
|---|---|---|---|---|
| Holdout, 8 photos jamais vues | 0,457 | 0,733 | 0,722 | **0,908** |
| Images produites, 28 prompts | 0,123 | 0,601 | 0,491 | 0,845 |

**Aucune image produite n'atteint 0,908**, et la distribution produite est plus
basse que celle du holdout. Nous donnons aussi la lecture severe, un maximum sur
8 echantillons etant une barre permissive : a la moyenne du holdout, 3 images
sur 28 depassent. Ces trois-la partagent la composition de leur plus proche
voisin, mais ni la palette ni le detail des tesselles -- la geometrie islamique
reposant sur un repertoire fini de symetries, deux zelliges authentiques et sans
rapport se ressemblent deja beaucoup.

Detail par image : `results/surapprentissage.json`.

## 4. Attribution et transparence

Les images generees ne sont pas presentees comme authentiques. La demo affiche
une mention visible, et le rapport precise le modele de base, les donnees et les
seeds, afin que les resultats soient verifiables.

## 5. Empreinte de calcul

Le projet entier represente environ **2 h 24 de GPU T4** : 88 minutes
d'entrainement cumulees sur cinq configurations et 56 minutes de generation pour
les quinze configurations evaluees. Le detail par entrainement est dans
`runs/*/entrainement.json`, celui des generations dans `results/results.csv`.

C'est l'un des arguments de la LoRA face au reaffinage complet, et il merite
d'etre chiffre plutot qu'affirme. A titre de comparaison, reentrainer les 860
millions de parametres du U-Net sortirait du budget de 12 Go impose par le
sujet.

## 6. Degradation du modele de base

Une adaptation qui deborderait de son token abimerait le modele pour tous les
autres usages. Mesure sur les trois prompts hors domaine, sans declencheur, le
CLIPScore passe de 33,14 a 32,94, soit -0,20 : le modele produit toujours des
chiens, des fraises et des rues sous la pluie normaux. Le pire ecart individuel,
-2,45, correspond a un recadrage et non a une degradation du contenu.

Voir `results/figures/hors_domaine.png`.

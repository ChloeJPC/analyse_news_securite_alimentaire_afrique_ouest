# Analyse de données textuelles sur la sécurité alimentaire en Afrique de l'Ouest
Dépôt de codes et résultats obtenus dans le cadre de mon stage de fin d'étude.

## Résumé
La prédiction des risques liés à la sécurité alimentaire en Afrique de l’Ouest est assurée par des systèmes d’alertes précoces, qui analysent différents types de données, comme des données satellitaires, des estimations de précipitation, des enquêtes ménages ou encore le prix des aliments. Cependant, des incohérences dans ces systèmes existent. C’est pourquoi d’autres types de données accessibles méritent d’être exploitées : les actualités. Dans ce mémoire nous utilisons deux types de documents d’actualité concernant l’Afrique de l’Ouest : des articles de journaux et des transcriptions de vidéos Youtube. Ces documents sont datés, et nous proposons un processus permettant d’identifier leurs thématiques sous-jacentes ainsi que leurs localisations, pour accéder à des informations permettant de lever une incohérence dans une situation d’insécurité alimentaire donnée.

## Corpus utilisés
 Trois corpus sont utilisés :
| Nom du corpus | Type de documents | Couverture temporelle | Portée géographique | Nb de documents |
| :--------------------- | :--------------- | :--------------- | :--------------- | :--------------- |
| Corpus BF | Articles d’actualité | 2009 à 2018 | Burkina Faso | 22856 |
| Corpus YT | Transcriptions automatiques de vidéos d’actualité Youtube | janvier à  mars 2022 | Afrique de l’Ouest | 1109 |
| Corpus PADI | Articles d’actualité & Transcriptions automatiques de vidéos d’actualité Youtube | 2012 à 2022 | Afrique de l’Ouest ; Bénin ; Burkina Faso ; Sénégal | 11638 |

Les corpus utilisés dans cette étude ne sont pas diffusables, ils ne sont donc pas présents dans ce dépôt.

## Dossier Localisation régions
Contient le code effectué pour identifier les régions citées dans le corpus BF et le lexique sur les localités du Burkina.

## Dossier Similarité w2v
Contient le code effectué pour calculer la similarité w2v des documents avec le lexique sur le sécurité alimentaire. Le lexique est présent dans le dossier.

## Dossier Ressources w2v
Contient les ressources nécessaires pour les calculs de similarité sémantique w2v.

## Dossier Topic Modeling
Contient le code réalisé pour effectuer le Topic Modeling sur les corpus. Le dossier devant contenir les corpus est vides car les corpus ne sont pas diffusables.

## Dossier Modèles choisis
Contient les 4 modèles sélectionnés : 
- **modele_BF_25T_seuil53** : modèle sélectionné pour les documents pertinents du corpus BF en fonction du seuil de pertinence à 0.53
- **modele_PADI_20T_seuil43** : modèle sélectionné pour les documents pertinents du corpus PADI en fonction du seuil de pertinence à 0.43
- **modele_PADI_35T_ClassifAuto** : modèle sélectionné pour les documents pertinents du corpus PADI définis par la classification automatique
- **modele_YT_25T_seuil26** : modèle sélectionné pour les documents pertinents du corpus YT en fonction du seuil de pertinence à 0.26

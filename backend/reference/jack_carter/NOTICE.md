# Code d'origine : Jack Carter

Ce dossier contient, sans modification, le code du dépôt
[Artificial-Intelligence-Number-Recognition-Neural-Network](https://github.com/thecarpetjasp/Artificial-Intelligence-Number-Recognition-Neural-Network)
de Jack Carter (compte GitHub `thecarpetjasp`), copié le 10 octobre 2026 depuis le commit
`e1cf26431738f4291b1532d5d05e42e6b6ef4eb8`.

- **Pourquoi il est ici** : le professeur du module DATA732, collègue de l'auteur, a demandé à
  l'équipe de partir de ce dépôt. Le dépôt d'origine ne déclare pas de licence : l'équipe demande
  à l'auteur son accord écrit (ou l'ajout d'une licence) et le note ici dès qu'elle l'a.
- **Ce qui a été repris** : `AI.py` (prédiction), `Trainer.py` (entraînement),
  `kannada/Trainer.py` (variante entraînée sur Kannada-MNIST), `README.md`. Les modèles
  entraînés (JSON de 4 Mo) et les échantillons (archive de 10 Mo) ne sont pas copiés
  ([DONNEES-LOURDES]) : ils restent dans le dépôt d'origine, et `network.model.load_reference_model`
  sait lire le JSON.
- **Ce qu'en fait Vision** : `backend/network/` reprend la même architecture (couche dense
  784 → 128 → 10, ReLU, softmax et entropie croisée, rétropropagation, SGD avec inertie et
  décroissance du pas), réécrite selon la charte du projet : typée, testée, en fonctions courtes,
  entraînée par mini-lots sur MNIST et des chiffres imprimés.
- **Sa propre source** : les noms et la structure (`Layer_Dense`, `Activation_ReLU`,
  `Optimizer_SGD`…) suivent le livre *Neural Networks from Scratch in Python* d'Harrison
  Kinsley et Daniel Kukieła (nnfs.io).

Ce dossier est exclu des contrôles (ruff, mypy, charte) : c'est une archive, on ne le modifie pas.

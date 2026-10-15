# Données

[SOURCES] : chaque jeu de données a ici sa source, sa date de récupération et sa licence.
[DONNEES-LOURDES] : `raw/` et `interim/` sont hors du dépôt ; seul ce qui est petit et
reproductible est commité (`models/`, le modèle entraîné : sa commande est notée ci-dessous).

| Fichier | Source | Récupéré le | Licence |
|---|---|---|---|
| `raw/mnist/*.gz` | Base MNIST de Yann LeCun, Corinna Cortes et Christopher J. C. Burges, par le miroir public de Google (`storage.googleapis.com/cvdf-datasets/mnist`) ; téléchargée au premier entraînement par `pipeline.sources.download_mnist` | au premier lancement | CC BY-SA 3.0 |
| chiffres imprimés | Générés à la volée par `pipeline.printed`, à partir des polices du poste (ou de la police livrée avec Pillow) ; rien n'est stocké | à chaque entraînement | — |
| `models/vision.npz` | Poids du réseau entraîné par `uv run python -m network.train` (graine 0, 10 époques) | 10 octobre 2026 | celle du projet |
| `models/historique.json` | Mesures de cet entraînement, époque par époque | 10 octobre 2026 | celle du projet |
| `models/evaluation.json` | Matrices de confusion du modèle sur le jeu de test (MNIST, imprimés), par `uv run python -m network.evaluate` (aussi écrit par `network.train`) | 15 octobre 2026 | celle du projet |

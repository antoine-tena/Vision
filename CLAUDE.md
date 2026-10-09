# Vision

Projet d'équipe du module DATA732 (visualisation de données) : Antoine, Nathan, Pavel.

## Stack

- **Pas encore décidée.** Langage, bibliothèques de visualisation, gestion des dépendances et
  façon de tester se décident en équipe, puis se décrivent ici avec les commandes pour lancer,
  tester et vérifier le code.

## Langue

- Échanges, documentation et messages de commit en français ; code en anglais.

## Règles

Une règle se cite par son identifiant entre crochets. Une règle impossible à tenir s'amende ici,
avec sa raison ; on ne la contourne pas en silence.

### Code
- **[NOMS-INTERDITS]** Jamais de noms génériques (`data`, `df`, `res`, `tmp`, `x`) : le nom dit ce
  que contient la variable (`ventes_par_mois`, `stations`).
- **[ZERO-MAGIE]** Pas de seuil, de couleur ou de chemin en dur répété : une constante nommée.
- **[REPRODUCTIBLE]** Une figure se régénère par une commande depuis les données sources ;
  aucune retouche à la main d'un fichier généré.

### Données
- **[SOURCES]** Chaque jeu de données a sa source (lien, date de récupération, licence) notée
  dans le dépôt.
- **[DONNEES-LOURDES]** Pas de fichier de données brut volumineux dans git (`data/raw/` est
  ignoré) : on décrit comment le récupérer.
- **[DONNEES-PERSO]** Aucune donnée personnelle réelle ni aucun secret dans le dépôt, les logs ou
  une conversation avec un agent.

### Visualisation
- **[ACCESSIBILITE]** Palettes lisibles par les daltoniens, axes titrés avec leur unité, texte
  alternatif pour toute figure publiée.

### IA
- **[IA-RESPONSABLE]** Le code produit par un agent suit les mêmes règles et la même relecture
  que le code humain ; celui qui fusionne en répond.

## Git & Commits

- **[PAS-SUR-MAIN]** Jamais de commit direct sur `main` : une branche puis une pull request,
  relue par un autre membre de l'équipe.
- **[BRANCHES]** Préfixes `feature/`, `fix/`, `chore/`, `docs/`, `refactor/` ; minuscules, sans
  espaces.
- **[COMMITS]** Petits et atomiques, titre de 50 caractères au plus, en français, préfixé :
  `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
- N'ajoute jamais de mention `Co-Authored-By: Claude` ni `Claude-Session:` dans les messages de
  commit, et n'utilise jamais `--author`.

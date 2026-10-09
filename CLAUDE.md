# Vision — consignes et charte de développement

Projet d'équipe du module DATA732 (visualisation de données) : Antoine, Nathan, Pavel. Ce
fichier est la **charte**, norme unique du code : elle vaut pour tout le monde, humains et
agents. Tout code nouveau la respecte.

## Stack

- **Pas encore décidée.** Langage, bibliothèques de visualisation, gestion des dépendances et
  façon de tester se décident en équipe, puis se décrivent ici, avec les commandes pour lancer,
  tester et vérifier le code, et se branchent dans le pre-commit et la CI.

## Langue

- Échanges, documentation et messages de commit en français ; code en anglais.

---

## 0. Mode d'emploi de la charte

### Identifiants stables
Chaque règle porte un identifiant entre crochets, par exemple **`[TAILLE-FICHIER]`**. Le code,
les commentaires et les PR citent une règle par cet identifiant — `CLAUDE.md [ZERO-MAGIE]` —,
**jamais par un numéro de ligne**. Un identifiant ne change jamais et n'est jamais réutilisé ;
une règle abolie garde le sien, marquée *abrogée*.

### Amender plutôt que contourner
Une règle impossible à tenir s'**amende ici**, explicitement, avec sa raison et sa date — on ne
la contourne pas en silence. Une exception ponctuelle se justifie par un commentaire au point
d'appel, qui cite l'identifiant de la règle.

### Hiérarchie
En cas de conflit : **sécurité et intégrité des données > lisibilité > règles de forme**.

---

## 1. Code

### Taille et découpage
- **[TAILLE-FICHIER]** 300 lignes au plus par fichier. Un module qui dépasse devient un
  dossier ou un paquet, dont le point d'entrée réexporte l'API publique. Exemptés : fichiers
  générés et verrous de dépendances (marqués `linguist-generated` dans `.gitattributes`).
- **[TAILLE-FONCTION]** Une fonction = une responsabilité, **30 lignes au plus** (hors
  docstring). Un test peut aller jusqu'à 40 lignes si le découper cacherait le scénario.
- **[UNE-CLASSE]** Une classe par fichier. Exceptions, qui restent sous [TAILLE-FICHIER] : les
  schémas de données d'un même domaine, les exceptions d'un même domaine, les classes de test
  d'un même fichier de test.
- **[NOTEBOOKS]** Un notebook explore ; le code qui produit une figure livrée vit dans un
  module, que le notebook importe. Pas de fonction de plus de [TAILLE-FONCTION] dans une
  cellule, sorties vidées avant commit.

### Nommage
- **[NOMS-INTERDITS]** Jamais de noms génériques : `data`, `df`, `res`, `val`, `tmp`, `info`,
  `obj`, `item` seul, `result` seul, `x`. Le nom dit *ce que* contient la variable
  (`sales_by_month`, `station_coordinates`). Seule exception : un paramètre imposé par une
  signature de bibliothèque.
- **[VOCABULAIRE]** Le code est en anglais ; les textes affichés (titres, axes, légendes) en
  français.
- **[BOOLEENS]** Un booléen est préfixé : `is`, `has`, `should`, `can` (`is_visible`,
  `has_missing_values`).
- **[CASSE]** Classes en PascalCase ; fonctions et variables selon l'usage du langage
  (snake_case en Python, camelCase en JavaScript) ; constantes en `UPPER_SNAKE_CASE`.

### Valeurs magiques
- **[ZERO-MAGIE]** Pas de littéral métier en dur (seuil, couleur, chemin, nom de colonne
  répété, année de référence) : constante nommée en tête de module ou fichier de constantes
  partagé. Ne sont pas magiques : `0`, `1`, `-1`, `""`, `None`, les index et les littéraux des
  tests.

### Dates, données personnelles, secrets
- **[FUSEAU]** Instants stockés en UTC ; toute notion calendaire (« aujourd'hui », jour,
  mois) se calcule en `Europe/Paris`, jamais en tronquant un instant UTC.
- **[DONNEES-PERSO]** Aucune donnée personnelle réelle (e-mail, nom, téléphone, adresse, IP
  complète) dans le dépôt, les logs, les messages d'erreur ou les figures : on anonymise ou on
  agrège avant.
- **[SECRETS]** Aucun secret dans le code ni dans le dépôt (clé d'API, jeton) : tout passe par
  l'environnement (`.env`, ignoré, avec un `.env.example` sans valeur réelle). Un secret commis
  par erreur est considéré compromis et changé.

### Erreurs et journalisation
- **[EXCEPTIONS]** Jamais d'exception avalée en silence (`except: pass`, `catch {}`) ni de
  capture générique sans la relancer ou la journaliser.
- **[JOURNALISATION]** Pas de `print` / `console.log` laissé dans le code livré : le logger du
  langage, avec des messages paramétrés.

### Entrées
- **[VALIDATION-AUX-FRONTIERES]** Toute donnée qui entre (fichier source, API tierce, paramètre
  d'URL) est validée avant usage : colonnes attendues, types, valeurs manquantes. Un jeu de
  données qui change de forme fait échouer le chargement, pas la figure.

---

## 2. Données et visualisation

- **[SOURCES]** Chaque jeu de données a sa source (lien, date de récupération, licence) notée
  dans le dépôt.
- **[DONNEES-LOURDES]** Pas de fichier de données brut volumineux dans git (`data/raw/` est
  ignoré, 1 Mo au plus par fichier commité) : on décrit ou on scripte sa récupération.
- **[REPRODUCTIBLE]** Une figure se régénère par une commande depuis les données sources ;
  aucune retouche à la main d'un fichier généré.
- **[ACCESSIBILITE]** Palettes lisibles par les daltoniens, axes titrés avec leur unité, texte
  alternatif pour toute figure publiée ; une interface est utilisable au clavier.

---

## 3. Tests et dépendances

- **[TESTS]** Toute fonction de transformation de données a ses tests.
- **[NON-REGRESSION]** Toute correction de bug arrive avec le test qui l'aurait détectée.
- **[RESEAU]** Aucun test n'émet de requête réseau réelle.
- **[DEPENDANCES-FIGEES]** Toute dépendance est résolue par un lockfile commité ; on installe en
  mode figé et on échoue si le lockfile dérive.
- **[NOUVELLE-DEPENDANCE]** Ajouter une dépendance se justifie dans le commit et se décide en
  équipe : maintenue (release de moins d'un an), licence compatible, pas d'équivalent déjà
  présent, pas de nom proche d'un paquet connu (typosquatting).

---

## 4. Développement assisté par IA

- **[IA-RESPONSABLE]** Le code produit par un agent obéit à la même charte et à la même
  relecture que le code humain ; celui qui fusionne en répond.
- **[IA-CONTEXTE]** Ce fichier est le contexte de référence des agents : une règle que l'agent
  enfreint régulièrement est une règle mal écrite, à reformuler ici.
- **[IA-CONFIDENTIALITE]** Aucun secret ni donnée personnelle réelle n'est collé dans une
  conversation avec un agent ; on travaille sur des données de test ou publiques.

---

## 5. Git

- **[PAS-SUR-MAIN]** Jamais de commit direct sur `main` ni sur `develop` : toujours une branche
  dédiée, de préférence dans son propre worktree (`.claude/worktrees/<nom>`), puis une pull
  request relue et approuvée par un autre membre de l'équipe. C'est imposé physiquement : le
  crochet `no-commit-to-branch` du pre-commit refuse le commit en local, et le ruleset GitHub
  « Protection main et develop » refuse tout push direct, force-push ou suppression.
- **[BRANCHES]** Préfixes `feat/`, `fix/`, `refactor/`, `test/`, `chore/`, `docs/` ;
  minuscules, sans espaces, en français (`feat/carte-stations`).
- **[COMMITS]** Petits et atomiques :
  - **10 fichiers au plus** par commit ;
  - titre de **50 caractères au plus**, en français, préfixé par le type, avec une portée
    facultative : `feat:`, `fix:`, `refactor:`, `style:`, `docs:`, `perf:`, `test:`,
    `build:`, `ci:`, `chore:`, `revert:` (`fix(carte): …`) ;
  - jamais de mention `Co-Authored-By: Claude` ni `Claude-Session:` dans le message ;
  - jamais `--author` : l'auteur reste celui de `git config`.
- **[FUSION]** Fusion de la PR sur GitHub en commit de fusion (pas de squash ni de rebase),
  puis suppression de la branche (et du worktree, s'il y en avait un).

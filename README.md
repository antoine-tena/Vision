# Vision

Projet du module DATA732 (visualisation de données).

## Équipe

- Antoine ([@antoine-tena](https://github.com/antoine-tena))
- Nathan ([@NathanBarrachin](https://github.com/NathanBarrachin))
- Pavel ([@Trofymov-Pavlo](https://github.com/Trofymov-Pavlo))

## Organisation

- [`CLAUDE.md`](CLAUDE.md) : la stack et les règles du projet, pour les humains comme pour les
  agents.
- `.vscode/` : réglages et extensions recommandées, partagés par toute l'équipe.

La stack n'est pas encore choisie : elle se décide en équipe, puis se décrit ici (prérequis,
commandes pour lancer et tester) et dans le `CLAUDE.md`.

## Installation

Une fois après le clone, pour activer les garde-fous du commit (dont le refus de commiter
sur `main` ou `develop`) :

```sh
uv tool install pre-commit   # ou : pipx install pre-commit
pre-commit install
```

## Contribuer

1. Créer une branche depuis `main` (`feat/…`, `fix/…`, voir [BRANCHES]).
2. Respecter les règles du [`CLAUDE.md`](CLAUDE.md).
3. Ouvrir une pull request : elle doit être approuvée par un autre membre de l'équipe avant la
   fusion. Personne ne peut pousser directement sur `main` ou `develop`.

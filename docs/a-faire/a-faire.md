# À faire

Tout ce qu'il reste à faire sur Vision, en une seule liste rangée par criticité. Créée le
**2026-10-10**. Ce qui est fait est dans la section « ✅ Fait » en bas ; ce qui a été refusé est
dans [ecarte.md](ecarte.md). **Rien d'autre n'est ouvert.**

La boussole est **Vision en service**.

---

## Mode d'emploi (humains et agents)

**Une entrée** = un titre `### AF-NN · Titre`, puis trois lignes fixes, puis le détail :

    ### AF-NN · Titre court qui dit ce qu'il faut obtenir
    - **État** : à faire
    - **Effort** : S · **Qui** : … · **Échéance** : —
    - **Origine** : identifiant d'origine · audit `docs/_archive/<dossier>/README.md` l. 10-20

    Pourquoi, ce qu'il faut faire (en liste à cocher `- [ ]` s'il y a plusieurs étapes), où regarder.

- **Identifiant** `AF-NN` : stable, jamais réattribué, cité tel quel dans les commits et les
  branches. **Prochain libre : AF-06.**
- **L'état** prend une seule valeur parmi : `à faire` ; `en cours — branche <nom>, depuis le
  JJ mois` ; `bloqué — attend <qui ou quoi>` ; `à vérifier — <où>`.
- **Criticité** = la section où vit l'entrée ; en changer, c'est déplacer l'entrée.
  🔴 critique (maintenant), 🟠 important (avant la boussole), 🟡 normal, ⚪ fond (chantiers
  longs, dette).
- **Effort** : XS (moins d'une heure), S (une demi-journée), M (un à deux jours), L (plus).
- **Terminer une entrée** : la retirer de sa section et ajouter en tête du journal de la
  section « ✅ Fait » une ligne `- **AAAA-MM-JJ** · AF-NN · Titre — ce qui a été fait (branche)`.
- **Détail dans les audits archivés** : quand une entrée cite des lignes de `docs/_archive/`,
  le constat d'origine y est en entier ; il est ancien, à vérifier dans le code actuel.
- **Les emplacements cités sont des pistes, pas des adresses.**
- Vue d'ensemble : `grep -E '^(## |### AF-|- \*\*État)' docs/a-faire/a-faire.md`.

---

## 🔴 Critique

## 🟠 Important

### AF-01 · Vérifier le dessin dans un navigateur
- **État** : à vérifier — `kiln dev`, onglet « Dessiner »
- **Effort** : XS · **Qui** : — · **Échéance** : —
- **Origine** : socle du 10 octobre 2026

La zone de dessin est un composant écrit pour Vision (`frontend/dashboard/canvas.py`,
`st.components.v2`) : les tests vérifient la page et le décodage de l'image, pas le tracé à la
souris. Dessiner quelques chiffres, vérifier la prédiction et le bouton « Effacer ».

### AF-02 · Mieux reconnaître les chiffres dessinés à la souris
- **État** : à faire
- **Effort** : M · **Qui** : — · **Échéance** : —
- **Origine** : socle du 10 octobre 2026

97,6 % sur le jeu de test (MNIST et imprimés), mais un trait à la souris n'est ni une écriture
scannée ni une police. Pistes :
- [ ] constituer un petit jeu de chiffres dessinés dans l'application, pour mesurer l'écart ;
- [ ] augmenter les données d'entraînement (rotations, épaisseurs, décalages) ;
- [ ] essayer une couche cachée plus large ou une deuxième couche.

## 🟡 Normal

### AF-04 · Rapport et soutenance DATA732
- **État** : à faire
- **Effort** : L · **Qui** : — · **Échéance** : —
- **Origine** : module DATA732

Architecture, équations de la propagation et de la rétropropagation, choix des données,
courbes d'apprentissage et matrice de confusion (notebook `backend/notebooks/exploration.py`),
crédits (*Neural Networks from Scratch*, MNIST).

## ⚪ Fond — chantiers longs et dette

### AF-05 · Régularisation et réglages
- **État** : à faire
- **Effort** : M · **Qui** : — · **Échéance** : —
- **Origine** : socle du 10 octobre 2026

L'écart entraînement / validation (99,5 % contre 97,4 %) montre un début de surapprentissage :
L2, dropout ou arrêt précoce, à comparer proprement.

---

## ✅ Fait

### Journal

Une ligne par entrée terminée, la plus récente en tête :
`- **AAAA-MM-JJ** · AF-NN · Titre — ce qui a été fait (branche)`.

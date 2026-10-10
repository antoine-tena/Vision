# Frontend — charte

Section 2 de la charte Vision, chargée quand on travaille dans `frontend/`. Le mode d'emploi
(identifiants, régimes, hiérarchie), les règles communes (§1), les standards 2026 (§4), Git
(§5) et les règles des données et des figures (§6) sont dans le `CLAUDE.md` de la racine.

## 2. Frontend (Streamlit)

Lancer : `uv run streamlit run app.py` (ou `kiln dev`). Vérifier : `kiln check`.

### Code
- **[TYPAGE]** Annotations de type sur toutes les signatures, vérifiées par mypy strict.
- **[PAGES]** `app.py` ne fait qu'appeler une page de `dashboard/` ; une page par module
  (`dashboard/<page>.py`), qui expose une fonction `render_<page>()`. Plusieurs pages passent par
  `st.navigation`.
- **[CALCULS-HORS-UI]** Aucun calcul, aucune transformation de données dans une page : elle
  appelle la chaîne `pipeline/` du backend et affiche le résultat.
- **[CACHE]** Tout chargement de données passe par `@st.cache_data` (avec `ttl` si la source
  change) : rien ne se recharge à chaque interaction.
- **[ETAT]** L'état partagé entre interactions vit dans `st.session_state`, sous des clés
  déclarées en constantes ; jamais de variable globale de module.
- **[ETATS-UI]** Toute zone qui charge des données traite explicitement les trois états :
  chargement (`st.spinner`), erreur (`st.error`, message compréhensible) et jeu vide.
- **[ACCESSIBILITE]** Chaque widget a un libellé explicite (`label`, et `help` s'il le faut),
  jamais `label_visibility="hidden"` sans texte de remplacement ; figures accessibles
  ([FIGURES-ACCESSIBLES]).

### Tests
- **[TESTS-UI]** Chaque page a son test `streamlit.testing.v1.AppTest` : elle se rend sans
  exception et affiche ce qu'elle doit. Les appels extérieurs sont simulés ([RESEAU]).

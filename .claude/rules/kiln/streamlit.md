---
paths:
  - "frontend/**"
---
<!-- Written by kiln and kept current by `kiln update`: do not edit, it would be overwritten. -->
# Frontend: Streamlit

`frontend/app.py`: entry point; `frontend/dashboard/`: pages, layout and constants;
`frontend/tests/`: tests. Run: `kiln dev` (or `uv run streamlit run app.py`); check: `kiln check`.

- **[PAGES]** `app.py` only calls a page of `dashboard/`; one page per module
  (`dashboard/<page>.py`) exposing a `render_<page>()` function. Several pages go through
  `st.navigation`.
- **[COMPUTE-OUTSIDE-UI]** No computation, no data transformation in a page: it calls the `pipeline/` chain of the backend and shows the result.
- **[CACHE]** Every data load goes through `@st.cache_data` (with a `ttl` when the source
  changes): nothing reloads at each interaction.
- **[SESSION-STATE]** State shared between interactions lives in `st.session_state`, under keys
  declared as constants; never a module-level global variable.
- **[UI-STATES]** Every area that loads data handles the three states explicitly: loading
  (`st.spinner`), error (`st.error`, an understandable message) and empty set.
- **[ACCESSIBILITY]** Every widget has an explicit label (`label`, and `help` when needed), never
  `label_visibility="hidden"` without replacement text; accessible figures ([ACCESSIBLE-FIGURES]).
- **[UI-TESTS]** Each page has its `streamlit.testing.v1.AppTest` test: it renders without an
  exception and shows what it must. Outside calls are simulated ([NO-NETWORK]).

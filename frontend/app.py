"""Point d'entrée Streamlit : `uv run streamlit run app.py` (ou `kiln dev`)."""

import streamlit as st

from dashboard.constants import HOME_TITLE, SITE_TITLE, TRAINING_TITLE
from dashboard.home import render_home
from dashboard.training import render_training

st.set_page_config(page_title=SITE_TITLE, layout="wide")
st.navigation(
    [
        st.Page(render_home, title=HOME_TITLE, url_path="reconnaitre", default=True),
        st.Page(render_training, title=TRAINING_TITLE, url_path="entrainement"),
    ]
).run()

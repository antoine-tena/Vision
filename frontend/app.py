"""Point d'entrée Streamlit : `uv run streamlit run app.py` (ou `kiln dev`)."""

import streamlit as st

from dashboard.constants import (
    GLOSSARY_TITLE,
    HOME_TITLE,
    LEARNING_TITLE,
    PROJECT_TITLE,
    SITE_TITLE,
    TRAINING_TITLE,
)
from dashboard.glossary import render_glossary
from dashboard.home import render_home
from dashboard.learning import render_learning
from dashboard.project import render_project
from dashboard.training import render_training

st.set_page_config(page_title=SITE_TITLE, layout="wide")
st.navigation(
    [
        st.Page(render_project, title=PROJECT_TITLE, url_path="projet", default=True),
        st.Page(render_home, title=HOME_TITLE, url_path="reconnaitre"),
        st.Page(render_learning, title=LEARNING_TITLE, url_path="apprendre"),
        st.Page(render_training, title=TRAINING_TITLE, url_path="entrainement"),
        st.Page(render_glossary, title=GLOSSARY_TITLE, url_path="lexique"),
    ]
).run()

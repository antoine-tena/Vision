"""Streamlit entry point: `uv run streamlit run app.py` (or `kiln dev`)."""

import streamlit as st

from dashboard.constants import GLOSSARY_TITLE, PRESENTATION_TITLE, SITE_TITLE
from dashboard.glossary import render_glossary
from dashboard.home import render_home

st.set_page_config(page_title=SITE_TITLE, layout="wide")
st.navigation(
    [
        st.Page(render_home, title=PRESENTATION_TITLE, url_path="presentation", default=True),
        st.Page(render_glossary, title=GLOSSARY_TITLE, url_path="lexique"),
    ]
).run()

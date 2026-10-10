"""Page d'accueil du tableau de bord."""

import polars as pl
import streamlit as st
from pipeline.figures import monthly_totals_figure
from pipeline.sources import load_measurements
from pipeline.transforms import monthly_totals

from dashboard.constants import SITE_DESCRIPTION, SITE_TITLE


# [CACHE] : le chargement ne se rejoue pas à chaque interaction.
@st.cache_data
def load_totals() -> pl.DataFrame:
    return monthly_totals(load_measurements())


def render_home() -> None:
    st.set_page_config(page_title=SITE_TITLE)
    st.title(SITE_TITLE)
    st.write(SITE_DESCRIPTION)
    st.plotly_chart(monthly_totals_figure(load_totals()))

"""Le réseau en direct : tous ses neurones, qui s'allument couche après couche."""

import time

import numpy as np
import streamlit as st
from network.model import DigitClassifier
from pipeline.activity import network_activity, propagation_schedule
from pipeline.figures import network_description, network_figure
from pipeline.recognition import Recognition

from dashboard.constants import LAST_ANIMATED_KEY, LIVE_FRAME_SECONDS


def render_live_network(model: DigitClassifier, recognition: Recognition) -> None:
    activity = network_activity(model, recognition.pixels)
    previous = st.session_state.get(LAST_ANIMATED_KEY)
    is_new_image = previous is None or not np.array_equal(previous, recognition.pixels)
    st.session_state[LAST_ANIMATED_KEY] = recognition.pixels
    # Une image déjà animée s'affiche d'emblée complète (la dernière étape atteint tout).
    schedule = (
        propagation_schedule(activity.layer_sizes)
        if is_new_image
        else np.array([activity.neuron_count])
    )
    placeholder = st.empty()
    for step, reached in enumerate(schedule):
        if step:
            time.sleep(LIVE_FRAME_SECONDS)
        placeholder.plotly_chart(network_figure(activity, int(reached)))
    st.caption(network_description(activity))

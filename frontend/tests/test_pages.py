from streamlit.testing.v1 import AppTest

from dashboard.constants import TRAINING_TITLE


def render_the_training_page() -> None:
    # Script lancé à part par AppTest : ses imports vivent dans la fonction.
    from dashboard.training import render_training

    render_training()


def test_training_page_shows_curves_test_and_patterns() -> None:
    app = AppTest.from_function(render_the_training_page).run()

    assert not app.exception
    assert app.title[0].value == TRAINING_TITLE
    assert [header.value for header in app.subheader] == [
        "Courbes d'apprentissage",
        "Jeu de test (images jamais vues)",
        "Ce que la couche cachée a appris",
    ]
    assert app.radio[0].options == ["manuscrits", "imprimés"]


def test_training_page_switches_test_source() -> None:
    app = AppTest.from_function(render_the_training_page).run()

    app.radio[0].set_value("imprimés").run()

    assert not app.exception
    assert app.metric[-1].label == "Exactitude sur les imprimés"


def render_reasoning_for_a_bar() -> None:
    from network.model import DEFAULT_MODEL_PATH, load_model
    from PIL import Image, ImageDraw
    from pipeline.recognition import recognize

    from dashboard.reasoning import render_steps

    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((50, 15, 50, 85), fill=0, width=10)
    model = load_model(DEFAULT_MODEL_PATH)
    render_steps(model, recognize(model, image))


def test_reasoning_explains_the_recognized_digit_then_another() -> None:
    app = AppTest.from_function(render_reasoning_for_a_bar).run()

    assert not app.exception
    assert [header.value for header in app.header] == [
        "Étape 1 · Ce que voit le réseau",
        "Étape 2 · Couche 1 : les neurones repèrent des motifs",
        "Étape 3 · Couche 2 : les neurones votent pour chaque chiffre",
        "Étape 4 · La softmax change les scores en pourcentages",
        "Résultat",
    ]
    assert app.caption[1].value.startswith(
        f"Carte des pixels décisifs pour le {app.selectbox[0].value}"
    )

    app.selectbox[0].set_value(8).run()

    assert app.caption[1].value.startswith("Carte des pixels décisifs pour le 8")


def render_home_on_examples() -> None:
    import streamlit as st

    from dashboard.constants import EXAMPLES_TAB, INPUT_TABS_KEY
    from dashboard.home import render_home

    st.session_state.setdefault(INPUT_TABS_KEY, EXAMPLES_TAB)
    render_home()


def test_examples_tab_runs_a_training_digit_through_the_live_network() -> None:
    app = AppTest.from_function(render_home_on_examples).run(timeout=20)

    assert not app.exception
    assert app.radio[0].options == ["manuscrits", "imprimés"]
    assert app.metric[0].label == "Chiffre reconnu"
    assert any(caption.value.startswith("Le réseau entier") for caption in app.caption)

    app.selectbox[0].set_value(7).run(timeout=20)

    assert app.metric[0].value == "7"

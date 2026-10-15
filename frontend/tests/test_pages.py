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
        "Données utilisées",
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


def render_every_slide_for_a_bar() -> None:
    from network.model import DEFAULT_MODEL_PATH, load_model
    from PIL import Image, ImageDraw
    from pipeline.recognition import recognize

    from dashboard.slides import SLIDES

    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((50, 15, 50, 85), fill=0, width=10)
    model = load_model(DEFAULT_MODEL_PATH)
    recognition = recognize(model, image)
    for _, render in SLIDES:
        render(model, recognition)


def test_every_slide_renders_and_follows_the_chosen_digit() -> None:
    app = AppTest.from_function(render_every_slide_for_a_bar).run(timeout=20)

    assert not app.exception
    assert any("l'emporte à" in text.value for text in app.markdown)
    assert app.metric[0].label == "Chiffre reconnu"
    digit = app.selectbox[1].value
    assert any(text.value.startswith(f"### Pixels pour le {digit}") for text in app.markdown)

    app.selectbox[1].set_value(8).run(timeout=20)

    assert any(text.value.startswith("### Pixels pour le 8") for text in app.markdown)


def render_home_on_examples() -> None:
    import streamlit as st

    from dashboard.constants import EXAMPLES_TAB, INPUT_TABS_KEY
    from dashboard.home import render_home

    st.session_state.setdefault(INPUT_TABS_KEY, EXAMPLES_TAB)
    render_home()


def test_examples_slide_then_next_slide_shows_the_live_network() -> None:
    app = AppTest.from_function(render_home_on_examples).run(timeout=20)

    assert not app.exception
    assert "Chiffre prêt" in app.success[0].value

    app.button[1].click().run(timeout=20)

    assert not app.exception
    assert app.subheader[0].value.startswith("Étape 1")


def render_the_new_pages() -> None:
    from dashboard.glossary import render_glossary
    from dashboard.learning import render_learning
    from dashboard.project import render_project

    render_project()
    render_learning()
    render_glossary()


def test_project_learning_and_glossary_pages_render() -> None:
    app = AppTest.from_function(render_the_new_pages).run(timeout=20)

    assert not app.exception
    assert [title.value for title in app.title] == ["Le projet", "Comment il apprend", "Lexique"]
    assert app.metric[0].label == "Paramètres (poids + biais)"
    assert app.metric[0].value == "101 770"

from streamlit.testing.v1 import AppTest


def render_every_slide_but_the_choice_for_a_bar() -> None:
    # Script lancé à part par AppTest : ses imports vivent dans la fonction.
    from network.model import DEFAULT_MODEL_PATH, load_model
    from PIL import Image, ImageDraw
    from pipeline.recognition import recognize

    from dashboard.slides import CHOICE_INDEX, SLIDES

    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line((50, 15, 50, 85), fill=0, width=10)
    model = load_model(DEFAULT_MODEL_PATH)
    recognition = recognize(model, image)
    for index, (_, render) in enumerate(SLIDES):
        if index != CHOICE_INDEX:
            render(model, recognition)


def test_every_slide_renders_and_follows_the_chosen_digit() -> None:
    app = AppTest.from_function(render_every_slide_but_the_choice_for_a_bar).run(timeout=30)

    assert not app.exception
    assert app.metric[0].label == "Paramètres"
    assert app.metric[0].value == "101 770"
    assert any("l'emporte à" in text.value for text in app.markdown)
    digit = app.selectbox[1].value
    assert any(f"Pixels pour le {digit} :" in text.value for text in app.markdown)

    app.selectbox[1].set_value(8).run(timeout=30)

    assert any("Pixels pour le 8 :" in text.value for text in app.markdown)


def test_test_slide_switches_source() -> None:
    app = AppTest.from_function(render_every_slide_but_the_choice_for_a_bar).run(timeout=30)

    app.radio[0].set_value("imprimés").run(timeout=30)

    assert not app.exception
    assert any("jamais vues" in text.value for text in app.markdown)


def render_home_on_examples() -> None:
    import streamlit as st

    from dashboard.constants import EXAMPLES_TAB, INPUT_TABS_KEY, SLIDE_KEY
    from dashboard.home import render_home
    from dashboard.slides import CHOICE_INDEX

    st.session_state.setdefault(INPUT_TABS_KEY, EXAMPLES_TAB)
    st.session_state.setdefault(SLIDE_KEY, CHOICE_INDEX)
    render_home()


def test_examples_slide_then_next_slide_starts_step_one() -> None:
    app = AppTest.from_function(render_home_on_examples).run(timeout=20)

    assert not app.exception
    assert "Chiffre prêt" in app.success[0].value

    app.button[1].click().run(timeout=20)

    assert not app.exception
    assert app.subheader[0].value == "3.2"


def render_the_glossary() -> None:
    from dashboard.glossary import render_glossary

    render_glossary()


def test_glossary_page_renders() -> None:
    app = AppTest.from_function(render_the_glossary).run()

    assert not app.exception
    assert app.title[0].value == "Lexique"

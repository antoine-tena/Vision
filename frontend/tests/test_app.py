from streamlit.testing.v1 import AppTest

from dashboard.constants import SITE_TITLE

# Relatif à ce fichier de test, comme l'attend AppTest.from_file.
APP_SCRIPT = "../app.py"


def test_home_renders_title() -> None:
    app = AppTest.from_file(APP_SCRIPT).run()

    assert not app.exception
    assert app.title[0].value == SITE_TITLE

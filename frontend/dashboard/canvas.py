"""Zone de dessin : un canevas écrit pour Vision avec `st.components.v2`.

Le composant s'enregistre à l'import : seul le script Streamlit importe ce module.

Le chiffre se trace en blanc sur fond noir, comme dans MNIST ; à chaque trait levé, l'image est
renvoyée à Python sous forme d'URL de données PNG ; le bouton efface tout.
"""

import streamlit as st
from PIL import Image

from dashboard.constants import CANVAS_KEY, CANVAS_PIXELS, PEN_WIDTH
from dashboard.images import decode_data_url

HTML = f"""
<div class="vision-canvas">
  <canvas width="{CANVAS_PIXELS}" height="{CANVAS_PIXELS}"
          aria-label="Zone de dessin : tracer un chiffre"></canvas>
  <button type="button">Effacer</button>
</div>
"""

CSS = """
.vision-canvas { display: flex; flex-direction: column; gap: 0.5rem; align-items: flex-start; }
.vision-canvas canvas { border: 2px solid #888; border-radius: 8px; touch-action: none;
                        cursor: crosshair; width: 280px; height: 280px; }
.vision-canvas button { padding: 0.3rem 1rem; border-radius: 6px; border: 1px solid #888;
                        background: transparent; color: inherit; cursor: pointer; }
"""

JS = f"""
export default function (component) {{
  const {{ parentElement, setStateValue }} = component;
  const canvas = parentElement.querySelector("canvas");
  const context = canvas.getContext("2d");
  const reset = () => {{
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
  }};
  if (!canvas.dataset.ready) {{ reset(); canvas.dataset.ready = "1"; }}
  Object.assign(context, {{ strokeStyle: "#fff", lineWidth: {PEN_WIDTH},
                           lineCap: "round", lineJoin: "round" }});
  let isDrawing = false;
  const point = (event) => {{
    const box = canvas.getBoundingClientRect();
    return [(event.clientX - box.left) * canvas.width / box.width,
            (event.clientY - box.top) * canvas.height / box.height];
  }};
  canvas.onpointerdown = (event) => {{
    isDrawing = true;
    canvas.setPointerCapture(event.pointerId);
    context.beginPath();
    context.moveTo(...point(event));
  }};
  canvas.onpointermove = (event) => {{
    if (isDrawing) {{ context.lineTo(...point(event)); context.stroke(); }}
  }};
  canvas.onpointerup = () => {{
    if (!isDrawing) return;
    isDrawing = false;
    setStateValue("image", canvas.toDataURL("image/png"));
  }};
  parentElement.querySelector("button").onclick = () => {{
    reset();
    setStateValue("image", null);
  }};
}}
"""

_canvas = st.components.v2.component("vision_canvas", html=HTML, css=CSS, js=JS)


def draw_digit() -> Image.Image | None:
    """Affiche la zone de dessin ; renvoie le dernier dessin, ou None s'il n'y en a pas."""
    result = _canvas(key=CANVAS_KEY, on_image_change=lambda: None)
    data_url = getattr(result, "image", None)
    return decode_data_url(data_url) if isinstance(data_url, str) else None

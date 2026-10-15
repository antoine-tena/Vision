"""Formule en grand, chaque terme pointé par une grande flèche colorée qui dit ce qu'il est."""

from dataclasses import dataclass

import streamlit as st

ARROW_WIDTH = 28
ARROW_HEIGHT = 56
ARROW_HEAD = 20
ARROW_STROKE = 6


@dataclass(frozen=True)
class Term:
    """Un terme de formule : son symbole (HTML), sa couleur et ce qu'il représente."""

    symbol: str
    color: str
    label: str


type Part = Term | str


def arrow_html(color: str) -> str:
    """Flèche vers le haut en CSS (pointe en triangle, puis tige) : st.html retire le SVG."""
    head = (
        f"width:0;height:0;border-left:{ARROW_WIDTH // 2}px solid transparent;"
        f"border-right:{ARROW_WIDTH // 2}px solid transparent;"
        f"border-bottom:{ARROW_HEAD}px solid {color}"
    )
    shaft = f"width:{ARROW_STROKE}px;height:{ARROW_HEIGHT - ARROW_HEAD}px;background:{color}"
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;margin:0.2rem 0">'
        f'<div style="{head}"></div><div style="{shaft}"></div></div>'
    )


def term_html(term: Term) -> str:
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;min-width:3rem">'
        f'<div style="color:{term.color};font-weight:700">{term.symbol}</div>'
        f"{arrow_html(term.color)}"
        f'<div style="color:{term.color};font-size:1rem;font-family:sans-serif;'
        f'font-weight:600;max-width:8rem;text-align:center;line-height:1.2">'
        f"{term.label}</div></div>"
    )


def operator_html(operator: str) -> str:
    return f'<div style="padding:0 0.3rem">{operator}</div>'


def show_formula(parts: list[Part], size_rem: float = 2.6) -> None:
    """Affiche la formule ; les opérateurs (texte simple) restent sans flèche."""
    body = "".join(
        term_html(part) if isinstance(part, Term) else operator_html(part) for part in parts
    )
    st.html(
        '<div style="display:flex;flex-wrap:nowrap;justify-content:center;align-items:flex-start;'
        f'gap:0.4rem;font-size:{size_rem}rem;font-family:serif;margin:1rem 0">'
        f"{body}</div>"
    )

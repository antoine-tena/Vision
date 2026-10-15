"""Formule en grand, chaque terme pointé par une grande flèche colorée qui dit ce qu'il est."""

from dataclasses import dataclass

import streamlit as st

ARROW_WIDTH = 28
ARROW_HEIGHT = 64
ARROW_HEAD = 20
ARROW_STROKE = 6


@dataclass(frozen=True)
class Term:
    """Un terme de formule : son symbole (HTML), sa couleur et ce qu'il représente."""

    symbol: str
    color: str
    label: str


type Part = Term | str


def arrow_svg(color: str) -> str:
    middle = ARROW_WIDTH / 2
    return (
        f'<svg width="{ARROW_WIDTH}" height="{ARROW_HEIGHT}" aria-hidden="true">'
        f'<line x1="{middle}" y1="{ARROW_HEIGHT}" x2="{middle}" y2="{ARROW_HEAD - 2}" '
        f'stroke="{color}" stroke-width="{ARROW_STROKE}" stroke-linecap="round"/>'
        f'<polygon points="{middle},0 0,{ARROW_HEAD} {ARROW_WIDTH},{ARROW_HEAD}" '
        f'fill="{color}"/></svg>'
    )


def term_html(term: Term) -> str:
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;min-width:3rem">'
        f'<div style="color:{term.color};font-weight:700">{term.symbol}</div>'
        f"{arrow_svg(term.color)}"
        f'<div style="color:{term.color};font-size:1rem;font-family:sans-serif;'
        f'font-weight:600;max-width:10rem;text-align:center;line-height:1.2">'
        f"{term.label}</div></div>"
    )


def operator_html(operator: str) -> str:
    return f'<div style="padding:0 0.3rem">{operator}</div>'


def show_formula(parts: list[Part]) -> None:
    """Affiche la formule ; les opérateurs (texte simple) restent sans flèche."""
    body = "".join(
        term_html(part) if isinstance(part, Term) else operator_html(part) for part in parts
    )
    st.html(
        '<div style="display:flex;flex-wrap:wrap;justify-content:center;align-items:flex-start;'
        'gap:0.4rem;font-size:2.6rem;font-family:serif;margin:1.5rem 0">'
        f"{body}</div>"
    )

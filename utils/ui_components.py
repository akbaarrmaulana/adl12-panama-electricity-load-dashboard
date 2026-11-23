"""
Zero-iframe Soft UI / Neumorphism component library for the Panama Electricity Load dashboard.
All multi-line HTML strings are flattened per line to guarantee CommonMark never renders them as code blocks.
"""

import html
import os
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")

_VALID_TONES = {"blue", "teal", "rose", "amber", "neutral"}


def _flatten_html(raw: str) -> str:
    return "\n".join(line.strip() for line in raw.splitlines() if line.strip())


def inject_styles() -> None:
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str = "", kicker: str = "") -> None:
    k_html = f'<div class="neu-pill-kicker">{html.escape(kicker)}</div>' if kicker else ""
    s_html = f'<p class="neu-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    block = _flatten_html(f"""
    <div class="neu-hero">
      {k_html}
      <h1 class="neu-title">{html.escape(title)}</h1>
      {s_html}
    </div>
    """)
    st.markdown(block, unsafe_allow_html=True)


def render_neu_kpis(cards: list[dict]) -> None:
    items_html = []
    for c in cards:
        tone = c.get("tone", "blue")
        if tone not in _VALID_TONES:
            tone = "blue"
        label_esc = html.escape(str(c["label"]))
        value_esc = html.escape(str(c["value"]))
        badge = c.get("badge")
        hint = c.get("hint")
        badge_html = f'<span class="neu-badge tone-{tone}">{html.escape(str(badge))}</span>' if badge else ""
        hint_html = f'<div class="neu-kpi-hint">{html.escape(str(hint))}</div>' if hint else ""
        items_html.append(
            f'<div class="neu-kpi-card">'
            f'<div class="neu-kpi-top"><span class="neu-kpi-label">{label_esc}</span>{badge_html}</div>'
            f'<div class="neu-kpi-value">{value_esc}</div>'
            f'{hint_html}'
            f'</div>'
        )
    block = _flatten_html(f'<div class="neu-kpi-grid">{"".join(items_html)}</div>')
    st.markdown(block, unsafe_allow_html=True)


def render_insight_box(title: str, text: str, tone: str = "blue") -> None:
    if tone not in _VALID_TONES:
        tone = "blue"
    block = _flatten_html(f"""
    <div class="neu-insight">
      <div class="neu-insight-header">
        <span class="neu-insight-dot dot-{tone}"></span>
        <span class="neu-insight-title tone-{tone}">{html.escape(str(title))}</span>
      </div>
      <div>{html.escape(str(text))}</div>
    </div>
    """)
    st.markdown(block, unsafe_allow_html=True)


def render_story_cards(items: list[dict]) -> None:
    cards_html = []
    for item in items:
        tone = item.get("tone", "blue")
        if tone not in _VALID_TONES:
            tone = "blue"
        step = html.escape(str(item.get("step", "•")))
        title = html.escape(str(item["title"]))
        body = html.escape(str(item["body"]))
        cards_html.append(
            f'<div class="neu-story-card">'
            f'<div class="neu-story-head">'
            f'<span class="neu-step-well tone-{tone}">{step}</span>'
            f'<span class="neu-story-title">{title}</span>'
            f'</div>'
            f'<p class="neu-story-body">{body}</p>'
            f'</div>'
        )
    block = _flatten_html(f'<div class="neu-story-grid">{"".join(cards_html)}</div>')
    st.markdown(block, unsafe_allow_html=True)

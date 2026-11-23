"""
Clean Shadcn-UI wrapper components for the Panama Electricity Load dashboard.
Avoids heavy custom HTML/CSS clutter, colored stripes, sparklines, and animations.
"""

import html
import os
import streamlit as st
import streamlit_shadcn_ui as ui

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSS_PATH = os.path.join(BASE_DIR, "assets", "styles.css")


def inject_styles():
    if os.path.exists(CSS_PATH):
        with open(CSS_PATH, "r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_page_header(title, subtitle=""):
    t_esc = html.escape(title)
    sub_html = f'<p class="page-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(f'<h1 class="page-title">{t_esc}</h1>{sub_html}', unsafe_allow_html=True)


def render_shadcn_kpis(cards, key_prefix="kpi"):
    """
    Renders a clean row of streamlit-shadcn-ui metric cards.
    cards: list of dicts with keys: label, value, description (opt), delta (opt)
    """
    cols = st.columns(len(cards), gap="medium")
    for idx, (col, c) in enumerate(zip(cols, cards)):
        with col:
            ui.metric_card(
                label=str(c["label"]),
                value=str(c["value"]),
                description=c.get("description"),
                delta=c.get("delta"),
                variant="dashboard",
                key=f"{key_prefix}_{idx}",
            )


def render_shadcn_alert(title, description, destructive=False, key="alert_main"):
    ui.alert(
        title=title,
        description=description,
        variant="destructive" if destructive else "default",
        key=key,
    )


def render_methodology_drawer(title, body_md):
    with st.expander(title, expanded=False):
        st.markdown(body_md)

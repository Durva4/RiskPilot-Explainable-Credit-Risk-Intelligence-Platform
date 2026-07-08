"""
Theme management for RiskPilot.
Streamlit has no native runtime theme switching, so this injects custom
CSS based on st.session_state['theme'], toggled from the sidebar.
"""
import streamlit as st

DARK_THEME = {
    "bg": "#0E1117",
    "card_bg": "#1A1F2B",
    "sidebar_bg": "#141821",
    "text": "#E8EAED",
    "muted_text": "#9CA3AF",
    "accent": "#1E56A0",        # Solid corporate blue matching your screenshot
    "accent_soft": "#1A2332",
    "border": "#2A2F3A",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}

LIGHT_THEME = {
    "bg": "#FFFFFF",
    "card_bg": "#F8F9FB",
    "sidebar_bg": "#F1F3F6",
    "text": "#1A1F2B",
    "muted_text": "#6B7280",
    "accent": "#2563EB",
    "accent_soft": "#DBEAFE",
    "border": "#E5E7EB",
    "success": "#16A34A",
    "warning": "#D97706",
    "danger": "#DC2626",
}

def init_theme():
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"

def get_theme():
    init_theme()
    return DARK_THEME if st.session_state["theme"] == "dark" else LIGHT_THEME

def toggle_theme():
    st.session_state["theme"] = "light" if st.session_state["theme"] == "dark" else "dark"

def inject_css():
    t = get_theme()
    st.markdown(f"""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
    
    <style>
    .stApp {{
        background-color: {t['bg']};
        color: {t['text']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {t['sidebar_bg']} !important;
        border-right: 1px solid {t['border']} !important;
    }}
    .rp-card {{
        background-color: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .rp-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.14);
    }}
    .rp-kpi-label {{
        color: {t['muted_text']};
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }}
    .rp-kpi-value {{
        color: {t['text']};
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
    }}
    .rp-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .rp-badge-success {{ background-color: {t['success']}22; color: {t['success']}; border: 1px solid {t['success']}55; }}
    .rp-badge-warning {{ background-color: {t['warning']}22; color: {t['warning']}; border: 1px solid {t['warning']}55; }}
    .rp-badge-danger  {{ background-color: {t['danger']}22;  color: {t['danger']};  border: 1px solid {t['danger']}55; }}
    
    .rp-section-title {{ font-size: 22px; font-weight: 700; color: {t['text']}; margin-bottom: 2px; }}
    .rp-section-sub {{ color: {t['muted_text']}; font-size: 14px; margin-bottom: 18px; }}
    
    div[data-testid="stMetric"] {{
        background-color: {t['card_bg']};
        border: 1px solid {t['border']};
        border-radius: 12px;
        padding: 14px 18px;
    }}
    
    /* =========================================================================
       BUG FIXED: HIGH-SPECIFICITY NAVIGATION PATTERN WITH INLINE PSEUDO-ICONS
       ========================================================================= */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{
        display: none !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{
        gap: 6px !important;
        padding: 8px 0 !important;
    }}
    
    /* Container Row Setup */
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        padding: 12px 16px !important;
        border-radius: 6px !important; 
        width: 100% !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        background-color: transparent !important;
        border: none !important;
        cursor: pointer !important;
    }}
    
    /* Hide native radio circles safely */
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }}
    
    /* Exact targeting of Streamlit's internal markdown text blocks */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {{
        font-size: 14px !important;
        color: {t['muted_text']} !important;
        font-weight: 500 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        overflow: visible !important;
    }}
    
    /* Hover Row Animation */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: rgba(255, 255, 255, 0.05) !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover div[data-testid="stMarkdownContainer"] p {{
        color: {t['text']} !important;
    }}
    
    /* ACTIVE FILLED BLOCK STATE MATCHING THE DESIGN EXPLICITLY */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        background-color: {t['accent']} !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }}

    /* FIXED: TARGET INTERNAL CONTAINER PARAGRAPH FOR BOOTSTRAP ICON INJECTION */
    section[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p::before {{
        font-family: "bootstrap-icons" !important;
        display: inline-block !important;
        margin-right: 12px !important;
        font-size: 16px !important;
        font-weight: normal !important;
        color: #3B82F6 !important; /* Vibrant blue icons for inactive elements */
    }}
    
    /* Map accurate list-child tags to specific asset unicode characters */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F58F"; }} /* Dashboard */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F4E0"; }} /* Customer Assessment */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F73B"; }} /* AI Risk Assistant */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F3E5"; }} /* Model Analytics */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F46A"; }} /* Business Insights */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(6) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F30B"; }} /* Explainability */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(7) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F374"; }} /* Reports */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(8) div[data-testid="stMarkdownContainer"] p::before {{ content: "\\F441"; }} /* About */

    /* Active item icon shifts cleanly to solid white contrast */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p::before {{
        color: #FFFFFF !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def kpi_card(label, value, badge_text=None, badge_type="success"):
    t = get_theme()
    badge_html = ""
    if badge_text:
        badge_html = f'<span class="rp-badge rp-badge-{badge_type}">{badge_text}</span>'
    st.markdown(f"""
    <div class="rp-card">
        <div class="rp-kpi-label">{label}</div>
        <div class="rp-kpi-value">{value}</div>
        <div style="margin-top:8px;">{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)

def risk_badge(tier):
    mapping = {
        "P1": ("Low Risk", "success"),
        "P2": ("Moderate Risk", "success"),
        "P3": ("Elevated Risk", "warning"),
        "P4": ("High Risk", "danger"),
    }
    label, badge_type = mapping.get(tier, (tier, "warning"))
    return f'<span class="rp-badge rp-badge-{badge_type}">{label}</span>'
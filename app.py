"""
RiskPilot — Explainable Credit Risk Intelligence Platform
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from utils.theme import init_theme, get_theme, toggle_theme, inject_css, kpi_card, risk_badge
from utils.shap_utils import get_explainer, explain_prediction, top_factors, plain_english_factor, business_sentence
from utils.ai_assistant import is_ai_available, build_context, call_llm, SUGGESTED_QUESTIONS

st.set_page_config(
    page_title="RiskPilot | Credit Risk Intelligence",
    page_icon="🏦", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

init_theme()
inject_css()
t = get_theme()

MODELS_DIR = "models"

# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(f"{MODELS_DIR}/deployment_model.pkl")
    preprocessor = joblib.load(f"{MODELS_DIR}/deployment_preprocessor.pkl")
    le = joblib.load(f"{MODELS_DIR}/deployment_label_encoder.pkl")
    common_cols = joblib.load(f"{MODELS_DIR}/deployment_columns.pkl")
    return model, preprocessor, le, common_cols

@st.cache_data
def load_perf_files():
    try:
        cm = pd.read_csv(f"{MODELS_DIR}/confusion_matrix_test.csv", index_col=0)
    except FileNotFoundError:
        cm = None
    try:
        report = pd.read_csv(f"{MODELS_DIR}/classification_report_test.csv", index_col=0)
    except FileNotFoundError:
        report = None
    try:
        importance = pd.read_csv(f"{MODELS_DIR}/shap_feature_importance.csv")
    except FileNotFoundError:
        importance = None
    return cm, report, importance

model, preprocessor, le, common_cols = load_artifacts()
cm_df, report_df, importance_df = load_perf_files()

def get_feature_names():
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    return list(num_cols) + cat_names

FEATURE_NAMES = get_feature_names()

def predict_customer(df_row):
    X = df_row[common_cols].copy()
    if "CC_utilization" in X.columns:
        X["CC_utilization"] = X["CC_utilization"].fillna(0)
    if "PL_utilization" in X.columns:
        X["PL_utilization"] = X["PL_utilization"].fillna(0)
    X_t = preprocessor.transform(X)
    pred = model.predict(X_t).ravel()
    proba = model.predict_proba(X_t)
    return X_t, pred, proba

def decision_from_tier(tier, confidence):
    if confidence < 0.55:
        return "Review", "warning"
    if tier == "P1":
        return "Approve", "success"
    if tier == "P4":
        return "Reject" if confidence > 0.75 else "Review", "danger"
    if tier == "P3":
        return "Review", "warning"
    return "Approve", "success"

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION (CLEAN TEXT FOR COMPONENT RENDERING)
# ---------------------------------------------------------------------------
NAV_MAP = {
    "Dashboard": "Dashboard",
    "Customer Assessment": "Customer Assessment",
    "AI Risk Assistant": "AI Risk Assistant",
    "Model Analytics": "Model Analytics",
    "Business Insights": "Business Insights",
    "Explainability": "Explainability",
    "Reports": "Reports",
    "About": "About"
}

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Dashboard"

if st.session_state.get("nav_pending"):
    st.session_state["current_page"] = st.session_state["nav_pending"]
    st.session_state["nav_pending"] = None

radio_options = list(NAV_MAP.keys())
default_radio_idx = radio_options.index(st.session_state["current_page"])

with st.sidebar:
    st.markdown(f"""
        <div style="padding: 16px 14px 12px 14px; margin-bottom: 4px;">
            <div style="font-size:24px; font-weight:800; color:{t['text']}; display: flex; align-items: center; gap: 8px;">🏦 RiskPilot</div>
            <div style="font-size:12px; color:{t['muted_text']}; margin-top: 4px; font-weight: 500; letter-spacing: 0.01em;">Explainable Credit Risk Intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    chosen_label = st.radio(
        "Navigation Menu",
        options=radio_options,
        index=default_radio_idx,
        key="nav_radio"
    )
    
    st.session_state["current_page"] = NAV_MAP[chosen_label]

    st.markdown("<div style='padding-top: 8px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    theme_label = "🌙 Switch to Dark" if st.session_state["theme"] == "light" else "☀️ Switch to Light"
    if st.button(theme_label, use_container_width=True):
        toggle_theme()
        st.rerun()
        
    st.markdown("<div style='margin-top: 12px; margin-bottom: 12px;'></div>", unsafe_allow_html=True)
    if is_ai_available():
        st.success("AI Assistant: Connected", icon="🤖")
    else:
        st.warning("AI Assistant: No API key set", icon="🤖")
        
    st.markdown("<div style='padding-bottom: 40px;'></div>", unsafe_allow_html=True)

def go_to(page_name):
    st.session_state["nav_pending"] = page_name
    st.rerun()

def get_batch_results():
    return st.session_state.get("batch_results")

def get_selected_customer():
    results = get_batch_results()
    if not results:
        return None
    idx = st.session_state.get("selected_customer_idx", 0)
    idx = min(idx, len(results) - 1)
    return results[idx]

def customer_selector_widget():
    results = get_batch_results()
    if not results:
        return
    if len(results) == 1:
        st.caption("Showing the single most recent assessment.")
        return
    labels = [f"Customer {i+1} — {r['tier']} ({r['confidence']:.0%} confidence)" for i, r in enumerate(results)]
    current_idx = st.session_state.get("selected_customer_idx", 0)
    current_idx = min(current_idx, len(results) - 1)
    idx = st.selectbox(
        "Select which customer to inspect:", 
        range(len(labels)),
        format_func=lambda i: labels[i], 
        index=current_idx,
        key="customer_selector_widget"
    )
    st.session_state["selected_customer_idx"] = idx

# ---------------------------------------------------------------------------
# PAGE CONTENT RENDERING ENGINE
# ---------------------------------------------------------------------------
if st.session_state["current_page"] == "Dashboard":
    hero_col, stats_col = st.columns([12, 8], gap="large")
    
    with hero_col:
        st.markdown(f"""
            <div style="padding: 10px 0 10px 0;">
                <h1 style="font-size:44px; font-weight:800; color:{t['text']}; margin-bottom: 4px;">
                    🏦 RiskPilot
                </h1>
                <div style="font-size:20px; color:{t['accent']}; font-weight:600; margin-bottom: 16px;">
                    Explainable Credit Risk Intelligence Engine
                </div>
                <div style="font-size:15px; color:{t['muted_text']}; max-width:560px; line-height:1.6; margin-bottom: 24px;">
                    Classify loan applicants into risk tiers in seconds — and get a 
                    plain-English reason for every decision, not just a black-box score. 
                    Built for credit officers, risk teams, and auditors alike.
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Get Started — Assess a Customer", type="primary"):
            go_to("Customer Assessment")
            
        st.write("")
        st.write("")
        
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown(f"""
                <div class="rp-card">
                    <div style="font-size:26px; margin-bottom: 8px;">⚡</div>
                    <h4>Instant Scoring</h4>
                    <p>Upload a CSV or fill a quick form — no complex fields required.</p>
                </div>
            """, unsafe_allow_html=True)
        with f2:
            st.markdown(f"""
                <div class="rp-card">
                    <div style="font-size:26px; margin-bottom: 8px;">🔍</div>
                    <h4>Plain-English Reasons</h4>
                    <p>Every prediction comes with a human-readable explanation, powered by SHAP.</p>
                </div>
            """, unsafe_allow_html=True)
        with f3:
            st.markdown(f"""
                <div class="rp-card">
                    <div style="font-size:26px; margin-bottom: 8px;">🤖</div>
                    <h4>Ask the AI Assistant</h4>
                    <p>Ask natural-language questions about any prediction, grounded in real data.</p>
                </div>
            """, unsafe_allow_html=True)
            
    with stats_col:
        st.markdown(f"""
            <div style="padding: 10px 0 10px 0;">
                <div style="font-size:11px; font-weight:700; color:{t['accent']}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:20px;">
                    📊 Engine Status & Performance
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div class="rp-card" style="min-height: 70px; margin-bottom: 12px; padding: 16px 20px !important;">
                <div style="font-size: 11px; color: {t['muted_text']}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Core Model Architecture</div>
                <div style="font-size: 19px; font-weight: 700; color: {t['text']}; margin-top: 2px;">CatBoost Classifier</div>
            </div>
            <div class="rp-card" style="min-height: 70px; margin-bottom: 12px; padding: 16px 20px !important;">
                <div style="font-size: 11px; color: {t['muted_text']}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Risk Target Metrics</div>
                <div style="font-size: 19px; font-weight: 700; color: {t['text']}; margin-top: 2px;">4 Tiers <span style="font-size: 14px; font-weight: 500; color: {t['muted_text']};">(P1 – P4)</span></div>
            </div>
            <div class="rp-card" style="min-height: 70px; margin-bottom: 12px; padding: 16px 20px !important;">
                <div style="font-size: 11px; color: {t['muted_text']}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Deployment Accuracy Baseline</div>
                <div style="font-size: 19px; font-weight: 700; color: #10B981; margin-top: 2px;">72.6%</div>
            </div>
            <div class="rp-card" style="min-height: 70px; margin-bottom: 20px; padding: 16px 20px !important;">
                <div style="font-size: 11px; color: {t['muted_text']}; text-transform: uppercase; font-weight: 700; letter-spacing: 0.04em;">Macro F1-Score Performance</div>
                <div style="font-size: 19px; font-weight: 700; color: {t['accent']}; margin-top: 2px;">69.1%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.caption("⚠️ Note: The baseline model achieves 99% Macro-F1 when Credit Bureau Scores are attached. "
                   "Since bureau pulls are restricted at the initial entry edge, the production engine runs at an honest "
                   "69.1% deployable performance.")
        
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        
        batch = get_batch_results()
        if batch:
            st.markdown(f"🟢 **Active Session state:** {len(batch)} customer profiles currently evaluated.")
            if st.button("View Current Analytical Logs →", use_container_width=True):
                go_to("Customer Assessment")
        else:
            st.markdown(f"<div style='font-size: 13px; color: {t['muted_text']}; font-style: italic;'>No calculations processed during this deployment instance.</div>", unsafe_allow_html=True)

elif st.session_state["current_page"] == "Customer Assessment":
    st.markdown('<div class="rp-section-title">Customer Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">Assess one customer, or a batch of many — pick the option that matches your data</div>', unsafe_allow_html=True)
    
    from utils.simple_input import SIMPLE_FIELDS, EDUCATION_OPTIONS, load_defaults, expand_simple_row, expand_simple_dataframe, simple_template_csv
    
    defaults = load_defaults()
    
    input_mode = st.radio(
        "Input method",
        ["Quick Assessment — one customer", "Quick Assessment — batch CSV"],
        horizontal=True
    )
    input_df = None
    
    if input_mode == "Quick Assessment — one customer":
        with st.form("quick_entry"):
            c1, c2 = st.columns(2)
            with c1:
                credit_history_years = st.number_input("Years since their first ever loan/credit card", 0.0, 40.0, 3.0, step=0.5)
                recent_apps = st.number_input("Loan/card applications in the last 3 months", 0, 20, 1)
                ontime_payments = st.number_input("On-time payments in the last 12 months", 0, 100, 6)
                missed_payments = st.number_input("Total payments ever missed", 0, 50, 0)
                deliq_severity = st.slider("How serious was their most recent missed payment? (0 = none)", 0, 10, 0)
            with c2:
                income = st.number_input("Monthly income (₹)", 0, 1000000, 30000, step=1000)
                education = st.selectbox("Education", EDUCATION_OPTIONS)
                months_employed = st.number_input("Months at current job", 0, 500, 24)
                has_cc = st.selectbox("Has a credit card?", ["No", "Yes"])
                has_pl = st.selectbox("Has a personal loan?", ["No", "Yes"])
            submitted = st.form_submit_button("Assess Risk", use_container_width=True)
            
        if submitted:
            simple_values = {
                "credit_history_years": credit_history_years,
                "recent_loan_applications_3m": recent_apps,
                "ontime_payments_12m": ontime_payments,
                "missed_payments_total": missed_payments,
                "monthly_income": income,
                "education": education,
                "months_at_current_job": months_employed,
                "has_credit_card": has_cc,
                "has_personal_loan": has_pl,
                "recent_missed_payment_severity": deliq_severity,
            }
            full_row = expand_simple_row(simple_values, defaults, common_cols)
            input_df = pd.DataFrame([full_row])
            
    else:
        st.markdown("**CSV format:** one row per customer, using these column names:")
        st.code(", ".join(simple_template_csv()))
        template_df = pd.DataFrame([{f[0]: "" for f in SIMPLE_FIELDS}])
        st.download_button("Download a blank template CSV", template_df.to_csv(index=False), file_name="quick_assessment_template.csv", use_container_width=True)
        
        uploaded = st.file_uploader("Upload your simplified CSV", type=["csv"], key="quick_csv")
        if uploaded is not None:
            simple_df = pd.read_csv(uploaded)
            missing_simple = set(f[0] for f in SIMPLE_FIELDS) - set(simple_df.columns)
            if missing_simple:
                st.error(f"Your CSV is missing these expected columns: {sorted(missing_simple)}.")
            else:
                input_df = expand_simple_dataframe(simple_df, defaults, common_cols)
                
    if input_df is not None:
        batch_results = []
        explainer = get_explainer(model)
        for i in range(len(input_df)):
            row_df = input_df.iloc[[i]]
            X_t, pred, proba = predict_customer(row_df)
            tier = le.inverse_transform(pred)[0]
            confidence = float(proba[0].max())
            
            class_idx = pred[0]
            exp_df = explain_prediction(explainer, X_t, class_idx, FEATURE_NAMES)
            pos, neg = top_factors(exp_df, n=5)
            
            batch_results.append({
                "tier": tier, "confidence": confidence,
                "proba_dict": {c: round(float(p), 3) for c, p in zip(le.classes_, proba[0])},
                "top_positive": list(zip(pos["feature"], pos["shap_value"])),
                "top_negative": list(zip(neg["feature"], neg["shap_value"])),
                "raw_snapshot": row_df[common_cols].iloc[0].to_dict(),
            })
            
        st.session_state["batch_results"] = batch_results
        st.session_state["selected_customer_idx"] = 0
        st.success(f"Assessed {len(batch_results)} customer(s).")
        
    results = get_batch_results()
    
    if not results:
        st.info("Run an assessment above to see results here.")
    else:
        st.markdown("---")
        if len(results) > 1:
            st.subheader(f"Batch Results — {len(results)} Customers")
            summary_rows = []
            for i, r in enumerate(results):
                decision, _ = decision_from_tier(r["tier"], r["confidence"])
                summary_rows.append({
                    "Customer": i + 1, "Tier": r["tier"],
                    "Confidence": f"{r['confidence']:.0%}", "Decision": decision
                })
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            tier_counts = pd.Series([r["tier"] for r in results]).value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0)
            fig = px.bar(x=tier_counts.index, y=tier_counts.values,
                         labels={"x": "Risk Tier", "y": "Number of Customers"},
                         title="Risk Tier Distribution",
                         color=tier_counts.index, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font_color=t["text"], height=320)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Inspect one customer in detail")
            customer_selector_widget()
            
        if st.button("🗑️ Clear these results", use_container_width=False):
            del st.session_state["batch_results"]
            st.session_state.pop("selected_customer_idx", None)
            st.rerun()
            
        selected = get_selected_customer()
        if selected:
            i = st.session_state.get("selected_customer_idx", 0)
            tier = selected["tier"]
            confidence = selected["confidence"]
            proba_arr = [selected["proba_dict"].get(c, 0) for c in le.classes_]
            decision, decision_type = decision_from_tier(tier, confidence)
            
            st.markdown("---")
            hc1, hc2, hc3, hc4 = st.columns(4)
            with hc1: kpi_card(f"Customer {i+1} — Risk Tier", tier)
            with hc2: kpi_card("Confidence", f"{confidence:.0%}")
            with hc3: st.markdown(f"<div class='rp-card'><div class='rp-kpi-label'>Risk Level</div>{risk_badge(tier)}</div>", unsafe_allow_html=True)
            with hc4: st.markdown(f"<div class='rp-card'><div class='rp-kpi-label'>Decision</div><span class='rp-badge rp-badge-{decision_type}'>{decision}</span></div>", unsafe_allow_html=True)
                
            proba_df = pd.DataFrame({"Tier": le.classes_, "Probability": proba_arr})
            fig = px.bar(proba_df, x="Tier", y="Probability", color="Tier", title="Class Probabilities", color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font_color=t["text"], height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            if confidence < 0.55:
                st.warning("Low confidence prediction — recommend mandatory manual review.")
                
            with st.expander("Why did the model give this result? (in plain language)", expanded=True):
                st.markdown("**What worked against this customer:**")
                for f, v in selected["top_positive"]:
                    st.markdown(f"🔴 {business_sentence(f, v)}")
                st.markdown("**What worked in this customer's favor:**")
                for f, v in selected["top_negative"]:
                    st.markdown(f"🟢 {business_sentence(f, v)}")

elif st.session_state["current_page"] == "AI Risk Assistant":
    st.markdown('<div class="rp-section-title">AI Risk Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">Interact natively with underwriting explanations — grounded directly in model metrics 🤖</div>', unsafe_allow_html=True)
    
    lp = get_selected_customer()
    if lp is None:
        center_layout_col, _ = st.columns([13, 7])
        with center_layout_col:
            st.info("Run a prediction on the Customer Assessment page first, then return here to ask questions about it.", icon="ℹ️")
    else:
        customer_selector_widget()
        lp = get_selected_customer()
        i = st.session_state.get("selected_customer_idx", 0)
        st.markdown(f"**Currently discussing Customer {i+1}:** Tier `{lp['tier']}`, Confidence `{lp['confidence']:.0%}`")
        
        if not is_ai_available():
            st.warning("AI Assistant is not configured. Add a GOOGLE_API_KEY to Streamlit secrets.")
        else:
            # Persistent state tracking variables to avoid component rerun drops
            if "submitted_query" not in st.session_state:
                st.session_state["submitted_query"] = None

            # Render Text input area cleanly
            user_q = st.text_input("Type your question regarding this profile:", key="ai_text_input")
            
            # Action button setup
            click_ask = st.button("Ask Assistant", type="primary")
            
            st.markdown("<div style='padding-top:4px;'></div>", unsafe_allow_html=True)
            st.caption("💡 **Quick Suggestions** (Click to submit immediately):")
            
            # Formulate clean auto-trigger horizontal capsule selectors
            cols = st.columns(2)
            for idx, q in enumerate(SUGGESTED_QUESTIONS):
                if cols[idx % 2].button(q, key=f"sugg_{idx}", use_container_width=True):
                    st.session_state["submitted_query"] = q
                    st.rerun()

            if click_ask and user_q:
                st.session_state["submitted_query"] = user_q

            # Handle execution of active question state
            active_query = st.session_state["submitted_query"]
            if active_query:
                st.markdown("---")
                with st.chat_message("user"):
                    st.write(active_query)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing profile weights..."):
                        context_json = build_context(lp["tier"], lp["confidence"], lp["proba_dict"], lp["top_positive"], lp["top_negative"], lp["raw_snapshot"])
                        answer = call_llm(active_query, context_json)
                        st.markdown(answer)
                # Clear active tracking bounds
                st.session_state["submitted_query"] = None
        
        # ---------------------------------------------------------------------------
        # PLOTLY SHAP VALUE HORIZONTAL PLOT ENGINE (CLEAN BUSINESS LABELS)
        # ---------------------------------------------------------------------------
        st.markdown("### 📊 Local Feature Impact Scale (SHAP Values)")
        
        # Comprehensive short glossary mapping dictionary
        glossary_mapping = {
            "Age_Oldest_TL": "Oldest Account Age",
            "Age_Newest_TL": "Newest Account Age",
            "credit_history_years": "Credit History Length",
            "Time_With_Curr_Empr": "Job Tenure Length",
            "NETMONTHLYINCOME": "Net Monthly Income",
            "monthly_income": "Net Monthly Income",
            "education": "Education Level",
            "MARITALSTATUS": "Marital Status",
            "GENDER": "Gender Profile",
            "Tot_Missed_Pmnt": "Total Missed Payments",
            "missed_payments_total": "Total Missed Payments",
            "num_deliq_6_12mts": "Mid-Term Delinquencies",
            "max_recent_level_of_deliq": "Worst Recent Delinquency",
            "recent_level_of_deliq": "Current Delinquency Level",
            "num_std_12mts": "On-Time Payments (Year)",
            "ontime_payments_12m": "On-Time Payments (Year)",
            "time_since_recent_payment": "Months Since Payment",
            "enq_L3m": "Recent Applications (3M)",
            "recent_loan_applications_3m": "Recent Applications (3M)",
            "time_since_recent_enq": "Months Since Application",
            "pct_PL_enq_L6m_of_ever": "Personal Loan Enquiries %",
            "pct_CC_enq_L6m_of_ever": "Credit Card Enquiries %",
            "last_prod_enq2": "Last Product Applied",
            "first_prod_enq2": "First Product Applied",
            "CC_enq_L12m": "Card Applications (Year)",
            "PL_enq_L12m": "Loan Applications (Year)",
            "Secured_TL": "Secured Debt Accounts",
            "Unsecured_TL": "Unsecured Debt Accounts",
            "Tot_TL_closed_L12M": "Closed Accounts (Year)",
            "pct_tl_closed_L12M": "Account Closure Rate",
            "pct_tl_closed_L6M": "Recent Closure Rate",
            "pct_tl_open_L6M": "Recent Open Rate",
            "CC_Flag": "Has Credit Card",
            "PL_Flag": "Has Personal Loan",
            "has_credit_card": "Has Credit Card",
            "has_personal_loan": "Has Personal Loan",
            "CC_TL": "Total Credit Cards",
            "PL_TL": "Total Personal Loans",
            "Home_TL": "Total Home Loans",
            "Auto_TL": "Total Auto Loans",
            "Gold_TL": "Total Gold Loans",
            "Other_TL": "Other Credit Lines",
        }

        # Dynamic reverse-lookup dictionary checks for both clean text components
        pos_features = [glossary_mapping.get(f, f.replace("_", " ").title()) for f, v in lp["top_positive"]]
        pos_values = [round(float(v), 3) for f, v in lp["top_positive"]]
        
        neg_features = [glossary_mapping.get(f, f.replace("_", " ").title()) for f, v in lp["top_negative"]]
        neg_values = [round(float(v), 3) for f, v in lp["top_negative"]]
        
        all_features = pos_features + neg_features
        all_values = pos_values + neg_values
        colors = ["#EF4444"] * len(pos_values) + ["#10B981"] * len(neg_values)
        
        if all_features:
            fig_shap = go.Figure(go.Bar(
                x=all_values,
                y=all_features,
                orientation='h',
                marker_color=colors,
                text=[f"+{v}" if v > 0 else str(v) for v in all_values],
                textposition='auto'
            ))
            fig_shap.update_layout(
                title="Features Pushing Risk Up (Red) vs Down (Green)",
                paper_bgcolor=t["bg"],
                plot_bgcolor=t["bg"],
                font_color=t["text"],
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.caption("No diagnostic metric arrays available for this profile.")

        # ---------------------------------------------------------------------------
        # ROBUST REVERSE LOOKUP DICTIONARY META GLOSSARY
        # ---------------------------------------------------------------------------
        with st.expander("💼 Human-Readable Customer Metadata Metrics", expanded=False):
            raw_data = lp["raw_snapshot"]
            
            # Map raw python metric nomenclature to corporate bank definitions cleanly
            glossary_mapping = {
    # Account & History
    "Age_Oldest_TL": "Oldest Account Age",
    "Age_Newest_TL": "Newest Account Age",
    "credit_history_years": "Credit History Length",
    "Time_With_Curr_Empr": "Job Tenure Length",
    
    # Wallet & Financials
    "NETMONTHLYINCOME": "Net Monthly Income",
    "monthly_income": "Net Monthly Income",
    "education": "Education Level",
    "MARITALSTATUS": "Marital Status",
    "GENDER": "Gender Profile",

    # Delinquency Metrics
    "Tot_Missed_Pmnt": "Total Missed Payments",
    "missed_payments_total": "Total Missed Payments",
    "num_deliq_6_12mts": "Recent Mid-Term Delinquencies",
    "max_recent_level_of_deliq": "Worst Recent Delinquency",
    "recent_level_of_deliq": "Current Delinquency Level",
    "num_std_12mts": "On-Time Payments (Year)",
    "ontime_payments_12m": "On-Time Payments (Year)",
    "time_since_recent_payment": "Months Since Payment",

    # Credit Enquiries & Applications
    "enq_L3m": "Recent Applications (3M)",
    "recent_loan_applications_3m": "Recent Applications (3M)",
    "time_since_recent_enq": "Months Since Application",
    "pct_PL_enq_L6m_of_ever": "Personal Loan Enquiries %",
    "pct_CC_enq_L6m_of_ever": "Credit Card Enquiries %",
    "last_prod_enq2": "Last Product Applied",
    "first_prod_enq2": "First Product Applied",
    "CC_enq_L12m": "Card Applications (Year)",
    "PL_enq_L12m": "Loan Applications (Year)",

    # Active Exposure & Portfolios
    "Secured_TL": "Secured Collateral Accounts",
    "Unsecured_TL": "Unsecured Debt Accounts",
    "Tot_TL_closed_L12M": "Closed Accounts (Year)",
    "pct_tl_closed_L12M": "Account Closure Rate",
    "pct_tl_closed_L6M": "Recent Closure Rate",
    "pct_tl_open_L6M": "Recent Open Rate",
    "CC_Flag": "Has Credit Card",
    "PL_Flag": "Has Personal Loan",
    "has_credit_card": "Has Credit Card",
    "has_personal_loan": "Has Personal Loan",
    
    # Product Specific Counts
    "CC_TL": "Total Credit Cards",
    "PL_TL": "Total Personal Loans",
    "Home_TL": "Total Home Loans",
    "Auto_TL": "Total Auto Loans",
    "Gold_TL": "Total Gold Loans",
    "Other_TL": "Other Credit Lines",

    # Substandard/Loss Flags
    "num_dbt": "Doubtful Accounts Count",
    "num_dbt_12mts": "Recent Doubtful Accounts",
    "num_lss": "Written-Off Losses",
    "num_sub": "Substandard Accounts Count",
    "num_sub_12mts": "Recent Substandard Accounts",
    "num_sub_6mts": "Critical Substandard Accounts",
    "num_times_60p_dpd": "Severe Default Incidents",
}
            
            formatted_rows = []
            for raw_key, raw_val in raw_data.items():
                friendly_label = glossary_mapping.get(raw_key, raw_key.replace("_", " ").title())
                
                # Format specific numeric types seamlessly
                if isinstance(raw_val, (int, float)) and "income" in raw_key.lower():
                    formatted_val = f"₹{raw_val:,.2f}"
                elif isinstance(raw_val, float):
                    formatted_val = f"{raw_val:.2f}"
                else:
                    formatted_val = str(raw_val)
                    
                formatted_rows.append({"Underwriting Parameter": friendly_label, "Active Value Profile": formatted_val})
                
            st.table(pd.DataFrame(formatted_rows))

elif st.session_state["current_page"] == "Model Analytics":
    st.markdown('<div class="rp-section-title">Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">Deployment model performance on the held-out test set</div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        if importance_df is not None:
            fig = px.bar(importance_df.head(10).sort_values("mean_abs_shap"), 
                         x="mean_abs_shap", y="feature", orientation="h", 
                         title="Top 10 Risk Drivers (SHAP)", 
                         color_discrete_sequence=[t["accent"]])
            fig.update_layout(paper_bgcolor=t["bg"], plot_bgcolor=t["bg"], font_color=t["text"], height=350)
            st.plotly_chart(fig, use_container_width=True)
    with col_b:
        if report_df is not None and "P1" in report_df.index:
            classes = ["P1", "P2", "P3", "P4"]
            f1s = [report_df.loc[c, "f1-score"] for c in classes if c in report_df.index]
            fig2 = px.pie(values=f1s, names=classes[:len(f1s)], title="F1-Score by Risk Tier", hole=0.5, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig2.update_layout(paper_bgcolor=t["bg"], font_color=t["text"], height=350)
            st.plotly_chart(fig2, use_container_width=True)
            
    st.info("The full model reaches 99% Macro-F1 using Credit_Score and Age, which aren't available at deployment.", icon="ℹ️")
        
    if cm_df is not None:
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix", labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(paper_bgcolor=t["bg"], font_color=t["text"], height=450)
        st.plotly_chart(fig, use_container_width=True)
        
    if report_df is not None:
        st.subheader("Classification Report")
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

elif st.session_state["current_page"] == "Business Insights":
    st.markdown('<div class="rp-section-title">Business Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">Portfolio-level risk drivers and recommendations</div>', unsafe_allow_html=True)
    
    batch = get_batch_results()
    if batch and len(batch) > 1:
        st.subheader(f"📊 Portfolio Distribution — {len(batch)} Customers Loaded")
        tier_counts = pd.Series([r["tier"] for r in batch]).value_counts().reindex(["P1", "P2", "P3", "P4"]).fillna(0)
        decisions = [decision_from_tier(r["tier"], r["confidence"])[0] for r in batch]
        decision_counts = pd.Series(decisions).value_counts()
        
        bc1, bc2 = st.columns(2)
        with bc1:
            fig_b = px.pie(values=tier_counts.values, names=tier_counts.index, hole=0.5, title="Risk Tier Breakdown", color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_b.update_layout(paper_bgcolor=t["bg"], font_color=t["text"], height=320)
            st.plotly_chart(fig_b, use_container_width=True)
        with bc2:
            fig_c = px.pie(values=decision_counts.values, names=decision_counts.index, hole=0.5, title="Recommended Decisions", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_c.update_layout(paper_bgcolor=t["bg"], font_color=t["text"], height=320)
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.warning("📊 No batch data active. Upload a 'Quick Assessment — batch CSV' to unlock batch metrics.")

elif st.session_state["current_page"] == "Explainability":
    st.markdown('<div class="rp-section-title">Explainability Workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">Global and local model explanations via SHAP values</div>', unsafe_allow_html=True)
    
    lp = get_selected_customer()
    if lp is not None:
        customer_selector_widget()
        lp = get_selected_customer()
        i = st.session_state.get("selected_customer_idx", 0)
        st.subheader(f"Why was Customer {i+1} rated \"{lp['tier']}\"?")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔴 What worked against this customer")
            for f, v in lp["top_positive"]: st.markdown(f"- {business_sentence(f, v)}")
        with c2:
            st.markdown("#### 🟢 What worked in this customer's favor")
            for f, v in lp["top_negative"]: st.markdown(f"- {business_sentence(f, v)}")
    else:
        center_layout_col, _ = st.columns([13, 7])
        with center_layout_col: st.info("Run a prediction on Customer Assessment first.", icon="ℹ️")

elif st.session_state["current_page"] == "Reports":
    st.markdown('<div class="rp-section-title">Reports Management</div>', unsafe_allow_html=True)
    lp = get_selected_customer()
    if lp is not None:
        report_text = f"RISKPILOT REPORT\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTier: {lp['tier']}\nConfidence: {lp['confidence']:.1%}"
        st.download_button("Download Assessment Report (TXT)", report_text, file_name=f"risk_report.txt", use_container_width=True)
    else:
        center_layout_col, _ = st.columns([13, 7])
        with center_layout_col: st.info("Run an assessment first to unlock file downloads.", icon="ℹ️")

else:
    st.markdown('<div class="rp-section-title">About RiskPilot</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-section-sub">What happens under the hood of this risk engine? 🧠</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 The Mission
    **RiskPilot** is a smart credit underwriting assistant built to sort loan applicants into four clear risk categories (**P1: Super Safe** to **P4: High Risk**). 
    
    Instead of acting like a mysterious "black box" that just spits out a score without explanation, RiskPilot works out exactly *why* a decision was made and breaks it down into plain English that humans can actually understand.
    
    ---
    
    ### 📊 The Data Twist
    During data exploration, we found a fascinating **Credit Score Threshold Paradox**: the bank's historical records showed a razor-sharp cutoff where anyone with a top credit score was automatically thrown into P1. 
    
    To make our engine genuinely useful for real-world scenarios where bureau data might be missing, we split our modeling into two pathways:
    * **Track A (The Cheat Code):** Uses every data point imaginable (including the credit score) to reach a near-perfect **99.0% Macro F1-Score**.
    * **Track B (The Real Deal):** Our actual deployable edge engine. It relies strictly on **42 raw transactional features** available right at checkout, scoring an honest, robust **69.1% Macro F1-Score**.
    
    ---
    
    ### 🛠️ The Tech Stack
    * **The Brains:** A finely-tuned **CatBoost Classifier** optimized for complex tabular banking data.
    * **The Translator:** **SHAP (SHapley Additive exPlanations)** values, which turn raw mathematical matrix weights into human-readable reasons.
    * **The Talker:** A **Retrieval-Augmented Generation (RAG)** loop using an LLM to power our AI Risk Assistant chat window safely without hallucinations.
    
    ---
    
    ### ⚠️ Honesty Box (System Limitations)
    1. **Eco-Blind:** The model assumes a stable economy and doesn't track current inflation spikes or interest rate changes.
    2. **Cold Starts:** If an applicant is completely brand new to credit with zero history, the system gracefully flags them for an automated human review loop rather than guessing blindly.
    """)
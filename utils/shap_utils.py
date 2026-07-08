"""
SHAP explanation helpers for RiskPilot.
Caches the explainer so it's built once per session, not on every prediction.
"""
import streamlit as st
import shap
import numpy as np
import pandas as pd


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)


def explain_prediction(explainer, X_row_t, predicted_class_idx, feature_names):
    """Return sorted (feature, shap_value, raw_value) for one prediction.

    Defensive: predicted_class_idx may arrive as a numpy scalar, a
    0-d/1-d array, or a plain int depending on caller (CatBoost's
    .predict() returns a 2D array, so callers must ravel it -- this
    coerces to a plain int regardless, to avoid a silent shape bug)."""
    class_idx = int(np.asarray(predicted_class_idx).ravel()[0])

    shap_vals = explainer.shap_values(X_row_t)
    # shap_vals shape: (1, n_features, n_classes) for multiclass CatBoost
    row_shap = np.asarray(shap_vals)[0, :, class_idx].ravel()
    raw_vals = np.asarray(X_row_t)[0].ravel()

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": row_shap,
        "raw_value": raw_vals,
    })
    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return df


def top_factors(explanation_df, n=5):
    """Split into top positive (risk-increasing) and negative (risk-reducing) factors."""
    positive = explanation_df[explanation_df["shap_value"] > 0].head(n)
    negative = explanation_df[explanation_df["shap_value"] < 0].head(n)
    return positive, negative


def plain_english_factor(feature_name, shap_value):
    """Translate a raw feature name + SHAP direction into a business sentence."""
    direction = "increased" if shap_value > 0 else "decreased"
    readable = feature_name.replace("_", " ")
    return f"**{readable}** {direction} this customer's predicted risk."


# --------------------------------------------------------------------------
# Non-technical translation layer
# --------------------------------------------------------------------------
# Maps a raw feature name to a plain-English description of what it means,
# and a template sentence for when it INCREASES risk vs. DECREASES risk.
# This is what should be shown to credit officers/executives -- never the
# raw feature name or the SHAP number itself.
FEATURE_EXPLANATIONS = {
    "enq_L3m": {
        "concept": "how many times the customer recently applied for new credit (loans, cards) in the last 3 months",
        "increases": "This customer has applied for credit several times very recently, which is a common warning sign lenders watch for -- it can mean the person is short on cash and seeking credit from multiple places at once.",
        "decreases": "This customer has not been applying for new credit recently, which is a reassuring, stable sign.",
    },
    "time_since_recent_enq": {
        "concept": "how long ago the customer's most recent credit application was",
        "increases": "This customer applied for credit very recently, which adds a small amount of risk on its own.",
        "decreases": "It has been a while since this customer applied for any new credit, which works in their favor.",
    },
    "Age_Oldest_TL": {
        "concept": "how long this customer has held their oldest loan or credit account",
        "increases": "This customer has a relatively short credit history, so there is less track record to judge them by, which adds risk.",
        "decreases": "This customer has a long credit history, giving the bank a solid track record to rely on -- this lowers their risk.",
    },
    "Age_Newest_TL": {
        "concept": "how recently this customer opened their newest loan or credit account",
        "increases": "This customer opened a new credit account very recently, which can be a minor risk signal.",
        "decreases": "This customer hasn't opened any new accounts recently, a stabilizing factor.",
    },
    "pct_PL_enq_L6m_of_ever": {
        "concept": "what share of all this customer's personal loan enquiries, ever, happened in just the last 6 months",
        "increases": "A large share of this customer's personal loan applications happened very recently, suggesting a sudden, concentrated need for credit -- a risk signal.",
        "decreases": "This customer's personal loan applications are spread out over time rather than clustered recently, which is a good sign.",
    },
    "recent_level_of_deliq": {
        "concept": "how seriously overdue the customer's most recent missed payment was",
        "increases": "This customer's most recent missed payment was fairly serious, which raises concern about their repayment reliability.",
        "decreases": "This customer's payment history shows no serious recent delays, which supports a lower risk rating.",
    },
    "max_recent_level_of_deliq": {
        "concept": "the worst level of missed payment the customer has had recently",
        "increases": "This customer has had a notably serious missed payment recently, which is a red flag for lenders.",
        "decreases": "This customer hasn't had any serious missed payments recently, which is reassuring.",
    },
    "num_std_12mts": {
        "concept": "how many payments this customer made on time in the last 12 months",
        "increases": "This customer has made fewer on-time payments than expected in the past year, which raises risk.",
        "decreases": "This customer has a strong record of on-time payments over the past year -- a strong positive signal.",
    },
    "num_deliq_6_12mts": {
        "concept": "how many missed payments happened 6 to 12 months ago",
        "increases": "This customer had missed payments in the 6-12 month period before now, which adds some risk.",
        "decreases": "This customer had no missed payments in that period, which helps their case.",
    },
    "Unsecured_TL": {
        "concept": "how many of the customer's loans are unsecured (not backed by collateral like a house or gold)",
        "increases": "This customer holds a relatively high number of loans with no collateral behind them, which is riskier for the lender if something goes wrong.",
        "decreases": "This customer holds relatively few uncollateralized loans, which is safer for the lender.",
    },
    "Time_With_Curr_Empr": {
        "concept": "how long the customer has been with their current employer",
        "increases": "This customer has been with their current employer for a shorter time, which can mean less income stability.",
        "decreases": "This customer has a long, stable employment history with their current employer, which supports a lower risk rating.",
    },
    "NETMONTHLYINCOME": {
        "concept": "the customer's stated monthly income",
        "increases": "This customer's income is on the lower side relative to other applicants, which adds some risk.",
        "decreases": "This customer's income is comfortably strong, which supports a lower risk rating.",
    },
    "Tot_Missed_Pmnt": {
        "concept": "the total number of payments this customer has ever missed",
        "increases": "This customer has a history of missed payments overall, which is a meaningful risk factor.",
        "decreases": "This customer has very few missed payments in their history, which is reassuring.",
    },
    "CC_utilization": {
        "concept": "how much of their available credit card limit the customer is currently using",
        "increases": "This customer is using a large portion of their available credit card limit, which often signals financial strain.",
        "decreases": "This customer is using only a small portion of their available credit, a healthy sign.",
    },
    "PL_utilization": {
        "concept": "how much of their personal loan limit the customer is currently using",
        "increases": "This customer is using a large share of their personal loan capacity, which adds risk.",
        "decreases": "This customer has room left on their personal loan limits, which is a good sign.",
    },
}


def business_sentence(feature_name, shap_value):
    """Full non-technical sentence for a feature's effect on this prediction.
    Falls back to a generic but still jargon-free sentence if the feature
    isn't in the curated dictionary above."""
    info = FEATURE_EXPLANATIONS.get(feature_name)
    if info is None:
        readable = feature_name.replace("_", " ").lower()
        direction = "raised" if shap_value > 0 else "lowered"
        return f"A factor related to **{readable}** {direction} this customer's risk level."
    return info["increases"] if shap_value > 0 else info["decreases"]


def feature_concept(feature_name):
    """Short plain description of what a feature measures, for tooltips."""
    info = FEATURE_EXPLANATIONS.get(feature_name)
    if info is None:
        return feature_name.replace("_", " ").lower()
    return info["concept"]
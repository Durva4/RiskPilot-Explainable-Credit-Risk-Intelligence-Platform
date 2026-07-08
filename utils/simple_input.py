"""
Simplified input handling for RiskPilot.

Most real users (loan officers, non-technical staff) don't know what
`pct_PL_enq_L6m_of_ever` means, and won't have a CSV with all 42 model
columns pre-formatted. This module provides:

1. A curated set of ~10 easy-to-understand fields that carry most of the
   real signal (credit history length, recent missed payments, recent
   credit-seeking activity, income, etc.)
2. A function to expand a simplified row/dataframe into the full 42-column
   shape the model needs, filling everything else with sensible defaults
   computed from the training data (median for numeric, mode for
   categorical) -- NOT zeros, which would misrepresent those customers.
3. A plain-English glossary for every one of the 42 columns, for anyone
   who wants to understand or build the full/advanced CSV themselves.
"""
import pandas as pd
import joblib

DEFAULTS_PATH = "models/deployment_defaults.pkl"


# ---------------------------------------------------------------------------
# Simplified field set -- shown in Quick Assessment mode (form + batch CSV)
# ---------------------------------------------------------------------------
# Each entry: (simple_column_name, target_model_column, field_type, help_text)
SIMPLE_FIELDS = [
    ("credit_history_years", "Age_Oldest_TL", "years_to_months",
     "How many years ago did this person open their very first loan or credit card?"),
    ("recent_loan_applications_3m", "enq_L3m", "int",
     "How many times has this person applied for any new loan or credit card in the last 3 months?"),
    ("ontime_payments_12m", "num_std_12mts", "int",
     "How many payments has this person made on time in the last 12 months?"),
    ("missed_payments_total", "Tot_Missed_Pmnt", "int",
     "How many payments has this person ever missed, in total?"),
    ("monthly_income", "NETMONTHLYINCOME", "int",
     "This person's monthly income (in rupees)."),
    ("education", "EDUCATION", "category",
     "Highest education level completed."),
    ("months_at_current_job", "Time_With_Curr_Empr", "int",
     "How many months has this person been at their current job?"),
    ("has_credit_card", "CC_Flag", "yesno",
     "Does this person currently have a credit card?"),
    ("has_personal_loan", "PL_Flag", "yesno",
     "Does this person currently have a personal loan?"),
    ("recent_missed_payment_severity", "recent_level_of_deliq", "int",
     "On a scale of 0 (no missed payments) to 10 (very seriously overdue), "
     "how bad was this person's most recent missed payment, if any? Enter 0 if none."),
]

EDUCATION_OPTIONS = ["GRADUATE", "12TH", "SSC", "POST-GRADUATE",
                     "UNDER GRADUATE", "OTHERS", "PROFESSIONAL"]


def load_defaults():
    return joblib.load(DEFAULTS_PATH)


def simple_template_csv():
    """Column headers for the easy/quick-upload CSV template."""
    return [f[0] for f in SIMPLE_FIELDS]


def expand_simple_row(simple_values: dict, defaults: dict, common_cols: list) -> dict:
    """Takes a dict of simplified field values, returns a full dict with
    all 42 model columns -- explicit fields override the default, every
    other column falls back to the training-data median/mode."""
    row = dict(defaults)  # start from sensible, data-derived defaults

    for simple_name, target_col, field_type, _ in SIMPLE_FIELDS:
        if simple_name not in simple_values or pd.isna(simple_values[simple_name]):
            continue
        val = simple_values[simple_name]
        if field_type == "years_to_months":
            row[target_col] = float(val) * 12
        elif field_type == "yesno":
            if isinstance(val, str):
                row[target_col] = 1 if val.strip().lower() in ("yes", "y", "true", "1") else 0
            else:
                row[target_col] = int(bool(val))
        else:
            row[target_col] = val

    return {c: row[c] for c in common_cols}


def expand_simple_dataframe(simple_df: pd.DataFrame, defaults: dict, common_cols: list) -> pd.DataFrame:
    """Batch version -- expands a whole CSV of simplified rows (e.g. 100
    customers) into full 42-column rows ready for the model."""
    expanded_rows = []
    for _, r in simple_df.iterrows():
        expanded_rows.append(expand_simple_row(r.to_dict(), defaults, common_cols))
    return pd.DataFrame(expanded_rows)


# ---------------------------------------------------------------------------
# Full glossary -- every one of the 42 model columns, plain English
# ---------------------------------------------------------------------------
COLUMN_GLOSSARY = {
    "Age_Newest_TL": "How recently (in months) this person opened their MOST RECENT loan or credit account.",
    "Age_Oldest_TL": "How long ago (in months) this person opened their FIRST EVER loan or credit account -- their credit history length.",
    "CC_Flag": "Whether this person currently holds a credit card (1 = yes, 0 = no).",
    "CC_TL": "How many credit card accounts this person has had in total.",
    "CC_enq_L12m": "How many times this person applied for a credit card in the last 12 months.",
    "EDUCATION": "This person's highest completed education level.",
    "GENDER": "This person's gender.",
    "GL_Flag": "Whether this person currently holds a gold loan (1 = yes, 0 = no).",
    "HL_Flag": "Whether this person currently holds a home loan (1 = yes, 0 = no).",
    "Home_TL": "How many home loan accounts this person has had in total.",
    "MARITALSTATUS": "This person's marital status.",
    "NETMONTHLYINCOME": "This person's self-reported monthly income.",
    "Other_TL": "How many other/miscellaneous loan accounts this person has had.",
    "PL_Flag": "Whether this person currently holds a personal loan (1 = yes, 0 = no).",
    "PL_TL": "How many personal loan accounts this person has had in total.",
    "PL_enq_L12m": "How many times this person applied for a personal loan in the last 12 months.",
    "Secured_TL": "How many of this person's loans are backed by collateral (like a house or gold).",
    "Time_With_Curr_Empr": "How many months this person has been with their current employer.",
    "Tot_Missed_Pmnt": "The total number of payments this person has ever missed, across all accounts.",
    "Tot_TL_closed_L12M": "How many of this person's accounts were closed in the last 12 months.",
    "Unsecured_TL": "How many of this person's loans have NO collateral behind them (like most credit cards and personal loans).",
    "enq_L3m": "How many times this person applied for any credit (loan or card) in the last 3 months.",
    "first_prod_enq2": "The very first type of credit product this person ever applied for.",
    "last_prod_enq2": "The most recent type of credit product this person applied for.",
    "max_recent_level_of_deliq": "The worst severity of missed payment this person has had recently (0 = none, higher = more serious).",
    "num_dbt": "Number of this person's accounts marked 'doubtful' by the credit bureau -- a serious repayment concern flag.",
    "num_dbt_12mts": "Same as above, but only counting the last 12 months.",
    "num_deliq_6_12mts": "Number of missed payments that happened 6 to 12 months ago.",
    "num_lss": "Number of this person's accounts written off as a full loss by a lender -- the most severe bureau status.",
    "num_std_12mts": "Number of payments this person made ON TIME in the last 12 months.",
    "num_sub": "Number of this person's accounts marked 'substandard' (partial/irregular payment) by the credit bureau.",
    "num_sub_12mts": "Same as above, but only counting the last 12 months.",
    "num_sub_6mts": "Same as above, but only counting the last 6 months.",
    "num_times_60p_dpd": "Number of times this person has been 60+ days late on a payment -- a serious delinquency marker.",
    "pct_CC_enq_L6m_of_ever": "Of all the credit card applications this person has EVER made, what share happened in just the last 6 months (0 to 1).",
    "pct_PL_enq_L6m_of_ever": "Of all the personal loan applications this person has EVER made, what share happened in just the last 6 months (0 to 1).",
    "pct_tl_closed_L12M": "What share of this person's total accounts were closed in the last 12 months (0 to 1).",
    "pct_tl_closed_L6M": "What share of this person's total accounts were closed in the last 6 months (0 to 1).",
    "pct_tl_open_L6M": "What share of this person's total accounts were opened in the last 6 months (0 to 1).",
    "recent_level_of_deliq": "How seriously overdue this person's MOST RECENT missed payment was (0 = none, higher = more serious).",
    "time_since_recent_enq": "How many days ago this person last applied for any credit.",
    "time_since_recent_payment": "How many days ago this person last made any payment.",
}


def glossary_dataframe(common_cols):
    rows = [{"Column": c, "What it means": COLUMN_GLOSSARY.get(c, "No description available.")}
            for c in common_cols]
    return pd.DataFrame(rows)
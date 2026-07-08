import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


"""
AI Risk Assistant for RiskPilot.

Design principle: this assistant NEVER retrains or re-predicts. It only
explains an ALREADY-COMPUTED prediction using the actual SHAP values,
confidence score, and raw feature values passed into it. If a required
piece of information isn't available, it says so rather than guessing.

Uses Google's Gemini API (free tier: ~1,500 requests/day on Gemini 3
Flash, no credit card required, no data-sharing tradeoff needed). Swap
the `call_llm` function body to use OpenAI, Grok, or another provider's
SDK if preferred -- the rest of the module is provider-agnostic.
"""
import streamlit as st
import os
import json

GEMINI_MODEL = "gemini-2.5-flash"


def is_ai_available():
    """Checks for an API key in Streamlit secrets or environment variables.
    Returns False (never crashes) if none is found."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return True
    except Exception:
        pass
    return bool(os.environ.get("GOOGLE_API_KEY"))


def _get_api_key():
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY")


def build_context(predicted_tier, confidence, proba_dict, top_positive_factors,
                   top_negative_factors, raw_feature_snapshot):
    """Assembles ONLY real, computed information into a structured context
    block. The LLM is instructed to answer strictly from this -- it must
    not invent numbers or facts not present here."""
    context = {
        "predicted_risk_tier": predicted_tier,
        "confidence_score": f"{confidence:.1%}",
        "class_probabilities": proba_dict,
        "top_risk_increasing_factors": [
            {"feature": f, "shap_value": round(v, 3)} for f, v in top_positive_factors
        ],
        "top_risk_reducing_factors": [
            {"feature": f, "shap_value": round(v, 3)} for f, v in top_negative_factors
        ],
        "key_raw_values": raw_feature_snapshot,
    }
    return json.dumps(context, indent=2)


SYSTEM_PROMPT = """You are a banking credit risk assistant embedded inside \
RiskPilot, an internal credit risk decision-support tool.

Rules you must follow strictly:
1. Only use the JSON context provided for this specific customer's prediction. \
Never invent numbers, feature values, or SHAP contributions not present in the context.
2. If the user asks something the context doesn't cover, say clearly that this \
information is not available in the current prediction data -- do not guess.
3. Explain in plain business language suitable for a credit officer, not a data \
scientist. Avoid ML jargon unless the user explicitly asks about the model.
4. When asked for a lending recommendation, base it strictly on the predicted \
tier, confidence score, and known model limitations (e.g., this model does not \
have access to the applicant's official CIBIL Credit_Score at this deployment \
stage, so recommend manual verification for any low-confidence or P3/P4 case).
5. Never state or imply you are making the final lending decision -- you are \
assisting a human credit officer, who makes the final call.
"""


def call_llm(user_question, context_json):
    """Calls Gemini with the prediction context + user question.
    Returns the response text, or an error message string (never raises
    up to the UI layer -- the caller should display the return value)."""
    api_key = _get_api_key()
    if not api_key:
        return ("AI Assistant is not configured. Add a GOOGLE_API_KEY to "
                "Streamlit secrets or environment variables to enable it. "
                "Get a free key (no credit card required) at "
                "https://aistudio.google.com/apikey")

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Prediction context:\n{context_json}\n\n"
            f"Question: {user_question}"
        )
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=full_prompt,
        )
        return response.text
    except ImportError:
        return "The 'google-genai' package is not installed. Run: pip install google-genai"
    except Exception as e:
        return f"AI Assistant error: {e}"


SUGGESTED_QUESTIONS = [
    "Why is this customer classified in this risk tier?",
    "What factors increased this customer's risk the most?",
    "What factors are working in this customer's favor?",
    "Should the bank approve this loan?",
    "What additional verification should be performed?",
    "How could this customer improve their risk profile?",
]
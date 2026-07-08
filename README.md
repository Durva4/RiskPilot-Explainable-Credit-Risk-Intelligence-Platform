# RiskPilot: Explainable Credit Risk Intelligence Platform

RiskPilot is an enterprise-grade, interactive analytics platform designed to bridge the gap between complex machine learning models and real-world banking operations. Built for risk managers, credit analysts, and banking executives, RiskPilot automates credit risk assessment while providing transparent, explainable predictions using Explainable AI (XAI).

---

## 📌 Problem Statement

### The Business Problem

Every time a customer applies for a loan or credit card, banks must determine the likelihood that the applicant will repay or default.

To make this decision, financial institutions rely on two major sources of information:

- **Internal Bank Data**
  - Existing account history
  - Transaction behavior
  - Previous loans
  - Account closures
  - Banking relationship

- **External Credit Bureau Data (e.g., CIBIL)**
  - Credit history across multiple institutions
  - Repayment behavior
  - Loan enquiries
  - Delinquency records
  - Credit utilization

Analyzing these datasets manually is time-consuming and often inconsistent. Traditional machine learning models improve prediction accuracy but usually operate as **black boxes**, making it difficult for analysts and regulators to understand *why* a prediction was made.

RiskPilot addresses this challenge by combining predictive analytics with explainable AI.

---

## 💡 A Simple Example

Imagine two applicants applying for a personal loan.

### Applicant A

- Credit history: **5 years**
- Missed payments: **0**
- Recent loan enquiries: **None**
- Stable repayment behavior

### Applicant B

- Credit history: **6 months**
- Missed payments: **2**
- Four loan applications within the last 90 days
- Irregular repayment history

Even without knowing their salaries, most loan officers would naturally trust Applicant A more than Applicant B.

RiskPilot automates this reasoning at scale by evaluating thousands of applications consistently and instantly.

Instead of producing an incomprehensible risk score, the platform classifies applicants into four operational risk tiers:

| Risk Tier | Meaning |
|-----------|---------|
| 🟢 **P1** | Lowest Risk |
| 🟢 **P2** | Low Risk |
| 🟡 **P3** | Moderate Risk |
| 🔴 **P4** | High Risk |

Each prediction is accompanied by a plain-language explanation, allowing underwriters and compliance teams to understand the reasoning behind every decision.

---

# 🚀 Why RiskPilot Matters

### ✅ Consistent Credit Decisions

Applies identical credit policies across all applications, eliminating subjective human bias.

### ✅ Explainable AI

Provides transparent model explanations using SHAP, helping analysts understand why an applicant received a particular risk classification.

### ✅ Regulatory Compliance

Creates an auditable decision trail that supports regulatory requirements for automated lending systems.

### ✅ Faster Decision Making

Processes large batches of applicants in seconds, significantly reducing loan approval turnaround time.

### ✅ Better Portfolio Monitoring

Identifies emerging portfolio-level risk patterns and supports proactive policy improvements.

---

# 🖼️ Application Walkthrough

---

## 1️⃣ Executive Risk Dashboard

The landing dashboard provides an executive summary of the entire model.

### Highlights

- Overall Accuracy
- Macro F1 Score
- Model Information
- Number of Features
- Dataset Summary
- Performance Overview
- Production vs Fallback Model Notice


## 2️⃣ Customer Credit Risk Assessment

Assess applicants individually or in bulk.

### Features

- Manual customer entry
- CSV batch upload
- Risk Tier Prediction (P1–P4)
- Probability distribution
- Prediction confidence
- Lending recommendation

Possible recommendations include:

- ✅ Approve
- ⚠️ Manual Review
- ❌ Reject


## 3️⃣ Deep Model Analytics

Designed for model validation and technical evaluation.

### Interactive Visualizations

- Confusion Matrix
- Classification Report
- Precision-Recall Curves
- ROC Curves
- Feature Distribution
- Performance Metrics

All charts are interactive using Plotly.

---

## 4️⃣ Explainability Layer

RiskPilot emphasizes model transparency through Explainable AI.

### Global Explainability

- SHAP Feature Importance
- Global Risk Drivers

### Local Explainability

Shows exactly why an individual applicant received a specific prediction.

Example explanations include:

- Short credit history increased risk.
- Multiple recent loan enquiries raised the probability of default.
- Clean repayment history reduced overall risk.

Users can switch between:

- Technical SHAP values
- Plain-English explanations

---

# 🌟 Advanced Features

## 🤖 AI Risk Assistant (Gemini)

A conversational assistant capable of answering questions such as:

- Why was this customer classified as P3?
- Which features influenced the prediction the most?
- What would improve this customer's credit profile?

Responses are generated strictly from SHAP explanations to minimize hallucinations.

---

## 📊 Business Insights Engine

Provides portfolio-level analytics including:

- Most influential risk factors
- Distribution across risk categories
- High-risk customer segments
- Portfolio health summary
- Trend analysis

---

## 📋 Policy Engine

Automatically flags applications requiring additional review.

Example rules include:

- Low prediction confidence
- Borderline probability scores
- Missing customer information
- High-risk combinations of features

---

## 📝 Compliance Reports

Generate downloadable reports for auditing purposes.

Supported formats include:

- TXT
- CSV

Each report contains:

- Applicant information
- Predicted risk tier
- Confidence score
- Recommendation
- SHAP explanation
- Timestamp
- Model version

---

# 🛠️ Technical Architecture

## Frontend

- Streamlit
- Streamlit Option Menu
- Plotly Express
- Custom CSS

---

## Machine Learning

- CatBoost Classifier
- Scikit-Learn
- Joblib
- Pandas
- NumPy

---

## Explainable AI

- SHAP (SHapley Additive exPlanations)

---

## AI Integration

- Google Gemini API
- Graceful fallback when API is unavailable

---

## Development Methodology

- CRISP-DM

---

# 📂 Project Structure

```text
RiskPilot/
│
├── app.py
├── requirements.txt
├── models/
├── data/
├── assets/
├── utils/
├── pages/
├── reports/
└── README.md
```

---

# ⚡ Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/Durva4/RiskPilot-Explainable-Credit-Risk-Intelligence-Platform.git
cd RiskPilot-Explainable-Credit-Risk-Intelligence-Platform
```

---

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate (Windows PowerShell)

```powershell
.venv\Scripts\Activate.ps1
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Launch the Application

```bash
streamlit run app.py
```

---

# 🎯 Key Features

- 📈 Multi-class credit risk prediction (P1–P4)
- 🔍 Explainable AI with SHAP
- 🤖 AI-powered Risk Assistant (Gemini)
- 📊 Executive dashboard with KPIs
- 📂 Batch CSV prediction support
- 📈 Interactive Plotly analytics
- 📝 Automated compliance reports
- ⚖️ Policy-based recommendation engine
- 🔒 Transparent and auditable decision-making
- 🚀 Enterprise-ready architecture

---

# 📜 License

This project is intended for educational and portfolio purposes. Commercial use should comply with applicable licensing and regulatory requirements.

---

## 👤 Author

**Durva Palkar**

If you found this project useful, consider giving it a ⭐ on GitHub.

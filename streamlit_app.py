import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="Credit Risk Scorer", layout="wide")
st.title("💳 Credit Risk Scorer")

# ==============================
# INPUT FORM
# ==============================
col1, col2, col3 = st.columns(3)

with col1:
    loan_amnt = st.number_input("Loan Amount", value=10000)
    int_rate = st.number_input("Interest Rate (%)", value=12.5)
    term = st.selectbox("Term", ["36 months", "60 months"])
    installment = st.number_input("Installment", value=332.0)
    grade = st.selectbox("Grade", ["A","B","C","D","E","F","G"])
    sub_grade = st.selectbox("Sub Grade", ["A1","A2","A3","A4","A5"])

with col2:
    emp_length = st.selectbox("Employment Length",
        ["< 1 year","1 year","2 years","3 years","4 years",
         "5 years","6 years","7 years","8 years","9 years","10+ years"])
    home_ownership = st.selectbox("Home Ownership", ["RENT","OWN","MORTGAGE"])
    annual_inc = st.number_input("Annual Income ($)", value=60000)
    verification_status = st.selectbox("Verification Status",
        ["Verified","Not Verified","Source Verified"])
    purpose = st.selectbox("Purpose",
        ["debt_consolidation","credit_card","home_improvement","major_purchase"])

with col3:
    dti = st.number_input("DTI (%)", value=15.0)
    delinq_2yrs = st.number_input("Delinquencies (2 yrs)", value=0)
    inq_last_6mths = st.number_input("Inquiries (6 mths)", value=1)
    open_acc = st.number_input("Open Accounts", value=5)
    pub_rec = st.number_input("Public Records", value=0)
    revol_util = st.number_input("Revolving Utilization (%)", value=30.0)
    total_acc = st.number_input("Total Accounts", value=10)

# ==============================
# BUTTON & LOGIC
# ==============================
if st.button("🚀 Score Applicant", use_container_width=True):

    payload = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "term": term,
        "installment": installment,
        "grade": grade,
        "sub_grade": sub_grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_inc": annual_inc,
        "verification_status": verification_status,
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": delinq_2yrs,
        "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc,
        "pub_rec": pub_rec,
        "revol_util": revol_util,
        "total_acc": total_acc
    }

    try:
        # Requesting the backend
        response = requests.post("http://127.0.0.1:9000/score", json=payload)
        response.raise_for_status() # Check for HTTP errors
        r = response.json()

        # ==============================
        # EXTRACT VALUES
        # ==============================
        score = r.get("score", "N/A")
        prob = r.get("probability_of_default", 0)
        decision = r.get("decision", "N/A")
        risk_tier = r.get("risk_tier", "N/A")
        drivers = r.get("risk_drivers", [])

        # ==============================
        # DISPLAY RESULTS
        # ==============================
        st.markdown("---")
        st.header("Analysis Results")

        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Credit Score", score)
        m2.metric("Default Risk", f"{prob*100:.1f}%")
        m3.metric("Risk Tier", risk_tier)
        
        # Color coding the decision
        color = "green" if decision.upper() == "APPROVE" else "red"
        m4.markdown(f"**Decision**\n### :{color}[{decision}]")

        # ==============================
        # SHAP / DRIVERS
        # ==============================
        st.subheader("🔍 Key Risk Drivers")
        st.info("Positive impacts (green) lower risk, negative impacts (red) increase it.")

        if drivers:
            # Displaying drivers in a clean list
            for d in drivers:
                feature = d["feature"]
                impact = d["impact"]
                sentiment_color = "green" if impact < 0 else "red"
                symbol = "▼" if impact < 0 else "▲"

                st.markdown(
                    f"<span style='color:{sentiment_color}'>{symbol} <b>{feature}</b>: {impact}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("No explanation drivers available for this prediction.")

    except requests.exceptions.ConnectionError:
        st.error("Connection Refused: Is your backend running on port 9000?")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")

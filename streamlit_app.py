import streamlit as st
import requests

st.set_page_config(page_title="Credit Risk Scorer", layout="wide")

# =============================
# 🎨 STYLES (RESTORED + IMPROVED)
# =============================
st.markdown("""
<style>
.block-container { padding: 2rem 3rem; max-width: 1100px; }

.header-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 16px;
    padding: 2rem;
    color: white;
    margin-bottom: 1.5rem;
}

.card {
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.score-card   { background: linear-gradient(135deg, #667eea, #764ba2); }
.prob-card    { background: linear-gradient(135deg, #f093fb, #f5576c); }
.tier-card    { background: linear-gradient(135deg, #4facfe, #00f2fe); }
.approve-card { background: linear-gradient(135deg, #43e97b, #38f9d7); }
.review-card  { background: linear-gradient(135deg, #f7971e, #ffd200); }
.reject-card  { background: linear-gradient(135deg, #fa709a, #fee140); }

.metric-value { font-size: 2.2rem; font-weight: 800; }
.metric-label { font-size: 0.95rem; font-weight: 600; }

.sidebar-metric {
    background: #111827;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="header-box">
<h1>💳 Credit Risk Scorer</h1>
<p>AI-powered credit assessment using ensemble ML model</p>
</div>
""", unsafe_allow_html=True)

# =============================
# INPUTS
# =============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    loan_amnt = st.number_input("Loan Amount", 500.0, 40000.0, 10000.0)
    int_rate = st.number_input("Interest Rate", 5.0, 30.0, 12.5)
    term = st.selectbox("Term", ["36 months", "60 months"])
    installment = st.number_input("Installment", 0.0, 2000.0, 332.0)
    purpose = st.selectbox("Purpose", ["debt_consolidation","credit_card","home_improvement"])

with col2:
    annual_inc = st.number_input("Annual Income", 0.0, 500000.0, 60000.0)
    emp_length = st.selectbox("Employment Length", ["10+ years","5 years","2 years"])
    home_ownership = st.selectbox("Home Ownership", ["RENT","OWN"])
    grade = st.selectbox("Grade", list("ABCDEFG"))
    sub_grade = st.selectbox("Sub Grade", [f"A{i}" for i in range(1,6)])

with col3:
    dti = st.number_input("DTI", 0.0, 100.0, 15.0)
    revol_util = st.number_input("Utilization", 0.0, 100.0, 30.0)
    open_acc = st.number_input("Open Accounts", 0, 50, 5)
    total_acc = st.number_input("Total Accounts", 0, 50, 10)

with col4:
    delinq_2yrs = st.number_input("Delinquencies", 0, 10, 0)
    inq_last_6mths = st.number_input("Inquiries", 0, 10, 1)
    pub_rec = st.number_input("Public Records", 0, 10, 0)
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🚀 Score Applicant", use_container_width=True)

# =============================
# API CALL
# =============================
if submit:
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
        "verification_status": "Verified",
        "purpose": purpose,
        "dti": dti,
        "delinq_2yrs": int(delinq_2yrs),
        "inq_last_6mths": int(inq_last_6mths),
        "open_acc": int(open_acc),
        "pub_rec": int(pub_rec),
        "revol_util": revol_util,
        "total_acc": int(total_acc),
    }

    r = requests.post("http://127.0.0.1:8000/score", json=payload).json()

    score = r["score"]
    prob = r["probability_of_default"]
    decision = r["decision"]
    drivers = r.get("risk_drivers", [])
    metrics = r.get("model_metrics", {})

    # =============================
    # SIDEBAR METRICS (CLEAN LOOK)
    # =============================
    if metrics:
        st.sidebar.markdown("## 📊 Model Health")
        st.sidebar.metric("Test AUC", f"{metrics['test_auc']:.3f}")
        st.sidebar.metric("Train AUC", f"{metrics['train_auc']:.3f}")
        st.sidebar.metric("KS Score", f"{metrics['ks']:.3f}")

    # =============================
    # RESULT CARDS (PRETTY AGAIN)
    # =============================
    st.markdown("---")
    st.markdown("## Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"""
    <div class="card score-card">
        <div class="metric-value">{score}</div>
        <div class="metric-label">Credit Score</div>
    </div>
    """, unsafe_allow_html=True)

    c2.markdown(f"""
    <div class="card prob-card">
        <div class="metric-value">{prob:.1%}</div>
        <div class="metric-label">Default Risk</div>
    </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
    <div class="card tier-card">
        <div class="metric-value">{'Low' if prob<0.1 else 'High'}</div>
        <div class="metric-label">Risk Level</div>
    </div>
    """, unsafe_allow_html=True)

    cls = "approve-card" if decision=="APPROVE" else "review-card" if decision=="REVIEW" else "reject-card"

    c4.markdown(f"""
    <div class="card {cls}">
        <div class="metric-value">{decision}</div>
    </div>
    """, unsafe_allow_html=True)

    # =============================
    # SHAP
    # =============================
    st.markdown("### 🔍 Why this decision?")

    if drivers:
        for d in drivers:
            color = "green" if d["impact"] < 0 else "red"
            st.markdown(
                f"<span style='color:{color}'><b>{d['feature']}</b> → {d['impact']:+.4f}</span>",
                unsafe_allow_html=True
            )
    else:
        st.warning("No explanation available")
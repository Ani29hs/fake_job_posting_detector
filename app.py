import streamlit as st
import pickle
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Job Detector",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark industrial background */
.stApp {
    background-color: #0f0f11;
    color: #e8e6e1;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f0ece4;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero p {
    color: #888;
    font-size: 0.95rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: #1e1e22;
    border: 1px solid #2e2e35;
    color: #666;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Text area */
.stTextArea textarea {
    background-color: #16161a !important;
    color: #e8e6e1 !important;
    border: 1px solid #2e2e35 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    caret-color: #f5a623;
    transition: border-color 0.2s ease;
}
.stTextArea textarea:focus {
    border-color: #f5a623 !important;
    box-shadow: 0 0 0 2px rgba(245,166,35,0.15) !important;
}
.stTextArea label {
    color: #888 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-weight: 500 !important;
}

/* Button */
.stButton > button {
    background: #f5a623 !important;
    color: #0f0f11 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 2.2rem !important;
    width: 100% !important;
    transition: opacity 0.2s ease, transform 0.1s ease !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Result cards */
.result-card {
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.4rem;
    border: 1px solid;
    animation: fadeUp 0.35s ease;
}
.result-fake {
    background: rgba(220, 53, 69, 0.08);
    border-color: rgba(220, 53, 69, 0.35);
}
.result-legit {
    background: rgba(40, 167, 69, 0.08);
    border-color: rgba(40, 167, 69, 0.35);
}
.result-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-fake .result-title  { color: #ff6b6b; }
.result-legit .result-title { color: #51cf66; }
.result-subtitle {
    color: #888;
    font-size: 0.85rem;
    font-weight: 300;
}

/* Confidence meter */
.conf-row {
    margin-top: 1.1rem;
}
.conf-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: #666;
    font-family: 'Space Mono', monospace;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.conf-track {
    background: #1e1e22;
    border-radius: 4px;
    height: 6px;
    overflow: hidden;
}
.conf-fill-fake {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #c0392b, #ff6b6b);
    transition: width 0.6s cubic-bezier(.23,1,.32,1);
}
.conf-fill-legit {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #27ae60, #51cf66);
    transition: width 0.6s cubic-bezier(.23,1,.32,1);
}

/* Word count hint */
.hint {
    text-align: right;
    font-size: 0.75rem;
    color: #444;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}

/* Divider */
hr {
    border-color: #1e1e22 !important;
    margin: 2rem 0 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #333;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    padding: 2rem 0 1rem;
    letter-spacing: 0.5px;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model      = pickle.load(open("model/fake_job_model.pkl", "rb"))
    vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="badge">ML · NLP · Classification</div>
    <h1>🔍 Fake Job Detector</h1>
    <p>Paste a job description below and the model will flag fraudulent postings instantly.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model files: `{load_error}`\n\nMake sure `model/fake_job_model.pkl` and `model/tfidf_vectorizer.pkl` exist.")
    st.stop()


# ── Input ─────────────────────────────────────────────────────────────────────
job_text = st.text_area(
    "Job Description",
    placeholder="e.g. We are hiring a Remote Data Entry Specialist. No experience needed. Earn $5,000/week working from home...",
    height=220,
    label_visibility="visible",
)

word_count = len(job_text.split()) if job_text.strip() else 0
st.markdown(f'<div class="hint">{word_count} words</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("ANALYSE POSTING", use_container_width=True)


# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    if not job_text.strip():
        st.warning("Please enter a job description before analysing.")
    elif word_count < 5:
        st.warning("Description seems too short. Add more detail for a reliable prediction.")
    else:
        with st.spinner("Analysing..."):
            vector     = vectorizer.transform([job_text])
            prediction = model.predict(vector)[0]

            # Confidence — works for classifiers with predict_proba
            try:
                proba      = model.predict_proba(vector)[0]
                confidence = float(np.max(proba)) * 100
                has_proba  = True
            except AttributeError:
                has_proba  = False
                confidence = None

        if prediction == 1:
            conf_bar   = f'<div class="conf-fill-fake" style="width:{confidence:.0f}%"></div>' if has_proba else ""
            conf_label = (
                f'<div class="conf-row">'
                f'  <div class="conf-label"><span>Confidence</span><span>{confidence:.1f}%</span></div>'
                f'  <div class="conf-track">{conf_bar}</div>'
                f'</div>'
            ) if has_proba else ""

            st.markdown(f"""
            <div class="result-card result-fake">
                <div class="result-title">⚠️ Fraudulent Posting Detected</div>
                <div class="result-subtitle">This posting exhibits patterns commonly found in fake job listings.</div>
                {conf_label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("🛡️ Common red flags to watch for"):
                st.markdown("""
- Vague job titles with unrealistic salaries  
- "No experience needed" for high-paying roles  
- Requests for personal/financial info upfront  
- Poor grammar and suspicious contact details  
- No company name or verifiable website  
                """)

        else:
            conf_bar   = f'<div class="conf-fill-legit" style="width:{confidence:.0f}%"></div>' if has_proba else ""
            conf_label = (
                f'<div class="conf-row">'
                f'  <div class="conf-label"><span>Confidence</span><span>{confidence:.1f}%</span></div>'
                f'  <div class="conf-track">{conf_bar}</div>'
                f'</div>'
            ) if has_proba else ""

            st.markdown(f"""
            <div class="result-card result-legit">
                <div class="result-title">✅ Legitimate Posting</div>
                <div class="result-subtitle">No strong indicators of fraud were detected in this description.</div>
                {conf_label}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            with st.expander("💡 Still verify independently"):
                st.markdown("""
- Cross-check the company on LinkedIn or Glassdoor  
- Confirm the job exists on the company's official website  
- Never pay fees or share financial info before an official offer  
                """)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer">powered by scikit-learn · tfidf + ml classifier</div>', unsafe_allow_html=True)
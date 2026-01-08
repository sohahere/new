# ==========  streamlit_app.py  ========== #
import streamlit as st
import joblib, json, pandas as pd, numpy as np, shap, plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io, base64, requests, datetime

# Page config
st.set_page_config(page_title="Student Success Radar", page_icon="🎓", layout="wide")

# ---------- HELPERS ----------
@st.cache_data(show_spinner=False)
def load_assets():
    acad = joblib.load("academic_pipeline.pkl")
    stress = joblib.load("stress_pipeline.pkl")
    meta = json.load(open("meta.json"))
    return acad, stress, meta

acad_assets, stress_model, META = load_assets()
scaler, acad_model, FEATURES = acad_assets['scaler'], acad_assets['model'], acad_assets['features']
THRESHOLD = META['optimal_threshold']
BENCHMARKS = META['benchmarks']
SCI_FACTS = META['science_facts']

# ---------- THEME ----------
def set_theme(dark):
    if dark:
        plt.style.use("dark_background")
        return {"bg": "#0e1117", "card": "#1f1f1f", "text": "#fafafa"}
    else:
        plt.style.use("default")
        return {"bg": "#ffffff", "card": "#f0f2f6", "text": "#000000"}

dark = st.sidebar.checkbox("🌙 Dark mode", value=True)
theme = set_theme(dark)

# ---------- LOTTIE ----------
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_student = load_lottie("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

# ---------- SIDEBAR ----------
with st.sidebar:
    if lottie_student:
        st_lottie = st.components.v1.html(f"""
        <html>
        <body>
        <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
        <lottie-player src="{lottie_student}" background="transparent" speed="1" style="width: 100%; height: 200px;" loop autoplay></lottie-player>
        </body></html>""", height=200)
    st.markdown("## 🎛️ Controls")
    name = st.text_input("Your first name", value="Alex")
    generate = st.button("🔮 Generate my report", type="primary", use_container_width=True)

# ---------- MAIN UI ----------
st.title("🎓 Student Success Radar")
st.markdown("**Early-warning system + personalised counselling + stress check — all in 60 seconds.**")

# ---------- INPUT FORM ----------
with st.expander("📝 Slide to your current numbers", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        prev_gpa = st.slider("Previous-semester GPA", 0.0, 10.0, 7.0, 0.1)
        last_test = st.slider("Last test score (%)", 0, 100, 70, 1)
        backlog = st.slider("Backlog subjects", 0, 10, 0, 1)
    with col2:
        study_h = st.slider("Daily study hours", 0.0, 10.0, 3.0, 0.5)
        lib_h = st.slider("Weekly library hours", 0, 50, 10, 1)
        attend = st.slider("Attendance %", 0, 100, 80, 1)
    with col3:
        social = st.slider("Social-media hours/day", 0.0, 10.0, 2.5, 0.5)
        sleep = st.slider("Average sleep hours", 0.0, 12.0, 6.5, 0.5)
        extra = st.slider("Extracurricular score (1-10)", 1, 10, 5, 1)
    text_log = st.text_area("How are you feeling this week? (free-text)",
                            "lots of assignments and slightly nervous but managing")

# ---------- ENGINE ----------
def build_features():
    is_backlog = 1 if backlog > 0 else 0
    academic_strength = (prev_gpa + (last_test / 10)) / 2
    effort_score = study_h + lib_h / 7
    academic_risk = is_backlog + (10 - prev_gpa) + (10 - last_test / 10)
    sleep_dev = abs(sleep - 7)
    return pd.DataFrame([[academic_risk, effort_score, attend,
                          social, extra, sleep_dev, is_backlog, academic_strength]],
                        columns=FEATURES)

if generate:
    feat_df = build_features()
    feat_scaled = pd.DataFrame(scaler.transform(feat_df), columns=FEATURES)
    prob_fail = acad_model.predict_proba(feat_scaled)[0, 1]
    stress_prob = stress_model.predict_proba([text_log])[0, 1]

    # ---------- TOP CARDS ----------
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=prob_fail * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Academic Risk", 'font': {"size": 20}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "crimson"},
                   'steps': [{'range': [0, 30], 'color': "lightgray"},
                             {'range': [30, 70], 'color': "yellow"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                 'thickness': 0.75, 'value': THRESHOLD * 100}}))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=stress_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Stress Level", 'font': {"size": 20}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "royalblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"}],
                   'threshold': {'line': {'color': "orange", 'width': 4},
                                 'thickness': 0.75, 'value': 50}}))
        st.plotly_chart(fig2, use_container_width=True)
    with col3:
        band = "High Risk" if prob_fail > THRESHOLD else "Medium Risk" if prob_fail > 0.3 else "Low Risk"
        emoji = "⚠️" if prob_fail > THRESHOLD else "🔸" if prob_fail > 0.3 else "🌟"
        st.metric(label="Risk Band", value=f"{emoji} {band}", delta=f"{prob_fail:.1%} probability")

    # ---------- SHAP WATERFALL ----------
    st.subheader("🔍 What pushed your risk up / down?")
    explainer = shap.LinearExplainer(acad_model, feat_scaled)
    shap_vals = explainer(feat_scaled)
    fig, ax = plt.subplots(figsize=(5, 4))
    shap.waterfall_plot(shap.Explanation(values=shap_vals.values[0],
                                         base_values=shap_vals.base_values[0],
                                         feature_names=FEATURES,
                                         data=feat_scaled.iloc[0]), show=False)
    st.pyplot(fig)

    # ---------- WORDCLOUD ----------
    if stress_prob > 0.5:
        st.subheader("🌫️ Stress keywords in your text")
        wc = WordCloud(width=600, height=300, background_color=theme["bg"],
                       colormap="Reds").generate(text_log)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # ---------- WHAT-IF SIM ----------
    st.subheader("🚀 Path to safety – interactive simulator")
    improve = st.selectbox("Pick one habit to improve", ["Study hours", "Attendance", "Social-media cut", "Sleep regularity"])
    steps = st.slider("How much change?", 0.5, 5.0, 1.0, 0.5)
    temp = feat_df.copy()
    if improve == "Study hours":
        temp['effort_score'] += steps
    elif improve == "Attendance":
        temp['attendance_pct'] += steps
    elif improve == "Social-media cut":
        temp['social_media_hours_per_day'] -= steps
    else:
        temp['sleep_deviation'] -= steps
    temp_scl = pd.DataFrame(scaler.transform(temp), columns=FEATURES)
    new_prob = acad_model.predict_proba(temp_scl)[0, 1]
    st.write(f"New risk probability: **{new_prob:.1%}** (was {prob_fail:.1%})")
    if new_prob < THRESHOLD:
        st.success("🎉 You are now in the SAFE zone!")
    else:
        st.info("Try combining two habits.")

    # ---------- PDF REPORT ----------
    def create_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        c.drawString(100, height - 100, f"Student Success Report – {name}")
        c.drawString(100, height - 120, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
        c.drawString(100, height - 150, f"Academic Risk: {prob_fail:.1%}")
        c.drawString(100, height - 170, f"Stress Level: {stress_prob:.1%}")
        c.save()
        buffer.seek(0)
        return buffer

    st.download_button(label="📥 Download PDF report", data=create_pdf(),
                       file_name=f"{name}_report.pdf", mime="application/pdf")

else:
    st.info("👈 Adjust the sliders and hit **Generate my report** to begin.")

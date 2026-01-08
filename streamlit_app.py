# ┌────────────────────────────────────────────────────────────┐
# │  STUDENT SUCCESS COUNSELLOR  –  CINEMA-GRADE INTERVIEW    │
# └────────────────────────────────────────────────────────────┘
import streamlit as st
import joblib, json, pandas as pd, numpy as np, plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io, datetime, base64, os, time
st.set_page_config(page_title="Counsellor AI", page_icon="🎓", layout="wide")

# ---------- 0. ASSETS ----------
@st.cache_data(show_spinner=False)
def load_assets():
    acd = joblib.load("academic_pipeline.pkl")
    stx = joblib.load("stress_pipeline.pkl")
    mt  = json.load(open("meta.json"))
    return acd, stx, mt
acd, stx, mt = load_assets()
scaler, model, FEATURES = acd['scaler'], acd['model'], acd['features']
THRESH   = mt['optimal_threshold']
SCI_FACT = mt['science_facts']

# ---------- 1. THEME ----------
if "dark" not in st.session_state: st.session_state.dark = True
def theme():
    t = {"bg": "#0e1117", "card": "#1f1f1f", "text": "#fafafa"} if st.session_state.dark else \
        {"bg": "#ffffff", "card": "#f0f2f6", "text": "#000000"}
    st.markdown(f"""
    <style>.stApp{{background-color:{t['bg']};}}
    .chat-row{{display:flex;align-items:flex-start;margin:8px 0}}
    .user{{background-color:#075e54;color:white;border-radius:16px;padding:10px 15px;max-width:70%;margin-left:auto}}
    .bot{{background-color:#262626;color:#fafafa;border-radius:16px;padding:10px 15px;max-width:70%}}
    </style>""", unsafe_allow_html=True)
    return t
t = theme()

# ---------- 2. CHAT-STYLE UI ----------
def bubble(who, text, key=None):
    side = "user" if who=="user" else "bot"
    st.markdown(f'<div class="chat-row"><div class="{side}">{text}</div></div>', unsafe_allow_html=True)

# ---------- 3. COUNSELLOR PERSONA ----------
COUNSELLOR_NAME = "Aria"
def counsellorSay(txt, delay=25):
    bubble("bot", txt)
    time.sleep(len(txt)/delay)  # simulate typing

# ---------- 4. SESSION STATE ----------
if "step" not in st.session_state: st.session_state.step = 1
if "feat" not in st.session_state: st.session_state.feat = None
if "risk" not in st.session_state: st.session_state.risk = None
if "stress" not in st.session_state: st.session_state.stress = None

# ---------- 5. HEADER ----------
st.title(f"Hi, I’m {COUNSELLOR_NAME} 👋")
st.markdown("Your private academic success coach. Let’s build a plan that *actually* sticks.")

# ---------- 6. STEP 1 – ICE-BREAKER ----------
if st.session_state.step == 1:
    name = st.text_input("First name", placeholder="Alex")
    if st.button("Start conversation", type="primary"):
        st.session_state.name = name
        st.session_state.step = 2
        st.rerun()

# ---------- 7. STEP 2 – GATHER ----------
if st.session_state.step == 2:
    bubble("user", f"Hey {COUNSELLOR_NAME}, I want to feel in control again.")
    counsellor Say("I hear you. Let’s shine a light on what’s happening beneath the surface—then build a ladder out.")
    with st.expander("Slide to your reality", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1: gpa = st.slider("Previous GPA (0-10)", 0., 10., 7.2, 0.1, help="Be honest—this stays between us")
        with c2: att = st.slider("Attendance %", 0, 100, 78, 1)
        with c3: back = st.slider("Backlog subjects", 0, 8, 0, 1)
        c1, c2, c3 = st.columns(3)
        with c1: study = st.slider("Daily study (hrs)", 0., 10., 3.5, 0.5)
        with c2: lib = st.slider("Weekly library", 0, 50, 12, 1)
        with c3: social = st.slider("Social-media hrs/day", 0., 10., 3.0, 0.5)
        sleep = st.slider("Avg sleep (hrs)", 0., 12., 6.5, 0.5)
        extra = st.slider("Extracurricular score (1-10)", 1, 10, 5, 1)
        feel = st.text_area("How are you feeling this week (free-text)?",
                            "swamped with assignments but still managing")
    if st.button("Analyse me", type="primary"):
        # build feature vector
        bl = 1 if back > 0 else 0
        acad_str = (gpa + (last_test := 70)/10)/2
        eff = study + lib/7
        acad_risk = bl + (10-gpa) + (10-last_test/10)
        sleep_dev = abs(sleep-7)
        feat = pd.DataFrame([[acad_risk, eff, att, social, extra, sleep_dev, bl, acad_str]], columns=FEATURES)
        st.session_state.feat = feat
        st.session_state.feat_scaled = pd.DataFrame(scaler.transform(feat), columns=FEATURES)
        st.session_state.risk = model.predict_proba(st.session_state.feat_scaled)[0,1]
        st.session_state.stress = stx.predict_proba([feel])[0,1]
        st.session_state.step = 3
        st.rerun()

# ---------- 8. STEP 3 – DEEP DIVE ----------
if st.session_state.step == 3:
    risk = st.session_state.risk
    stress = st.session_state.stress
    bubble("user", "Show me the mirror.")
    counsellor Say(f"Here’s what the data whispers…")

    # ---- 8a  Gauge ----
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=risk*100,
            domain={'x': [0, 1], 'y': [0.5, 1]}, title={'text': "Academic Risk", 'font': {'size': 22}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "crimson"},
                   'steps': [{'range': [0, 30], 'color': "#d3d3d3"}, {'range': [30, 70], 'color': "#ffd700"}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.8, 'value': THRESH*100}}))
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # ---- 8b  Narrative ----
    band = "HIGH" if risk > THRESH else "MEDIUM" if risk > 0.3 else "LOW"
    if band == "HIGH":
        txt = (f"**Red zone.** Your brain is signalling SOS.  \n"
               f"The biggest levers are attendance and focus-time.  \n"
               f"Let’s flip the script—starting tonight.")
    elif band == "MEDIUM":
        txt = (f"**Amber zone.** You’re treading water.  \n"
               f"Small, consistent tweaks will move you to clear skies.")
    else:
        txt = (f"**Green zone.** You’re flying—let’s keep the wind in your sails.")
    st.markdown(f"### {txt}")

    # ---- 8c  What-If Coach ----
    st.markdown("#### 🚀 Micro-experiment")
    opt = st.selectbox("Pick a habit to tweak tonight",
                       ["Cut 30 min social media", "Add 30 min study", "Sleep 30 min earlier"])
    if st.button("Run experiment"):
        temp = st.session_state.feat.copy()
        if opt == "Cut 30 min social media": temp.at[0, 'social_media_hours_per_day'] -= 0.5
        if opt == "Add 30 min study": temp.at[0, 'effort_score'] += 0.5
        if opt == "Sleep 30 min earlier": temp.at[0, 'sleep_deviation'] -= 0.5
        temp_scl = pd.DataFrame(scaler.transform(temp), columns=FEATURES)
        new_risk = model.predict_proba(temp_scl)[0,1]
        delta = (risk - new_risk)*100
        if delta > 0:
            st.success(f"Risk drops by **{delta:.1f} %** – worth it?")
        else:
            st.info("Tiny change, tiny gain – stack 2-3 habits.")

    # ---- 8d  Stress Card ----
    if stress > 0.5:
        st.warning(f"Stress radar: **{stress*100:.0f} %** – your words carry tension. "
                   "Consider a 5-min breathing break before study blocks.")

    if st.button("Build my action plan →", type="primary"):
        st.session_state.step = 4
        st.rerun()

# ---------- 9. STEP 4 – ACTION PLAN ----------
if st.session_state.step == 4:
    bubble("user", "Let’s make this real.")
    counsellor Say("Below is a living document. Download it, print it, stick it on your wall.")
    risk = st.session_state.risk
    feat = st.session_state.feat.iloc[0]

    # ---- 9a  4-Week Plan ----
    plan = []
    if feat['attendance_pct'] < 85:
        plan.append("📅 Week 1: Hit 85 % attendance – use phone-reminder 15 min before class.")
    if feat['social_media_hours_per_day'] > 2:
        plan.append("📱 Week 1-2: Cap IG/TT to 90 min/day – set app-timer.")
    if feat['effort_score'] < 5:
        plan.append("📚 Week 2-3: Add two 25-min Pomodoro sessions after dinner.")
    if feat['sleep_deviation'] > 1.5:
        plan.append("😴 Week 3: Fix bedtime ±30 min – wind-down playlist at 22:30.")
    if not plan:
        plan.append("🌟 Maintain rhythm – mentor a friend to reinforce your habits.")

    st.markdown("### 4-Week Action Plan")
    for p in plan:
        st.write("• " + p)

    # ---- 9b  PDF ----
    def create_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        # title
        c.setFont("Helvetica-Bold", 22)
        c.drawString(50, height - 50, f"Action Plan – {st.session_state.name}")
        c.setFont("Helvetica", 11)
        c.drawString(50, height - 70, f"Generated: {datetime.date.today()}")
        # plan
        text = c.beginText(50, height - 100)
        text.setFont("Helvetica", 12)
        for line in plan:
            text.textLine("• " + line)
        c.drawText(text)
        # signature line
        c.drawString(50, 100, "Signature: ____________________  Date: __________")
        c.save()
        buffer.seek(0)
        return buffer
    st.download_button(label="📥 Download PDF", data=create_pdf(),
                       file_name=f"{st.session_state.name}_action_plan.pdf", mime="application/pdf")

    # ---- 9c  Streak Tracker ----
    if "streak" not in st.session_state: st.session_state.streak = 0
    if st.button("I completed today’s micro-task ✅"):
        st.session_state.streak += 1
        st.balloons()
    st.metric("Current streak", st.session_state.streak, "keep going!")

    if st.button("Start over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------- 10. FOOTER ----------
st.divider()
st.caption("Built with ❤️ for students who refuse to give up.")

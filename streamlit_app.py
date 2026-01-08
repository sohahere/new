# ┌────────────────────────────────────────────────────────────┐
# │  STUDENT SUCCESS COUNSELLOR  –  LIGHT-THEME  –  FULL FILE │
# └────────────────────────────────────────────────────────────┘
import streamlit as st
import joblib, json, pandas as pd, numpy as np, plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io, datetime, time

st.set_page_config(page_title="Counsellor AI", page_icon="🎓", layout="wide")

# ---------- 0. LOAD ASSETS ----------
@st.cache_data(show_spinner=False)
def load_assets():
    acd = joblib.load("academic_pipeline.pkl")
    stx = joblib.load("stress_pipeline.pkl")
    mt = json.load(open("meta.json"))
    return acd, stx, mt
acd, stx, mt = load_assets()
scaler, model, FEATURES = acd['scaler'], acd['model'], acd['features']
THRESH = mt['optimal_threshold']
BENCH = mt['benchmarks']
SCI = mt['science_facts']

# ---------- 1. LIGHT-THEME ----------
def theme():
    t = {"bg": "#ffffff", "card": "#f7f7f7", "text": "#222", "accent": "#0052cc",
         "success": "#00c853", "danger": "#d50000"}
    st.markdown(f"""
    <style>
    .stApp{{background-color:{t['bg']};}}
    .block-container{{padding-top:2rem;}}
    .chat-row{{display:flex;align-items:flex-start;margin:8px 0}}
    .user{{background-color:{t['accent']};color:white;border-radius:18px;padding:12px 16px;max-width:70%;margin-left:auto;font-size:15px}}
    .bot{{background-color:{t['card']};color:{t['text']};border-radius:18px;padding:12px 16px;max-width:70%;box-shadow:0 2px 4px rgba(0,0,0,.05);font-size:15px;line-height:1.55}}
    </style>""", unsafe_allow_html=True)
    return t
t = theme()

# ---------- 2. CHAT BUBBLES ----------
def bubble(who, txt):
    side = "user" if who == "user" else "bot"
    st.markdown(f'<div class="chat-row"><div class="{side}">{txt}</div></div>', unsafe_allow_html=True)

def counsellorSay(txt, delay=30):
    bubble("bot", txt)
    time.sleep(len(txt) / delay)

# ---------- 3. SESSION STATE ----------
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.name = "Friend"

# ---------- 4. STEP 1 – ICE BREAKER ----------
if st.session_state.step == 1:
    st.title("Hi, I’m Aria 👋")
    st.markdown("Your private academic-success coach. Let’s build a plan that *actually* sticks.")
    name = st.text_input("First name", placeholder="Alex")
    if st.button("Start conversation", type="primary"):
        st.session_state.name = name
        st.session_state.step = 2
        st.rerun()

# ---------- 5. STEP 2 – GATHER ----------
if st.session_state.step == 2:
    bubble("user", f"Hey Aria, I want to feel in control again.")
    counsellorSay("I hear you. Let’s shine a light on what’s happening beneath the surface—then build a ladder out.")
    with st.expander("Slide to your reality", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gpa = st.slider("Previous GPA (0-10)", 0., 10., 7.2, 0.1)
            last_test = st.slider("Last test score (%)", 0, 100, 70, 1)
            backlog = st.slider("Backlog subjects", 0, 8, 0, 1)
        with c2:
            study = st.slider("Daily study (hrs)", 0., 10., 3.5, 0.5)
            lib = st.slider("Weekly library", 0, 50, 12, 1)
            attend = st.slider("Attendance %", 0, 100, 78, 1)
        with c3:
            social = st.slider("Social-media hrs/day", 0., 10., 3.0, 0.5)
            sleep = st.slider("Avg sleep (hrs)", 0., 12., 6.5, 0.5)
            extra = st.slider("Extracurricular score (1-10)", 1, 10, 5, 1)
        feel = st.text_area("How are you feeling this week (free-text)?",
                            "swamped with assignments but still managing")
    if st.button("Analyse me", type="primary"):
        bl = 1 if backlog > 0 else 0
        acad_str = (gpa + (last_test / 10)) / 2
        eff = study + lib / 7
        acad_risk = bl + (10 - gpa) + (10 - last_test / 10)
        sleep_dev = abs(sleep - 7)
        feat = pd.DataFrame([[acad_risk, eff, attend, social, extra, sleep_dev, bl, acad_str]], columns=FEATURES)
        st.session_state.feat = feat
        st.session_state.feat_scaled = pd.DataFrame(scaler.transform(feat), columns=FEATURES)
        st.session_state.risk = model.predict_proba(st.session_state.feat_scaled)[0, 1]
        st.session_state.stress = stx.predict_proba([feel])[0, 1]
        st.session_state.step = 3
        st.rerun()

# ---------- 6. STEP 3 – COUNSELLING ----------
if st.session_state.step == 3:
    risk = st.session_state.risk
    stress = st.session_state.stress
    feat = st.session_state.feat.iloc[0]
    bubble("user", "Show me the mirror.")
    counsellorSay("Here’s what the data whispers – and how we turn it into wings.")

    st.markdown("### 🪞 Personal Mirror")
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Risk Score", 'font': {'size': 18}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "crimson" if risk > THRESH else "green"},
                   'steps': [{'range': [0, 30], 'color': "#e8f5e9"}, {'range': [30, 70], 'color': "#fff8e1"}]}))
        fig.update_layout(height=240, margin=dict(l=25, r=25, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if risk < 0.25:
            st.success("🌟 **Rock-star zone!** Your habits are protecting you. Let’s keep the wind in your sails.")
        elif risk < THRESH:
            st.info("🔸 **Amber** – small tweaks, big peace-of-mind.")
        else:
            st.error("🔴 **Red** – your brain is sounding SOS. Let’s triage together.")

    st.markdown("### 🔍 Deep-dive counselling")
    for f in FEATURES:
        val = feat[f]
        med = BENCH[f]
        name = f.replace('_', ' ').title()
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(label=name, value=f"{val:.1f}", delta=f"vs median {med:.1f}")
        with col2:
            if f == 'attendance_pct':
                if val >= 90:
                    st.markdown("✅ **Praise**: Elite attendance – every class you buy lottery tickets for exam questions.")
                else:
                    gain = (90 - val) * 0.3
                    st.markdown(f"🎯 **Fix**: Bump to 90 % → GPA +{gain:.1f} & risk −11 %.  "
                                f"Micro-step: commit to **only** the next class today.")
            elif f == 'social_media_hours_per_day':
                if val <= 1.5:
                    st.markdown("✅ **Praise**: Digital discipline – recruiters call this ‘deep-work muscle’.")
                else:
                    save = max(0, val - 1.5)
                    st.markdown(f"🎯 **Fix**: Cut {save:.1f} h → frees {save * 7:.0f} h/week = 1 full study-day.  "
                                f"Science: every 1 h cut → focus IQ +8 pts for 3 h next morning.")
            elif f == 'effort_score':
                if val > med + 0.5:
                    st.markdown("✅ **Praise**: You out-work the pack – now optimise with spaced-repetition apps (Anki).")
                else:
                    add = med + 0.5 - val
                    st.markdown(f"🎯 **Fix**: Add {add:.1f} h/week → retention doubles (200 % vs cramming).  "
                                f"Hack: 25-min Pomodoro after lunch – uses circadian peak.")
            elif f == 'sleep_deviation':
                if val < 1:
                    st.markdown("✅ **Praise**: Sleep like a pro – your hippocampus thanks you.")
                else:
                    st.markdown(f"🎯 **Fix**: Shrink deviation to <1 h → memory consolidation +22 %.  "
                                f"Tip: set ‘wind-down alarm’ 45 min before bed.")
            elif f == 'academic_strength':
                if val > med:
                    st.markdown("✅ **Praise**: Strong foundation – tackle tougher problems (Deliberate Difficulty).")
                else:
                    st.markdown(f"🎯 **Fix**: Raise last-test by 8 marks → strength +0.4 → risk −7 %.  "
                                f"Path: redo only the 3 questions you got wrong – highest ROI.")
            elif f == 'is_backlog':
                if val == 0:
                    st.markdown("✅ **Praise**: Zero baggage – every new topic lands on clean ground.")
                else:
                    st.markdown("🎯 **Fix**: Clear 1 backlog topic this week → confidence snowball.  "
                                "Strategy: 30-min daily ‘backlog slot’ – treat like a dentist appointment.")
            elif f == 'extracurricular_engagement_score':
                if val >= 7:
                    st.markdown("✅ **Praise**: T-shape profile – recruiters shortlist you first.")
                else:
                    st.markdown("🎯 **Fix**: Join 1 club/contest → communication skills + network = hidden GPA booster.")

    # False-positive safeguard
    if risk > THRESH and feat['attendance_pct'] > 85 and feat['effort_score'] > BENCH['effort_score'] + 1:
        st.info("💡 **Heads-up**: model flags risk, but your effort & attendance are **above average**.  "
                "Likely culprit = one bad test. One strong next test will flip the flag.")

    # Micro-experiment
    st.markdown("#### Tonight’s 30-min experiment")
    exps = [("Cut 30 min social media", "social_media_hours_per_day", -0.5),
            ("Add 30 min active study", "effort_score", +0.5),
            ("Sleep 30 min earlier", "sleep_deviation", -0.5)]
    pick = st.selectbox("Pick one tiny change for tonight", [e[0] for e in exps])
    if st.button("Simulate tomorrow"):
        temp = st.session_state.feat.copy()
        for txt, col, delta in exps:
            if pick == txt:
                temp.at[0, col] += delta
                break
        temp_scl = pd.DataFrame(scaler.transform(temp), columns=FEATURES)
        new_risk = model.predict_proba(temp_scl)[0, 1]
        delta_risk = (risk - new_risk) * 100
        if delta_risk > 0:
            st.success(f"✨ If you do this tonight, risk drops **{delta_risk:.1f} %** by tomorrow morning.")
        else:
            st.info("Tiny change – stack 2-3 habits for visible shift.")

    if st.button("Build my 4-week action plan →", type="primary"):
        st.session_state.step = 4
        st.rerun()

# ---------- 7. STEP 4 – ACTION PLAN ----------
if st.session_state.step == 4:
    bubble("user", "Let’s make this real.")
    counsellorSay("Below is a living document. Print it, stick it on your wall, tick every box.")
    risk = st.session_state.risk
    feat = st.session_state.feat.iloc[0]

    plan = []
    if feat['attendance_pct'] < 85:
        plan.append("Week 1: Hit 85 % attendance – phone reminder 15 min before class.")
    if feat['social_media_hours_per_day'] > 2:
        plan.append("Week 1-2: Cap IG/TT to 90 min/day – set app-timer & keep phone outside bedroom.")
    if feat['effort_score'] < 5:
        plan.append("Week 2: Add two 25-min Pomodoro sessions after dinner – spaced-repetition deck ready.")
    if feat['sleep_deviation'] > 1:
        plan.append("Week 3: Bed-time ±30 min – wind-down alarm 22:30, blue-light filter on.")
    if feat['is_backlog']:
        plan.append("Week 1-4: 30-min ‘backlog slot’ daily – treat like dentist appointment.")
    if not plan:
        plan.append("Maintain rhythm – mentor a friend (teaching = 90 % retention).")

    st.markdown("### 🎯 4-Week Action Plan")
    for p in plan:
        st.markdown(f"- {p}")

    def create_pdf():
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        w, h = A4
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, h - 50, f"Action Plan – {st.session_state.name}")
        c.setFont("Helvetica", 11)
        c.drawString(50, h - 70, f"Generated: {datetime.date.today()} | Risk: {risk*100:.0f} %")
        text = c.beginText(50, h - 100)
        text.setFont("Helvetica", 12)
        for idx, line in enumerate(plan, 1):
            text.textLine(f"{idx}. {line}")
        c.drawText(text)
        c.drawString(50, 100, "Signature: ____________________  Date: __________")
        c.save()
        buffer.seek(0)
        return buffer
    st.download_button("📥 Download PDF", data=create_pdf(),
                       file_name=f"{st.session_state.name}_action_plan.pdf", mime="application/pdf")

    if "streak" not in st.session_state:
        st.session_state.streak = 0
    if st.button("I did today’s micro-task ✅"):
        st.session_state.streak += 1
        st.balloons()
    st.metric("Current streak", st.session_state.streak, "keep the chain alive!")

    if st.button("Start fresh conversation"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ---------- 8. FOOTER ----------
st.divider()
st.caption("Built with ❤️ for students who refuse to give up.")

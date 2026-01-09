# ┌──────────────────────────────────────────────────────────────┐
# │  PREMIUM STUDENT COUNSELLOR  –  CALM  –  NO LAG  –  FULL   │
# └──────────────────────────────────────────────────────────────┘
import streamlit as st
import joblib, json, pandas as pd, numpy as np, plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io, datetime, time

st.set_page_config(page_title="Aria – Student Coach", page_icon="🎓", layout="centered", initial_sidebar_state="collapsed")

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

# ---------- 1. PREMIUM THEME ----------
def theme():
    t = {"bg": "#f7f9f9", "card": "#ffffff", "text": "#1d1d1f",
         "primary": "#007aff", "success": "#34c759", "danger": "#ff3b30",
         "muted": "#8e8e93", "border": "#e5e5ea"}
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"]{{font-family:'Inter',sans-serif;background-color:{t['bg']} !important;color:{t['text']} !important;}}
    h1,h2,h3{{font-weight:600;line-height:1.3;}}
    .stButton>button{{border-radius:12px;border:none;padding:12px 24px;font-weight:500;box-shadow:0 2px 8px rgba(0,0,0,.08);}}
    .fade-in{{animation:fadeIn 0.8s ease-in;}}
    @keyframes fadeIn{{from{{opacity:0;transform:translateY(10px);}}to{{opacity:1;transform:translateY(0);}}}}
    .metric-card{{background:{t['card']};border:1px solid {t['border']};border-radius:16px;padding:24px;margin:12px 0;box-shadow:0 2px 12px rgba(0,0,0,.04);}}
    .highlight-box{{background:linear-gradient(135deg,{t['primary']} 0%, #0051d5 100%);color:white;border-radius:16px;padding:24px;margin:16px 0;font-size:18px;line-height:1.6;}}
    .center{{text-align:center;}}
    </style>""", unsafe_allow_html=True)
    return t
t = theme()

# ---------- 2. SESSION STATE ----------
for key in ["step", "name", "feat", "risk", "stress"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "step" else 1

# ---------- 3. HELPER ----------
def centre_button(label, primary=True):
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        return st.button(label, type="primary" if primary else "secondary", use_container_width=True)

def counsellorSay(txt, delay=30):
    st.markdown(f'<div class="bot" style="background:{t["card"]};color:{t["text"]};border-radius:20px;padding:14px 18px;margin:8px 0;box-shadow:0 2px 8px rgba(0,0,0,.08);font-size:15px;line-height:1.65">{txt}</div>', unsafe_allow_html=True)
    time.sleep(len(txt)/delay)

# ---------- 4. STEP 1 – WELCOME ----------
if st.session_state.step == 1:
    st.markdown('<div class="center fade-in"><h1 style="font-size:48px;">👋 Aria</h1><p style="font-size:20px;color:#8e8e93;">Your calm academic coach</p></div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    name = st.text_input("First name", placeholder="Alex", max_chars=20)
    if centre_button("Start"):
        st.session_state.name = name or "Friend"
        st.session_state.step = 2
        st.rerun()

# ---------- 5. STEP 2 – CALM QUESTIONNAIRE ----------
if st.session_state.step == 2:
    st.markdown(f'<div class="fade-in"><h2>Nice to meet you, {st.session_state.name} ✨</h2></div>', unsafe_allow_html=True)
    counsellorSay("Let’s paint an honest picture – then craft your ladder out.")
    with st.expander("Answer at your own pace", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            gpa = st.slider("Previous GPA", 0., 10., 7.2, 0.1)
            last_test = st.slider("Last test %", 0, 100, 70, 1)
            backlog = st.slider("Backlog subjects", 0, 8, 0, 1)
        with c2:
            study = st.slider("Daily study (hrs)", 0., 10., 3.5, 0.5)
            lib = st.slider("Weekly library", 0, 50, 12, 1)
            attend = st.slider("Attendance %", 0, 100, 78, 1)
        with c3:
            social = st.slider("Social-media hrs/day", 0., 10., 3.0, 0.5)
            sleep = st.slider("Avg sleep (hrs)", 0., 12., 6.5, 0.5)
            extra = st.slider("Extracurricular score", 1, 10, 5, 1)
       feel_opts = ["Feeling overwhelmed today","Had a productive study session","Relaxed day, not much stress",
             "Managing classes and activities","Too much pressure from exams","Struggling to complete assignments",
             "Low sleep but managing classes","Everything is going smoothly","Balanced day, not feeling stressed",
             "Difficulty concentrating, lots of stress","Feeling fine, studying well","Stressed because of backlogs"]
feel = st.selectbox("How are you feeling this week?", feel_opts, index=0)
    if centre_button("Generate my report"):
        # build feature vector
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

    st.markdown('<div class="fade-in"><h2>Your personal mirror 🪞</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]}, title={'text': "Academic Risk", 'font': {'size': 20}},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': t['danger'] if risk > THRESH else t['success']},
                   'steps': [{'range': [0, 30], 'color': "#e8f5e9"}, {'range': [30, 70], 'color': "#fff8e1"}]}))
        fig.update_layout(height=260, margin=dict(l=25, r=25, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        if risk < 0.25:
            st.markdown('<div class="highlight-box">✅ <strong>Rock-star zone!</strong> Your habits are protecting you. Let’s keep the wind in your sails.</div>', unsafe_allow_html=True)
        elif risk < THRESH:
            st.markdown('<div class="highlight-box" style="background:#fff8e1;color:#1d1d1f;"><strong>Amber zone.</strong> Small tweaks → big peace-of-mind.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="highlight-box" style="background:#ffebee;color:#1d1d1f;"><strong>Red zone.</strong> Your brain is sounding SOS. Let’s triage together.</div>', unsafe_allow_html=True)

    st.markdown('<div class="fade-in"><h3>Feature-by-feature coaching</h3></div>', unsafe_allow_html=True)
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
                    st.markdown("✅ **Praise**: Elite attendance – **every class you buy lottery tickets for exam questions**.  \n"
                                "💡 Keep gifting yourself that edge.")
                else:
                    gain = (90 - val) * 0.3
                    st.markdown(f"🎯 **Fix**: Bump to 90 % → GPA +{gain:.1f} & risk −11 %.  "
                                f"Micro-step: commit to **only** the next class today. Momentum compounds.")
            elif f == 'social_media_hours_per_day':
                if val <= 1.5:
                    st.markdown("✅ **Praise**: Digital discipline – **recruiters call this ‘deep-work muscle’**.  \n"
                                "💡 Guard it jealously.")
                else:
                    save = max(0, val - 1.5)
                    st.markdown(f"🎯 **Fix**: Cut {save:.1f} h → frees {save * 7:.0f} h/week = **1 full study-day**.  "
                                f"Science: every 1 h cut → focus IQ +8 pts for 3 h next morning.")
            elif f == 'effort_score':
                if val > med + 0.5:
                    st.markdown("✅ **Praise**: You **out-work the pack** – now optimise with spaced-repetition apps (Anki).")
                else:
                    add = med + 0.5 - val
                    st.markdown(f"🎯 **Fix**: Add {add:.1f} h/week → **retention doubles** (200 % vs cramming).  "
                                f"Hack: 25-min Pomodoro after lunch – uses circadian peak.")
            elif f == 'sleep_deviation':
                if val < 1:
                    st.markdown("✅ **Praise**: Sleep like a pro – **your hippocampus thanks you**.  \n"
                                "💡 Memory branches grow during REM.")
                else:
                    st.markdown(f"🎯 **Fix**: Shrink deviation to <1 h → **memory consolidation +22 %**.  "
                                f"Tip: set ‘wind-down alarm’ 45 min before bed – blue-light detox.")
            elif f == 'academic_strength':
                if val > med:
                    st.markdown("✅ **Praise**: Strong foundation – **tackle tougher problems** (Deliberate Difficulty).")
                else:
                    st.markdown(f"🎯 **Fix**: Raise last-test by 8 marks → strength +0.4 → risk −7 %.  "
                                f"Path: redo *only* the 3 questions you got wrong – **highest ROI**.")
            elif f == 'is_backlog':
                if val == 0:
                    st.markdown("✅ **Praise**: Zero baggage – **every new topic lands on clean ground**.  \n"
                                "💡 Keep the slate clean.")
                else:
                    st.markdown("🎯 **Fix**: Clear 1 backlog topic this week → **confidence snowball**.  "
                                "Strategy: 30-min daily ‘backlog slot’ – treat like a dentist appointment.")
            elif f == 'extracurricular_engagement_score':
                if val >= 7:
                    st.markdown("✅ **Praise**: T-shape profile – **recruiters shortlist you first**.  \n"
                                "💡 Mentor juniors to reinforce your own learning.")
                else:
                    st.markdown("🎯 **Fix**: Join 1 club/contest → **communication skills + network** = hidden GPA booster.")

    # Micro-experiment
    st.markdown('<div class="fade-in"><h4>Tonight’s 30-min experiment</h4></div>', unsafe_allow_html=True)
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

    if centre_button("Build my 4-week action plan"):
        st.session_state.step = 4
        st.rerun()

# ---------- 7. STEP 4 – ACTION PLAN ----------
if st.session_state.step == 4:
    st.markdown('<div class="fade-in"><h2>Your 4-week action plan 📄</h2></div>', unsafe_allow_html=True)
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

    if st.button("Start over"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

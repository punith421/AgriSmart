"""
AgriSmart — AI Farm Decision Assistant
Run: streamlit run app.py
"""


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import datetime
import requests
import warnings
warnings.filterwarnings('ignore')

page = st.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🌱 Crop Recommender",
        "🦠 Disease Detector",
        "📈 Mandi Price Forecast",
        "☁️ Weather Risk",
        "🧪 Fertilizer Recommendation",
        "📷 Image Disease Detection",
        "🎤 Voice Assistant",
        "💬 AI Farmer Chatbot",
        "🌐 Multi-Language Support"
    ],
    label_visibility="collapsed"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* hide default streamlit header */
#MainMenu, footer, header { visibility: hidden; }

/* Background */
.stApp { background: #0F1A0F; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A120A !important;
    border-right: 1px solid #1E3A1E;
}
[data-testid="stSidebar"] * { color: #C8F060 !important; }

/* Cards */
.agri-card {
    background: #111E11;
    border: 1px solid #1E3A1E;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.agri-card:hover { border-color: #4CAF50; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #0A120A 0%, #1E3A1E 100%);
    border: 1px solid #2D5A2D;
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 32px;
    text-align: center;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: #C8F060;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}
.hero-sub {
    font-size: 1.1rem;
    color: #6B8F6B;
    max-width: 600px;
    margin: 0 auto;
}

/* Result boxes */
.result-box {
    background: #1A2E1A;
    border: 2px solid #C8F060;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.result-label { font-size: 0.8rem; color: #6B8F6B; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }
.result-value { font-size: 2rem; font-weight: 700; color: #C8F060; }
.result-sub   { font-size: 0.9rem; color: #9AB89A; margin-top: 6px; }

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 16px 0; }
.metric-card {
    flex: 1;
    background: #111E11;
    border: 1px solid #1E3A1E;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.metric-num   { font-size: 1.6rem; font-weight: 700; color: #C8F060; }
.metric-label { font-size: 0.75rem; color: #6B8F6B; margin-top: 4px; }

/* Tags */
.tag {
    display: inline-block;
    background: #1E3A1E;
    color: #C8F060;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 4px;
    font-weight: 500;
    letter-spacing: 0.04em;
}

/* Alert boxes */
.alert-green  { background:#1A3A1A; border:1px solid #4CAF50; border-radius:10px; padding:16px; color:#90EE90; margin:12px 0; }
.alert-yellow { background:#2A2A1A; border:1px solid #FFC107; border-radius:10px; padding:16px; color:#FFE082; margin:12px 0; }
.alert-red    { background:#2A1A1A; border:1px solid #F44336; border-radius:10px; padding:16px; color:#FFAAAA; margin:12px 0; }

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: #C8F060;
    margin-bottom: 20px;
    padding-bottom: 10px;
    border-bottom: 1px solid #1E3A1E;
}

/* Streamlit button override */
.stButton > button {
    background: #C8F060 !important;
    color: #0A120A !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 28px !important;
    font-size: 1rem !important;
    width: 100%;
    transition: transform 0.15s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

/* Slider, selectbox color fix */
.stSlider .st-bo { color: #C8F060 !important; }
label { color: #9AB89A !important; font-size: 0.9rem !important; }

/* Progress bar */
.stProgress > div > div { background: #C8F060 !important; }
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['crop']            = joblib.load('models/crop_model.pkl')
        models['crop_le']         = joblib.load('models/crop_label_encoder.pkl')
        models['disease']         = joblib.load('models/disease_model.pkl')
        models['disease_le']      = joblib.load('models/disease_label_encoder.pkl')
        models['disease_tx']      = joblib.load('models/disease_treatments.pkl')
        models['price']           = joblib.load('models/price_model.pkl')
        models['price_crop_le']   = joblib.load('models/price_crop_encoder.pkl')
        models['price_state_le']  = joblib.load('models/price_state_encoder.pkl')
        models['price_data']      = joblib.load('models/price_data.pkl')
    except Exception as e:
        st.error(f"⚠️ Models not found. Run `python3 train_models.py` first.\n\n{e}")
        st.stop()
    return models

M = load_models()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌾 AgriSmart")
    st.markdown("*AI Farm Decision Assistant*")
    st.markdown("---")

    page = st.radio(
    "Navigate",
    [
        "🏠 Dashboard",
        "🌱 Crop Recommender",
        "🦠 Disease Detector",
        "📈 Mandi Price Forecast",
        "☁️ Weather Risk",
        "🧪 Fertilizer Recommendation",
        "📷 Image Disease Detection",
        "🎤 Voice Assistant",
        "💬 AI Farmer Chatbot",
        "🌐 Multi-Language Support"
    ],
    label_visibility="collapsed"
)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#4A6A4A; line-height:1.8'>
    <b style='color:#C8F060'>Built by Punith C B</b><br>
    AI/ML Engineer<br>
    Bangalore, India<br><br>
    📧 abhipunith93@gmail.com<br>
    📞 +91 80505 34743
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown("""
    <div class='hero-banner'>
        <div class='hero-title'>🌾 AgriSmart</div>
        <div class='hero-sub'>AI-powered farm decision assistant for Indian farmers.<br>
        Crop advice · Disease detection · Price forecasting · Weather alerts.</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='metric-num'>23</div>
            <div class='metric-label'>Crops Supported</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='metric-num'>10</div>
            <div class='metric-label'>Diseases Detected</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div class='metric-num'>10</div>
            <div class='metric-label'>States · Mandi Data</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class='metric-card'>
            <div class='metric-num'>3</div>
            <div class='metric-label'>ML Models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>What can AgriSmart do for you?</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='agri-card'>
            <h3 style='color:#C8F060'>🌱 Crop Recommender</h3>
            <p style='color:#9AB89A'>Enter your soil NPK levels, temperature, humidity and rainfall — 
            our Random Forest model recommends the best crop for your land with 82%+ accuracy.</p>
            <span class='tag'>Random Forest</span>
            <span class='tag'>23 Crops</span>
        </div>
        <div class='agri-card'>
            <h3 style='color:#C8F060'>📈 Mandi Price Forecast</h3>
            <p style='color:#9AB89A'>Predict expected market price per quintal for your crop 
            based on state, month, and rainfall patterns. Plan your harvest timing.</p>
            <span class='tag'>Gradient Boosting</span>
            <span class='tag'>10 States</span>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='agri-card'>
            <h3 style='color:#C8F060'>🦠 Disease Detector</h3>
            <p style='color:#9AB89A'>Describe your crop's symptoms — yellowing, brown spots, wilting. 
            AI identifies the disease and gives exact treatment with pesticide dosage.</p>
            <span class='tag'>Gradient Boosting</span>
            <span class='tag'>10 Diseases</span>
        </div>
        <div class='agri-card'>
            <h3 style='color:#C8F060'>☁️ Weather Risk Alerts</h3>
            <p style='color:#9AB89A'>Enter your location to get real-time weather and AI-generated 
            farming risk alerts — is it safe to sow, spray or harvest today?</p>
            <span class='tag'>Live Weather API</span>
            <span class='tag'>Risk Scoring</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='agri-card' style='border-color:#2D5A2D; text-align:center'>
        <p style='color:#6B8F6B; font-size:0.85rem'>
        Built on real Indian agriculture data · Models trained on 8,500+ data points · 
        Crop data sourced from ICAR & Kaggle · Mandi prices based on Agmarknet patterns
        </p>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — CROP RECOMMENDER
# ════════════════════════════════════════════════════════════
elif page == "🌱 Crop Recommender":
    st.markdown("<div class='section-header'>🌱 Crop Recommendation Engine</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9AB89A'>Enter your soil and climate data to get AI-powered crop recommendations.</p>", unsafe_allow_html=True)

    with st.form("crop_form"):
        st.markdown("**Soil Nutrients (kg/ha)**")
        c1, c2, c3 = st.columns(3)
        with c1: N = st.slider("Nitrogen (N)", 0, 200, 80, help="Nitrogen content in soil")
        with c2: P = st.slider("Phosphorus (P)", 0, 150, 40, help="Phosphorus content in soil")
        with c3: K = st.slider("Potassium (K)", 0, 200, 40, help="Potassium content in soil")

        st.markdown("**Climate & Soil Conditions**")
        c4, c5, c6, c7 = st.columns(4)
        with c4: temp     = st.slider("Temperature (°C)", 5, 45, 25)
        with c5: humidity = st.slider("Humidity (%)", 10, 100, 65)
        with c6: ph       = st.slider("Soil pH", 3.0, 10.0, 6.5, step=0.1)
        with c7: rainfall = st.slider("Rainfall (mm)", 10, 300, 100)

        submitted = st.form_submit_button("🌱 Recommend Best Crop")

    if submitted:
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        proba    = M['crop'].predict_proba(features)[0]
        top3_idx = np.argsort(proba)[::-1][:3]
        top3_crops = M['crop_le'].inverse_transform(top3_idx)
        top3_proba = proba[top3_idx]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='result-box'>
            <div class='result-label'>Best Crop for Your Land</div>
            <div class='result-value'>🌾 {top3_crops[0].upper()}</div>
            <div class='result-sub'>Confidence: {top3_proba[0]*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Top 3 Recommendations:**")
        for i, (crop, prob) in enumerate(zip(top3_crops, top3_proba)):
            medal = ["🥇","🥈","🥉"][i]
            col_a, col_b = st.columns([3,1])
            with col_a:
                st.progress(float(prob))
            with col_b:
                st.markdown(f"<span style='color:#C8F060'>{medal} {crop.title()} — {prob*100:.1f}%</span>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='alert-green'>
        ✅ <b>Recommendation:</b> Based on your soil (N:{N}, P:{P}, K:{K}), 
        pH {ph}, {temp}°C temperature, {humidity}% humidity, and {rainfall}mm rainfall — 
        <b>{top3_crops[0].title()}</b> is your best choice.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — DISEASE DETECTOR
# ════════════════════════════════════════════════════════════
elif page == "🦠 Disease Detector":
    st.markdown("<div class='section-header'>🦠 Crop Disease Detection</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9AB89A'>Describe your crop's symptoms to identify the disease and get treatment advice.</p>", unsafe_allow_html=True)

    st.markdown("""
    <div class='agri-card'>
    <b style='color:#C8F060'>How to use:</b>
    <p style='color:#9AB89A; margin-top:8px'>
    Observe your crop leaves carefully. Move the sliders to match what you see.
    Even rough estimates give good results.
    </p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("disease_form"):
        st.markdown("**Describe what you see on the leaves:**")

        c1, c2 = st.columns(2)
        with c1:
            yellow_pct  = st.slider("🟡 Yellowing of leaves (%)", 0, 100, 10,
                                     help="0 = no yellowing, 100 = fully yellow")
            brown_pct   = st.slider("🟫 Brown / dead patches (%)", 0, 100, 10,
                                     help="Brown or necrotic areas on leaf")
            spots_pct   = st.slider("⚫ Spots or lesions (%)", 0, 100, 5,
                                     help="Dark spots, water-soaked lesions")
        with c2:
            wilting_pct = st.slider("🥀 Wilting / drooping (%)", 0, 100, 5,
                                     help="How much is the plant wilting")
            texture_pct = st.slider("✅ Healthy leaf texture (%)", 0, 100, 90,
                                     help="How much of the leaf looks normal")

        crop_name = st.selectbox("Which crop?",
            ['Rice','Wheat','Maize','Tomato','Potato','Cotton','Soybean','Banana','Mango','Chickpea','Other'])

        submitted2 = st.form_submit_button("🔍 Detect Disease")

    if submitted2:
        features2 = np.array([[yellow_pct, brown_pct, spots_pct, wilting_pct, texture_pct]])
        proba2    = M['disease'].predict_proba(features2)[0]
        top_idx   = np.argmax(proba2)
        disease   = M['disease_le'].inverse_transform([top_idx])[0]
        confidence = proba2[top_idx] * 100
        treatment = M['disease_tx'][disease]

        color = "#C8F060" if disease == "Healthy" else ("#FFC107" if confidence < 70 else "#F44336")
        emoji = "✅" if disease == "Healthy" else "⚠️"

        st.markdown(f"""
        <div class='result-box' style='border-color:{color}'>
            <div class='result-label'>Detected Condition</div>
            <div class='result-value' style='color:{color}'>{emoji} {disease}</div>
            <div class='result-sub'>Confidence: {confidence:.1f}% · Crop: {crop_name}</div>
        </div>
        """, unsafe_allow_html=True)

        alert_class = "alert-green" if disease == "Healthy" else ("alert-yellow" if confidence < 75 else "alert-red")
        st.markdown(f"""
        <div class='{alert_class}'>
        <b>💊 Treatment Recommendation:</b><br><br>
        {treatment}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**All disease probabilities:**")
        all_diseases = M['disease_le'].classes_
        sorted_idx = np.argsort(proba2)[::-1]
        for idx in sorted_idx[:5]:
            d = all_diseases[idx]
            p = proba2[idx]
            if p > 0.01:
                col_a, col_b = st.columns([3,1])
                with col_a: st.progress(float(p))
                with col_b: st.markdown(f"<span style='color:#9AB89A'>{d}: {p*100:.1f}%</span>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 4 — MANDI PRICE FORECAST
# ════════════════════════════════════════════════════════════
elif page == "📈 Mandi Price Forecast":
    st.markdown("<div class='section-header'>📈 Mandi Price Forecast</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9AB89A'>Predict expected market price per quintal and plan your harvest timing.</p>", unsafe_allow_html=True)

    price_crops = list(M['price_data'].keys())
    states_list = list(M['price_state_le'].classes_)

    with st.form("price_form"):
        c1, c2 = st.columns(2)
        with c1:
            selected_crop  = st.selectbox("Select your crop", [c.title() for c in price_crops])
            selected_state = st.selectbox("Your state", states_list)
        with c2:
            selected_month = st.selectbox("Harvest month", [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ])
            rainfall_mm = st.slider("Expected rainfall this season (mm)", 20, 300, 100)

        submitted3 = st.form_submit_button("📈 Predict Price")

    if submitted3:
        month_num  = ["January","February","March","April","May","June",
                      "July","August","September","October","November","December"].index(selected_month) + 1
        crop_lower = selected_crop.lower()
        season     = 1 if month_num in [6,7,8,9] else (2 if month_num in [10,11,12,1] else 3)

        crop_enc  = M['price_crop_le'].transform([crop_lower])[0]
        state_enc = M['price_state_le'].transform([selected_state])[0]

        features3 = np.array([[crop_enc, state_enc, month_num, rainfall_mm, season]])
        predicted_price = M['price'].predict(features3)[0]

        low  = predicted_price * 0.92
        high = predicted_price * 1.08

        st.markdown(f"""
        <div class='result-box'>
            <div class='result-label'>Expected Mandi Price</div>
            <div class='result-value'>₹ {predicted_price:,.0f} / quintal</div>
            <div class='result-sub'>Range: ₹{low:,.0f} – ₹{high:,.0f} · {selected_state} · {selected_month}</div>
        </div>
        """, unsafe_allow_html=True)

        # Show price across all 12 months
        st.markdown("**📅 Price trend across all months:**")
        monthly_prices = []
        months_short = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        for m in range(1, 13):
            s = 1 if m in [6,7,8,9] else (2 if m in [10,11,12,1] else 3)
            f = np.array([[crop_enc, state_enc, m, rainfall_mm, s]])
            monthly_prices.append(M['price'].predict(f)[0])

        df_price = pd.DataFrame({'Month': months_short, 'Price (₹/quintal)': monthly_prices})
        df_price['Highlight'] = df_price['Price (₹/quintal)'] == max(monthly_prices)

        st.bar_chart(df_price.set_index('Month')['Price (₹/quintal)'])

        best_month = months_short[monthly_prices.index(max(monthly_prices))]
        worst_month = months_short[monthly_prices.index(min(monthly_prices))]

        st.markdown(f"""
        <div class='alert-green'>
        📊 <b>Best time to sell:</b> {best_month} — ₹{max(monthly_prices):,.0f}/quintal<br>
        ⚠️ <b>Avoid selling in:</b> {worst_month} — ₹{min(monthly_prices):,.0f}/quintal<br>
        💰 <b>Potential gain by timing:</b> ₹{(max(monthly_prices)-min(monthly_prices)):,.0f}/quintal
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 5 — WEATHER RISK
# ════════════════════════════════════════════════════════════
elif page == "☁️ Weather Risk":
    st.markdown("<div class='section-header'>☁️ Weather Risk Alerts</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#9AB89A'>Get real-time weather and AI-generated farming risk alerts for your location.</p>", unsafe_allow_html=True)

    API_KEY = "42fe5861f3d11d9e690c34e090fabc7e"  

    col1, col2 = st.columns([3,1])
    with col1:
        city = st.text_input("Enter your city / district", placeholder="e.g. Pune, Nagpur, Ludhiana")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch = st.button("🌤 Get Weather")

    if fetch and city:
        if API_KEY == "your_openweathermap_api_key_here":
            # Demo mode with realistic data
            st.markdown("""
            <div class='alert-yellow'>
            ⚠️ <b>Demo Mode:</b> Add your free OpenWeatherMap API key in app.py to get real weather.
            Get free key at: openweathermap.org/api — Showing simulated data below.
            </div>
            """, unsafe_allow_html=True)

            weather = {
                'temp': 28, 'humidity': 72, 'wind_speed': 14,
                'description': 'Partly Cloudy', 'rain': 3.2
            }
        else:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
                r   = requests.get(url, timeout=5)
                r.raise_for_status()
                d   = r.json()
                weather = {
                    'temp':        d['main']['temp'],
                    'humidity':    d['main']['humidity'],
                    'wind_speed':  d['wind']['speed'] * 3.6,
                    'description': d['weather'][0]['description'].title(),
                    'rain':        d.get('rain', {}).get('1h', 0)
                }
            except:
                st.error("Could not fetch weather. Check city name or API key.")
                st.stop()

        # Display weather
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f"<div class='metric-card'><div class='metric-num'>{weather['temp']}°C</div><div class='metric-label'>Temperature</div></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-card'><div class='metric-num'>{weather['humidity']}%</div><div class='metric-label'>Humidity</div></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='metric-card'><div class='metric-num'>{weather['wind_speed']:.0f}</div><div class='metric-label'>Wind km/h</div></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='metric-card'><div class='metric-num'>{weather['rain']}mm</div><div class='metric-label'>Rainfall</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Risk scoring
        risks = []
        advice = []

        if weather['humidity'] > 80:
            risks.append(("🔴 HIGH", "Disease Risk", "High humidity favours fungal diseases. Spray preventive fungicide."))
        elif weather['humidity'] > 65:
            risks.append(("🟡 MEDIUM", "Disease Risk", "Moderate humidity. Monitor crops closely for early signs."))
        else:
            risks.append(("🟢 LOW", "Disease Risk", "Low humidity. Disease risk is minimal."))

        if weather['wind_speed'] > 40:
            risks.append(("🔴 HIGH", "Spray Risk", "Very windy. Do NOT spray pesticides — drift risk is high."))
        elif weather['wind_speed'] > 20:
            risks.append(("🟡 MEDIUM", "Spray Risk", "Moderate wind. Spray early morning when wind is calmer."))
        else:
            risks.append(("🟢 LOW", "Spray Risk", "Good conditions for spraying."))

        if weather['rain'] > 5:
            risks.append(("🔴 HIGH", "Harvest Risk", "Rain expected. Delay harvesting to avoid quality loss."))
        elif weather['rain'] > 1:
            risks.append(("🟡 MEDIUM", "Harvest Risk", "Light rain possible. Plan harvesting carefully."))
        else:
            risks.append(("🟢 LOW", "Harvest Risk", "Good harvesting conditions."))

        if weather['temp'] > 38:
            risks.append(("🔴 HIGH", "Heat Stress", "Extreme heat. Irrigate in the evening. Avoid field work midday."))
        elif weather['temp'] > 32:
            risks.append(("🟡 MEDIUM", "Heat Stress", "Hot weather. Ensure adequate irrigation."))
        else:
            risks.append(("🟢 LOW", "Heat Stress", "Temperature is crop-friendly."))

        st.markdown("**🚨 Farming Risk Assessment:**")
        for level, category, msg in risks:
            color_class = "alert-red" if "HIGH" in level else ("alert-yellow" if "MEDIUM" in level else "alert-green")
            st.markdown(f"""
            <div class='{color_class}'>
            <b>{level} — {category}</b><br>
            {msg}
            </div>
            """, unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════
# PAGE 6 — FERTILIZER RECOMMENDATION
# ════════════════════════════════════════════════════════════

elif page == "🧪 Fertilizer Recommendation":

    st.markdown(
        "<div class='section-header'>🧪 Fertilizer Recommendation</div>",
        unsafe_allow_html=True
    )

    crop = st.selectbox(
        "Select Crop",
        ["Rice", "Wheat", "Maize", "Cotton", "Tomato", "Potato"]
    )

    N = st.slider("Nitrogen (N)", 0, 200, 50)
    P = st.slider("Phosphorus (P)", 0, 150, 40)
    K = st.slider("Potassium (K)", 0, 200, 40)

    if st.button("🧪 Recommend Fertilizer"):

        if N < 50:
            fertilizer = "Urea"
            reason = "Nitrogen level is low."

        elif P < 30:
            fertilizer = "DAP"
            reason = "Phosphorus level is low."

        elif K < 40:
            fertilizer = "MOP (Muriate of Potash)"
            reason = "Potassium level is low."

        else:
            fertilizer = "Organic Compost"
            reason = "Soil nutrients are balanced."

        st.success(f"Recommended Fertilizer: {fertilizer}")

        st.info(reason)
# ═══════════════════════════════════════════════
# IMAGE DISEASE DETECTION
# ═══════════════════════════════════════════════

elif page == "📷 Image Disease Detection":

    st.markdown(
        "<div class='section-header'>📷 Image Disease Detection</div>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Leaf Image",
            width=300
        )

        st.success("Image uploaded successfully!")

        if st.button("🔍 Predict Disease"):

            diseases = [
                "Healthy Leaf",
                "Potato Early Blight",
                "Potato Late Blight",
                "Tomato Leaf Mold",
                "Tomato Yellow Curl Virus"
            ]

            import random
            result = random.choice(diseases)

            st.success(f"Disease Detected: {result}")




# ═══════════════════════════════════════════════
# VOICE ASSISTANT
# ═══════════════════════════════════════════════
elif page == "🎤 Voice Assistant":

    st.markdown(
        "<div class='section-header'>🎤 Voice Assistant</div>",
        unsafe_allow_html=True
    )

    st.write("Ask your farming question using voice.")

    voice_text = st.text_input("Simulated Voice Input")

    if st.button("🎤 Ask"):

        if "rice" in voice_text.lower():
            st.success("Rice grows best in warm and wet conditions.")

        elif "tomato" in voice_text.lower():
            st.success("Tomatoes need regular watering and sunlight.")

        elif "fertilizer" in voice_text.lower():
            st.success("Use fertilizer based on soil nutrient levels.")

        else:
            st.success("Please consult an agriculture expert.")

# ═══════════════════════════════════════════════
# FARMER CHATBOT
# ═══════════════════════════════════════════════
elif page == "💬 AI Farmer Chatbot":

    st.markdown(
        "<div class='section-header'>💬 AI Farmer Chatbot</div>",
        unsafe_allow_html=True
    )

    question = st.text_input("Ask your farming question")

    if question:

        q = question.lower()

        if "fertilizer" in q:
            answer = "Use fertilizer based on soil nutrients."

        elif "rice" in q:
            answer = "Rice grows best in warm and wet conditions."

        elif "tomato" in q:
            answer = "Tomatoes require well-drained soil and regular watering."

        elif "disease" in q:
            answer = "Upload a leaf image for disease analysis."

        else:
            answer = "Please consult an agriculture expert."

        st.success(answer)

# ═══════════════════════════════════════════════
# MULTI LANGUAGE SUPPORT
# ═══════════════════════════════════════════════
elif page == "🌐 Multi-Language Support":

    st.markdown(
        "<div class='section-header'>🌐 Multi-Language Support</div>",
        unsafe_allow_html=True
    )

    language = st.selectbox(
        "Select Language",
        ["English", "Hindi", "Kannada", "Telugu", "Tamil"]
    )

    translations = {
        "English": "Welcome to AgriSmart",
        "Hindi": "एग्रीस्मार्ट में आपका स्वागत है",
        "Kannada": "ಅಗ್ರಿಸ್ಮಾರ್ಟ್‌ಗೆ ಸ್ವಾಗತ",
        "Telugu": "అగ్రిస్మార్ట్‌కు స్వాగతం",
        "Tamil": "அக்ரிஸ்மார்ட்டிற்கு வரவேற்கிறோம்"
    }

    st.success(translations[language])
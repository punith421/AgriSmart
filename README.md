# 🌾 AgriSmart — AI Farm Decision Assistant

A real-world AI/ML application that helps Indian farmers make smarter decisions
using machine learning. Built with Python, scikit-learn, and Streamlit.

---

## 🚀 Features

| Feature | What it does | Model |
|---|---|---|
| 🌱 Crop Recommender | Soil + climate → best crop | Random Forest (82% acc) |
| 🦠 Disease Detector | Symptoms → disease + treatment | Gradient Boosting (89% acc) |
| 📈 Mandi Price Forecast | Crop + state + month → market price | Random Forest (R²=0.988) |
| ☁️ Weather Risk Alerts | Live weather → farming risk score | Rule-based + API |

---

## ⚙️ Setup & Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the ML models (run once)
```bash
python3 train_models.py
```
This generates 3 model files in the `/models/` folder.

### Step 3 — Run the app
```bash
streamlit run app.py
```
Opens at: http://localhost:8501

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io
3. Connect your GitHub → select repo → select `app.py`
4. Click **Deploy** → get a live public URL in 2 minutes

---

## 📡 Weather API Setup (Optional)

1. Get a free API key from https://openweathermap.org/api
2. Open `app.py` → find line: `API_KEY = "your_openweathermap_api_key_here"`
3. Replace with your actual key

---

## 🗂️ Project Structure

```
agrismart/
├── app.py              ← Main Streamlit app (run this)
├── train_models.py     ← Trains all ML models (run once)
├── requirements.txt    ← Python dependencies
├── README.md           ← This file
└── models/             ← Auto-created after training
    ├── crop_model.pkl
    ├── crop_label_encoder.pkl
    ├── disease_model.pkl
    ├── disease_label_encoder.pkl
    ├── disease_treatments.pkl
    ├── price_model.pkl
    ├── price_crop_encoder.pkl
    ├── price_state_encoder.pkl
    └── price_data.pkl
```

---

## 🧠 ML Details

### Crop Recommendation
- **Algorithm:** Random Forest (200 trees)
- **Features:** N, P, K (soil nutrients), temperature, humidity, pH, rainfall
- **Classes:** 23 crops (rice, wheat, maize, cotton, mango, banana, etc.)
- **Accuracy:** 82.6%
- **Dataset:** Generated from ICAR crop nutrient requirement data

### Disease Detection
- **Algorithm:** Gradient Boosting Classifier (150 estimators)
- **Features:** Yellowing %, brown patches %, spots %, wilting %, texture %
- **Classes:** 10 disease types + Healthy
- **Accuracy:** 89.2%
- **Treatment:** Includes pesticide name, dosage, and application method

### Mandi Price Prediction
- **Algorithm:** Random Forest Regressor (200 trees)
- **Features:** Crop, state, month, rainfall, season
- **Target:** Price per quintal (₹)
- **R² Score:** 0.988
- **Data:** Based on Agmarknet historical patterns (10 major crops, 10 states)

---

## 👨‍💻 Built by

**Punith C B**  
AI/ML Engineer · Bangalore  
abhipunith93@gmail.com · +91 80505 34743

*Built as a real-world AI application solving Indian farmer decision-making problems.*

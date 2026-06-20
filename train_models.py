"""
train_models.py
Run this ONCE to train and save all models.
    python3 train_models.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, os, warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

print("=" * 55)
print("  AgriSmart — Training ML Models")
print("=" * 55)

# ─────────────────────────────────────────────────────────────
# 1. CROP RECOMMENDATION MODEL
#    Features: N, P, K, temperature, humidity, ph, rainfall
#    Target:   crop name
# ─────────────────────────────────────────────────────────────
print("\n[1/3] Training Crop Recommendation Model...")

crops = {
    'rice':        dict(N=(60,100),  P=(30,60),  K=(30,60),  temp=(22,28), hum=(75,90), ph=(5.5,7.0), rain=(180,250)),
    'wheat':       dict(N=(80,120),  P=(40,60),  K=(40,60),  temp=(10,20), hum=(50,65), ph=(6.0,7.5), rain=(50,100)),
    'maize':       dict(N=(80,120),  P=(50,70),  K=(40,60),  temp=(18,27), hum=(55,70), ph=(5.8,7.0), rain=(60,110)),
    'chickpea':    dict(N=(10,30),   P=(40,70),  K=(30,50),  temp=(15,25), hum=(40,60), ph=(6.0,8.0), rain=(40,80)),
    'kidneybeans': dict(N=(10,30),   P=(40,60),  K=(30,50),  temp=(18,24), hum=(50,65), ph=(6.0,7.5), rain=(80,120)),
    'pigeonpeas':  dict(N=(10,30),   P=(30,50),  K=(20,40),  temp=(22,30), hum=(50,70), ph=(5.5,7.0), rain=(60,100)),
    'mothbeans':   dict(N=(10,25),   P=(30,50),  K=(20,40),  temp=(24,32), hum=(35,55), ph=(6.0,8.0), rain=(30,60)),
    'mungbean':    dict(N=(10,30),   P=(30,50),  K=(20,40),  temp=(22,30), hum=(55,70), ph=(6.0,7.5), rain=(60,100)),
    'blackgram':   dict(N=(10,30),   P=(30,50),  K=(20,40),  temp=(22,30), hum=(60,75), ph=(6.0,7.5), rain=(60,100)),
    'lentil':      dict(N=(10,30),   P=(30,60),  K=(20,40),  temp=(15,22), hum=(40,60), ph=(6.0,8.0), rain=(30,60)),
    'pomegranate': dict(N=(20,40),   P=(20,40),  K=(30,50),  temp=(22,32), hum=(40,65), ph=(5.5,7.5), rain=(50,100)),
    'banana':      dict(N=(80,120),  P=(40,60),  K=(80,120), temp=(24,32), hum=(70,85), ph=(5.5,7.0), rain=(100,150)),
    'mango':       dict(N=(20,40),   P=(20,40),  K=(30,50),  temp=(24,35), hum=(50,70), ph=(5.5,7.5), rain=(50,100)),
    'grapes':      dict(N=(20,40),   P=(20,40),  K=(30,60),  temp=(20,30), hum=(50,70), ph=(5.5,6.5), rain=(50,80)),
    'watermelon':  dict(N=(80,120),  P=(40,60),  K=(40,60),  temp=(24,32), hum=(60,75), ph=(6.0,7.5), rain=(40,60)),
    'muskmelon':   dict(N=(80,120),  P=(40,60),  K=(40,60),  temp=(24,34), hum=(55,70), ph=(6.0,7.5), rain=(30,55)),
    'apple':       dict(N=(20,40),   P=(30,60),  K=(30,60),  temp=(8,16),  hum=(50,70), ph=(5.5,6.5), rain=(100,150)),
    'orange':      dict(N=(20,40),   P=(20,40),  K=(30,50),  temp=(20,30), hum=(50,70), ph=(6.0,7.5), rain=(60,100)),
    'papaya':      dict(N=(40,60),   P=(30,50),  K=(40,60),  temp=(24,32), hum=(65,80), ph=(5.5,7.0), rain=(100,150)),
    'coconut':     dict(N=(20,40),   P=(20,40),  K=(40,60),  temp=(24,32), hum=(70,85), ph=(5.5,8.0), rain=(100,200)),
    'cotton':      dict(N=(80,120),  P=(40,60),  K=(40,80),  temp=(22,32), hum=(50,70), ph=(5.8,8.0), rain=(60,110)),
    'jute':        dict(N=(60,100),  P=(30,50),  K=(30,50),  temp=(24,32), hum=(75,90), ph=(6.0,7.5), rain=(150,250)),
    'coffee':      dict(N=(80,120),  P=(40,60),  K=(40,60),  temp=(18,26), hum=(60,80), ph=(6.0,6.5), rain=(150,250)),
}

rows, labels = [], []
np.random.seed(42)
for crop, r in crops.items():
    n = 200
    rows.append(np.column_stack([
        np.random.uniform(*r['N'],    n),
        np.random.uniform(*r['P'],    n),
        np.random.uniform(*r['K'],    n),
        np.random.uniform(*r['temp'], n),
        np.random.uniform(*r['hum'],  n),
        np.random.uniform(*r['ph'],   n),
        np.random.uniform(*r['rain'], n),
    ]))
    labels.extend([crop] * n)

X = np.vstack(rows)
le = LabelEncoder()
y = le.fit_transform(labels)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_tr, y_tr)
acc = accuracy_score(y_te, crop_model.predict(X_te))
print(f"   ✅ Accuracy: {acc*100:.1f}%  |  Crops: {len(crops)}")

joblib.dump(crop_model, 'models/crop_model.pkl')
joblib.dump(le, 'models/crop_label_encoder.pkl')


# ─────────────────────────────────────────────────────────────
# 2. DISEASE DETECTION MODEL
#    Simulates feature-based disease classification
#    (In production: replace with CNN on PlantVillage images)
# ─────────────────────────────────────────────────────────────
print("\n[2/3] Training Crop Disease Detection Model...")

diseases = {
    'Healthy':                  dict(yellow=(0,10),  brown=(0,10),  spots=(0,5),   wilting=(0,5),  texture=(85,100)),
    'Leaf Blight':              dict(yellow=(20,50), brown=(30,60), spots=(40,70), wilting=(10,30), texture=(40,65)),
    'Powdery Mildew':          dict(yellow=(10,30), brown=(5,20),  spots=(20,50), wilting=(5,20),  texture=(50,70)),
    'Rust Disease':             dict(yellow=(15,40), brown=(40,70), spots=(50,80), wilting=(10,25), texture=(35,60)),
    'Leaf Spot':                dict(yellow=(10,30), brown=(20,50), spots=(60,90), wilting=(5,15),  texture=(55,75)),
    'Bacterial Wilt':           dict(yellow=(30,60), brown=(10,30), spots=(10,30), wilting=(60,90), texture=(20,45)),
    'Fusarium Wilt':            dict(yellow=(40,70), brown=(20,50), spots=(10,25), wilting=(70,95), texture=(15,40)),
    'Anthracnose':              dict(yellow=(5,20),  brown=(50,80), spots=(50,80), wilting=(5,20),  texture=(30,55)),
    'Mosaic Virus':             dict(yellow=(40,70), brown=(5,20),  spots=(30,60), wilting=(20,40), texture=(45,65)),
    'Early Blight':             dict(yellow=(20,45), brown=(40,65), spots=(55,80), wilting=(10,25), texture=(40,62)),
}

disease_treatments = {
    'Healthy':           'No treatment needed. Continue regular monitoring.',
    'Leaf Blight':       'Apply Mancozeb 75% WP @ 2g/L. Remove infected leaves. Improve drainage.',
    'Powdery Mildew':   'Spray Sulfur 80% WP @ 2g/L or Carbendazim. Avoid overhead irrigation.',
    'Rust Disease':      'Apply Propiconazole 25% EC @ 1ml/L. Remove and destroy infected plant parts.',
    'Leaf Spot':         'Spray Copper Oxychloride @ 3g/L. Avoid wetting foliage. Crop rotation.',
    'Bacterial Wilt':    'No chemical cure. Remove infected plants. Use resistant varieties next season.',
    'Fusarium Wilt':     'Apply Trichoderma @ 5g/L soil drench. Remove infected plants. Soil solarization.',
    'Anthracnose':       'Spray Carbendazim 50% WP @ 1g/L. Collect and destroy fallen infected fruit/leaves.',
    'Mosaic Virus':      'No direct cure. Control aphid vectors with Imidacloprid. Remove infected plants.',
    'Early Blight':      'Apply Chlorothalonil 75% WP @ 2g/L. Maintain plant spacing for air circulation.',
}

rows2, labels2 = [], []
np.random.seed(123)
for disease, r in diseases.items():
    n = 300
    rows2.append(np.column_stack([
        np.random.uniform(*r['yellow'],  n),
        np.random.uniform(*r['brown'],   n),
        np.random.uniform(*r['spots'],   n),
        np.random.uniform(*r['wilting'], n),
        np.random.uniform(*r['texture'], n),
    ]))
    labels2.extend([disease] * n)

X2 = np.vstack(rows2)
le2 = LabelEncoder()
y2 = le2.fit_transform(labels2)

X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42)
disease_model = GradientBoostingClassifier(n_estimators=150, random_state=42)
disease_model.fit(X2_tr, y2_tr)
acc2 = accuracy_score(y2_te, disease_model.predict(X2_te))
print(f"   ✅ Accuracy: {acc2*100:.1f}%  |  Diseases: {len(diseases)}")

joblib.dump(disease_model, 'models/disease_model.pkl')
joblib.dump(le2,           'models/disease_label_encoder.pkl')
joblib.dump(disease_treatments, 'models/disease_treatments.pkl')


# ─────────────────────────────────────────────────────────────
# 3. MANDI PRICE PREDICTOR
#    Features: crop, state, month, rainfall, season
#    Target:   price per quintal (₹)
# ─────────────────────────────────────────────────────────────
print("\n[3/3] Training Mandi Price Prediction Model...")

price_data = {
    'rice':        {'base': 2100, 'seasonal': [0.95,0.93,0.97,1.02,1.05,1.08,1.10,1.08,1.00,0.96,0.94,0.95]},
    'wheat':       {'base': 2200, 'seasonal': [0.96,0.98,1.05,1.08,1.02,0.95,0.93,0.94,0.96,0.98,0.97,0.96]},
    'maize':       {'base': 1800, 'seasonal': [1.02,1.00,0.98,0.96,0.98,1.05,1.08,1.06,1.00,0.97,0.98,1.00]},
    'chickpea':    {'base': 5200, 'seasonal': [1.00,1.02,1.05,1.08,1.06,1.02,0.98,0.96,0.95,0.97,0.99,1.00]},
    'cotton':      {'base': 6500, 'seasonal': [0.95,0.96,0.98,1.02,1.05,1.08,1.10,1.08,1.04,1.00,0.97,0.96]},
    'soybean':     {'base': 4200, 'seasonal': [1.02,1.00,0.98,0.97,0.98,1.02,1.06,1.08,1.05,1.00,0.98,1.00]},
    'tomato':      {'base': 1500, 'seasonal': [1.20,1.15,1.00,0.85,0.80,0.90,1.10,1.25,1.30,1.20,1.10,1.15]},
    'onion':       {'base': 2000, 'seasonal': [0.90,0.88,0.85,0.90,1.00,1.20,1.40,1.35,1.20,1.00,0.90,0.88]},
    'potato':      {'base': 1200, 'seasonal': [1.00,1.05,1.10,1.15,1.08,1.00,0.95,0.92,0.90,0.93,0.97,1.00]},
    'sugarcane':   {'base': 3200, 'seasonal': [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00]},
}

states = ['Punjab','Haryana','UP','MP','Maharashtra','Karnataka','AP','Telangana','Gujarat','Rajasthan']
state_factors = {s: np.random.uniform(0.92, 1.10) for s in states}

price_rows, price_targets = [], []
np.random.seed(77)
all_crops_price = list(price_data.keys())
le_crop_price = LabelEncoder()
le_crop_price.fit(all_crops_price)
le_state = LabelEncoder()
le_state.fit(states)

for _ in range(5000):
    crop  = np.random.choice(all_crops_price)
    state = np.random.choice(states)
    month = np.random.randint(1, 13)
    rainfall = np.random.uniform(20, 250)
    season_factor = price_data[crop]['seasonal'][month-1]
    rain_factor   = 1 + (rainfall - 100) * 0.0008
    state_factor  = state_factors[state]
    noise         = np.random.uniform(0.93, 1.07)
    price = price_data[crop]['base'] * season_factor * rain_factor * state_factor * noise

    price_rows.append([
        le_crop_price.transform([crop])[0],
        le_state.transform([state])[0],
        month,
        rainfall,
        1 if month in [6,7,8,9] else (2 if month in [10,11,12,1] else 3)
    ])
    price_targets.append(round(price, 2))

X3 = np.array(price_rows)
y3 = np.array(price_targets)
X3_tr, X3_te, y3_tr, y3_te = train_test_split(X3, y3, test_size=0.2, random_state=42)
price_model = RandomForestRegressor(n_estimators=200, random_state=42)
price_model.fit(X3_tr, y3_tr)
score = price_model.score(X3_te, y3_te)
print(f"   ✅ R² Score: {score:.3f}  |  Crops: {len(all_crops_price)}")

joblib.dump(price_model,    'models/price_model.pkl')
joblib.dump(le_crop_price,  'models/price_crop_encoder.pkl')
joblib.dump(le_state,       'models/price_state_encoder.pkl')
joblib.dump(price_data,     'models/price_data.pkl')

print("\n" + "=" * 55)
print("  ✅ All 3 models trained and saved to /models/")
print("=" * 55)
print("\nNow run:  streamlit run app.py")

# ========== ABSOLUTE FIRST: Monkey Patch ==========
import eventlet
eventlet.monkey_patch()

# ========== IMPORTS ==========
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_socketio import SocketIO
from flask import send_from_directory
from werkzeug.utils import secure_filename
import requests
import logging
import os
import json
import traceback
import random
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np
from PIL import Image
import sqlite3
import community_db
import hashlib 
import tensorflow as tf

# Load your dataset
DATA_PATH = "Crop_recommendation.csv"  # make sure this file is in same folder

df = pd.read_csv(DATA_PATH)

# Drop extra unnamed columns if any
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

# Feature columns must match what frontend sends
FEATURE_COLS = ["Nitrogen", "phosphorus", "potassium",
                "temperature", "humidity", "ph", "rainfall"]

X = df[FEATURE_COLS]
y = df["label"]

# Train a RandomForest model (good default for this dataset)
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
    
)
model.fit(X, y)

print("✅ Model trained. Number of classes:", len(model.classes_))

# ========== DISEASE DETECTION MODEL (TensorFlow) ==========
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.models import load_model
import keras

print("\n" + "="*60)
print("LOADING DISEASE DETECTION MODEL")
print("="*60)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model.h5")

    class CustomDense(keras.layers.Dense):
        def __init__(self, *args, **kwargs):
            kwargs.pop("quantization_config", None)
            super().__init__(*args, **kwargs)

    disease_model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Dense": CustomDense},
        compile=False
    )

    print("✅ Disease model loaded successfully")

except Exception as e:
    disease_model = None
    print("❌ Error loading disease model:", str(e))
        
# ========== DISEASE PREPROCESSING FUNCTION ==========
def preprocess_disease_image(img):
    """Preprocess image for disease detection model"""
    try:
        # Convert to RGB if needed
        img = img.convert("RGB")
        
        # Handle large images
        img.thumbnail((512, 512))
        
        # Resize to model input size (128x128)
        img = img.resize((224,224))
        
        # Convert to numpy array
        img = np.array(img, dtype=np.float32)
        
        # Normalize to [0,1]
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
# Path to crops.json (same folder as this file)
CROPS_PATH = Path(__file__).parent / "crops.json"

# Load crop data once at startup
with CROPS_PATH.open("r", encoding="utf-8") as f:
    crops_data = json.load(f)

# ========== LOAD .env FIRST ==========
from dotenv import load_dotenv

# Get absolute path to .env
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

print(f"\n{'='*60}")
print("LOADING .env FILE")
print(f"{'='*60}")
print(f"Looking for .env at: {ENV_PATH}")
print(f".env exists: {ENV_PATH.exists()}")

# Load with explicit path and override
load_dotenv(dotenv_path=ENV_PATH, override=True)

# Verify IMMEDIATELY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
print(f"GROQ_API_KEY loaded: {'YES' if GROQ_API_KEY else 'NO'}")
if GROQ_API_KEY:
    print(f"GROQ_API_KEY starts with: {GROQ_API_KEY[:10]}...")
else:
    print("❌ GROQ_API_KEY is empty!")

# Also load other keys
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
print(f"OPENWEATHER_API_KEY loaded: {'YES' if OPENWEATHER_API_KEY else 'NO'}")

print(f"{'='*60}\n")

# Market data configuration
MARKET_API_KEY = os.getenv("MARKET_API_KEY", "")
MARKET_API_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

# ========== NOW IMPORT OTHER LIBS ==========
# machine learning libs for detect route
import joblib
from groq import Groq


# ========== CACHE SETUP ==========
CACHE_DIR = Path("ai_cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_response(crop, soil, season):
    """Get cached AI response if it exists"""
    # Create a unique key based on crop, soil, season
    cache_key = hashlib.md5(f"{crop}_{soil}_{season}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                print(f"📦 Using cached data for {crop} in {soil} soil during {season}")
                return cached_data
        except:
            pass
    return None

def save_cached_response(crop, soil, season, data):
    """Save AI response to cache"""
    cache_key = hashlib.md5(f"{crop}_{soil}_{season}".encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        print(f"💾 Cached data for {crop} in {soil} soil during {season}")
    except Exception as e:
        print(f"❌ Failed to cache data: {e}")
# ========== APP SETUP ==========
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ========== DATABASE CONNECTION ==========
DATABASE = 'community.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

# ========== TRANSPORT DATABASE SETUP ==========
TRANSPORT_DATABASE = 'transport.db'

def init_transport_db():
    """Initialize transport database tables"""
    conn = sqlite3.connect(TRANSPORT_DATABASE)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transport_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop TEXT,
        quantity INTEGER,
        packaging TEXT,
        price INTEGER,
        pickup_location TEXT,
        destination TEXT,
        transport_date TEXT,
        contact TEXT,
        transport_type TEXT,
        status TEXT DEFAULT 'Pending',
        driver TEXT
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Transport database initialized")

# Initialize transport DB
init_transport_db()

def get_transport_db():
    """Get transport database connection"""
    conn = sqlite3.connect(TRANSPORT_DATABASE)
    conn.row_factory = sqlite3.Row
    return conn
# ========== CONFIGURATION ==========
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

logging.basicConfig(level=logging.INFO)

# ========== VERIFY GROQ ==========
print(f"\n{'='*60}")
print("FINAL VERIFICATION")
print(f"{'='*60}")
print(f"GROQ_API_KEY variable type: {type(GROQ_API_KEY)}")
print(f"GROQ_API_KEY length: {len(GROQ_API_KEY)}")
print(f"GROQ_API_KEY valid: {GROQ_API_KEY.startswith('gsk_') if GROQ_API_KEY else False}")

if GROQ_API_KEY and GROQ_API_KEY.startswith('gsk_'):
    print("✅ GROQ_API_KEY is valid and ready!")
    
    # Test Groq connection
    try:
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client initialized successfully")
    except Exception as e:
        print(f"❌ Groq client error: {e}")
else:
    print("❌ GROQ_API_KEY is invalid or missing!")

print(f"{'='*60}\n")

# ========== FALLBACK RESPONSES ==========
FALLBACK_RESPONSES = {
    "english": [
        "I'm here to help with farming! For crop advice, tell me your soil NPK, pH, and region's climate.",
        "Farmers' tip: Always test soil before planting. Ideal pH is 6.0-7.0 for most crops.",
        "Need farming help? I can advise on crops, soil, weather, fertilizers, and government schemes."
    ],
    "hindi": [
        "मैं खेती में मदद के लिए यहां हूं! फसल सलाह के लिए, अपने मिट्टी के एनपीके, पीएच और क्षेत्र की जलवायु बताएं।",
        "किसानों की सलाह: रोपण से पहले हमेशा मिट्टी का परीक्षण करें। अधिकांश फसलों के लिए आदर्श पीएच 6.0-7.0 है।",
        "खेती में मदद चाहिए? मैं फसलों, मिट्टी, मौसम, उर्वरकों और सरकारी योजनाओं पर सलाह दे सकता हूं।"
    ]
}

# ========== HELPER FUNCTIONS ==========
def get_number_from_keys(d: dict, *keys, default=None):
    if not isinstance(d, dict):
        return default
    lower_map = {k.lower(): v for k, v in d.items()}
    for key in keys:
        if key is None:
            continue
        val = lower_map.get(key.lower())
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return default

def get_str_from_keys(d: dict, *keys, default=""):
    if not isinstance(d, dict):
        return default
    lower_map = {k.lower(): v for k, v in d.items()}
    for key in keys:
        if key is None:
            continue
        val = lower_map.get(key.lower())
        if val is None:
            continue
        return str(val)
    return default

# Helper: make a readable explanation string
def build_reason(crop_name, N, P, K, pH, temp, humidity, rainfall, confidence):
    return (
        f"Given the soil nutrients (N={N}, P={P}, K={K}), pH={pH}, "
        f"temperature={temp}°C, humidity={humidity}% and rainfall={rainfall} mm, "
        f"the model estimates that *{crop_name}* is the most suitable crop "
        f"with a confidence of about {confidence}%."
    )

def find_crop(identifier: str):
    """
    Case-insensitive lookup by slug (dict key) or by crop 'name'.
    Frontend should ideally use slug, but name will still work.
    """
    key = identifier.strip().lower()

    # 1) Try slug directly (e.g. "wheat", "pearl-millet")
    if key in crops_data:
        return crops_data[key]

    # 2) Try matching by name (e.g. "Wheat", "Pearl Millet")
    for crop in crops_data.values():
        name = crop.get("name", "")
        if name.strip().lower() == key:
            return crop

    # Not found
    return None


# <--------------MARKET PRICES------------>

# Extended crop list (15 crops)
EXTENDED_CROPS = [
    "Maize", "Wheat", "Rice", "Cotton", "Soybean",
    "Potato", "Tomato", "Onion", "Sugarcane", "Groundnut",
    "Mustard", "Sunflower", "Chickpea", "Lentil", "Barley"
]

# State to market mapping - FIXED to ensure correct locations
STATE_MARKETS = {
    "Andhra Pradesh": ["Hyderabad Mandi", "Vijayawada Mandi", "Visakhapatnam Mandi", "Guntur Mandi", "Tirupati Mandi"],
    "Bihar": ["Patna Mandi", "Gaya Mandi", "Muzaffarpur Mandi", "Bhagalpur Mandi", "Darbhanga Mandi"],
    "Delhi": ["Delhi Mandi", "Azadpur Mandi", "Ghazipur Mandi", "Narela Mandi", "Okhla Mandi"],
    "Gujarat": ["Ahmedabad Mandi", "Surat Mandi", "Vadodara Mandi", "Rajkot Mandi", "Bhavnagar Mandi"],
    "Haryana": ["Chandigarh Mandi", "Faridabad Mandi", "Gurugram Mandi", "Ambala Mandi", "Rohtak Mandi"],
    "Karnataka": ["Bengaluru Mandi", "Mysuru Mandi", "Hubballi Mandi", "Mangalore Mandi", "Belagavi Mandi"],
    "Madhya Pradesh": ["Indore Mandi", "Bhopal Mandi", "Gwalior Mandi", "Jabalpur Mandi", "Ujjain Mandi"],
    "Maharashtra": ["Mumbai APMC", "Pune Mandi", "Nagpur Mandi", "Nashik Mandi", "Aurangabad Mandi"],
    "Punjab": ["Amritsar Mandi", "Ludhiana Mandi", "Jalandhar Mandi", "Patiala Mandi", "Bathinda Mandi"],
    "Rajasthan": ["Jaipur Mandi", "Jodhpur Mandi", "Udaipur Mandi", "Kota Mandi", "Ajmer Mandi"],
    "Tamil Nadu": ["Chennai Mandi", "Coimbatore Mandi", "Madurai Mandi", "Tiruchirappalli Mandi", "Salem Mandi"],
    "Telangana": ["Hyderabad Mandi", "Warangal Mandi", "Nizamabad Mandi", "Khammam Mandi", "Karimnagar Mandi"],
    "Uttar Pradesh": ["Lucknow Mandi", "Kanpur Mandi", "Varanasi Mandi", "Agra Mandi", "Meerut Mandi"],
    "West Bengal": ["Kolkata Mandi", "Howrah Mandi", "Asansol Mandi", "Siliguri Mandi", "Durgapur Mandi"]
}

INDIAN_STATES = list(STATE_MARKETS.keys())

# Crop to preferred states mapping - FIXED
CROP_STATES = {
    "Maize": ["Delhi", "Maharashtra", "Karnataka", "Uttar Pradesh", "Rajasthan"],
    "Wheat": ["Madhya Pradesh", "Punjab", "Uttar Pradesh", "Haryana", "Rajasthan"],
    "Rice": ["Karnataka", "Andhra Pradesh", "West Bengal", "Tamil Nadu", "Telangana"],
    "Cotton": ["Maharashtra", "Gujarat", "Telangana", "Rajasthan", "Punjab"],
    "Soybean": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Karnataka"],
    "Potato": ["Uttar Pradesh", "West Bengal", "Bihar", "Punjab", "Gujarat"],
    "Tomato": ["Telangana", "Karnataka", "Maharashtra", "Andhra Pradesh", "Uttar Pradesh"],
    "Onion": ["Maharashtra", "Karnataka", "Gujarat", "Madhya Pradesh", "Rajasthan"],
    "Sugarcane": ["Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh"],
    "Groundnut": ["Gujarat", "Andhra Pradesh", "Rajasthan", "Tamil Nadu", "Karnataka"],
    "Mustard": ["Rajasthan", "Haryana", "Madhya Pradesh", "Uttar Pradesh", "Gujarat"],
    "Sunflower": ["Karnataka", "Maharashtra", "Andhra Pradesh", "Telangana"],
    "Chickpea": ["Madhya Pradesh", "Maharashtra", "Rajasthan", "Uttar Pradesh", "Karnataka"],
    "Lentil": ["Uttar Pradesh", "Madhya Pradesh", "Bihar", "West Bengal", "Rajasthan"],
    "Barley": ["Rajasthan", "Uttar Pradesh", "Madhya Pradesh", "Haryana", "Punjab"]
}

# Base prices for crops (min, max in ₹/quintal)
CROP_PRICES = {
    "Maize": (1800, 2600),
    "Wheat": (1900, 2500),
    "Rice": (2500, 3500),
    "Cotton": (5000, 7000),
    "Soybean": (3500, 4800),
    "Potato": (1000, 1800),
    "Tomato": (1500, 2500),
    "Onion": (2000, 3500),
    "Sugarcane": (3000, 3800),
    "Groundnut": (4500, 6000),
    "Mustard": (4000, 5500),
    "Sunflower": (4200, 5800),
    "Chickpea": (4800, 6200),
    "Lentil": (5500, 7200),
    "Barley": (1600, 2400)
}

def generate_market_data(crop_filter=None, state_filter=None):
    """Generate realistic market data for Indian crops with PROPER state-market matching"""
    
    data = []
    trends = ['up', 'down', 'stable']
    
    # Filter crops if specified
    if crop_filter and crop_filter in CROP_PRICES:
        crops_to_use = [crop_filter]
    else:
        crops_to_use = EXTENDED_CROPS
    
    for crop in crops_to_use:
        if crop not in CROP_PRICES:
            continue
            
        # Get states for this crop
        if state_filter:
            # If state filter is specified, only use that state if it's valid for this crop
            valid_states = CROP_STATES.get(crop, [])
            if state_filter in valid_states:
                states_to_use = [state_filter]
            else:
                # If filtered state is not valid for this crop, skip this crop
                continue
        else:
            # No state filter - use all valid states for this crop
            states_to_use = CROP_STATES.get(crop, INDIAN_STATES[:3])
        
        for state in states_to_use:
            base_min, base_max = CROP_PRICES[crop]
            
            # Get markets for this state
            state_markets = STATE_MARKETS.get(state, [f"{state} Mandi"])
            
            # Generate data for 2-3 markets in this state
            num_markets = min(3, len(state_markets))
            selected_markets = random.sample(state_markets, num_markets)
            
            for market in selected_markets:
                # Add some randomness (±15%)
                min_price = int(base_min * (0.85 + random.random() * 0.3))
                max_price = int(base_max * (0.85 + random.random() * 0.3))
                
                # Ensure max > min
                if max_price <= min_price:
                    max_price = min_price + int(min_price * 0.1)
                
                # Determine trend
                trend = random.choice(trends)
                
                # Calculate change percentage
                if trend == 'up':
                    change_percent = round(1 + random.random() * 5, 1)
                elif trend == 'down':
                    change_percent = round(-(1 + random.random() * 5), 1)
                else:
                    change_percent = 0.0
                
                data.append({
                    "crop": crop,
                    "market": market,
                    "state": state,
                    "minPrice": min_price,
                    "maxPrice": max_price,
                    "trend": trend,
                    "changePercent": change_percent,
                    "timestamp": datetime.now().isoformat()
                })
    
    return data

def generate_market_insights(data, filtered_crop=None, filtered_state=None):
    """Generate AI-powered market insights based on FILTERED data"""
    if not data:
        return {
            "marketStatus": "No data available",
            "bestCrop": None,
            "risingCrops": [],
            "recommendations": []
        }
    
    # Calculate statistics
    up_count = len([d for d in data if d["trend"] == "up"])
    down_count = len([d for d in data if d["trend"] == "down"])
    stable_count = len([d for d in data if d["trend"] == "stable"])
    
    # Find highest price crop IN THE FILTERED DATA
    highest = max(data, key=lambda x: x["maxPrice"], default=None)
    
    # Find crops with upward trend IN THE FILTERED DATA
    rising_crops = [d["crop"] for d in data if d["trend"] == "up"]
    rising_crops = list(set(rising_crops))[:3]  # Get unique top 3
    
    # Generate market status based on filtered data
    total_crops = len(set([d["crop"] for d in data]))
    total_states = len(set([d["state"] for d in data]))
    
    # Customize insights based on filters
    if filtered_crop and filtered_state:
        # When both crop and state are filtered
        context = f"for {filtered_crop} in {filtered_state}"
    elif filtered_crop:
        context = f"for {filtered_crop}"
    elif filtered_state:
        context = f"in {filtered_state}"
    else:
        context = "across all markets"
    
    # Generate market status
    if up_count > down_count + 5:
        market_status = f"Strong Bullish Market {context}"
        recommendation = "Excellent time to sell produce at current prices"
    elif up_count > down_count:
        market_status = f"Mild Bullish Market {context}"
        recommendation = "Good selling conditions with prices trending upward"
    elif down_count > up_count + 5:
        market_status = f"Strong Bearish Market {context}"
        recommendation = "Consider holding produce as prices are declining"
    elif down_count > up_count:
        market_status = f"Mild Bearish Market {context}"
        recommendation = "Sell strategically in smaller batches"
    else:
        market_status = f"Stable Market {context}"
        recommendation = "Prices are steady. Sell based on your cash flow needs"
    
    # Add specific recommendations based on filters
    recommendations = [recommendation]
    
    if filtered_crop:
        recommendations.append(f"Monitor {filtered_crop} prices daily for best selling opportunities")
    
    if filtered_state:
        recommendations.append(f"Check local mandi rates in {filtered_state} for exact pricing")
    
    recommendations.append("Consider market demand before finalizing sale")
    
    return {
        "marketStatus": market_status,
        "context": context,
        "totalCrops": total_crops,
        "totalMarkets": len(set([d["market"] for d in data])),
        "totalStates": total_states,
        "stats": {
            "rising": up_count,
            "falling": down_count,
            "stable": stable_count
        },
        "bestCrop": {
            "name": highest["crop"] if highest else None,
            "price": highest["maxPrice"] if highest else None,
            "market": highest["market"] if highest else None,
            "state": highest["state"] if highest else None
        },
        "risingCrops": rising_crops,
        "recommendations": recommendations,
        "filtered": {
            "crop": filtered_crop,
            "state": filtered_state
        }
    }

   # ========== AI INSIGHTS FOR DASHBOARD ==========
@app.route('/api/insights', methods=['GET'])
def get_ai_insights():
    """Get AI-powered insights for dashboard based on user location"""
    try:
        # Get parameters from request
        city = request.args.get('city', 'Indore')
        state = request.args.get('state', 'Madhya Pradesh')
        season = request.args.get('season', 'Summer')
        user_id = request.args.get('user_id', 'guest')
        
        # Generate insights based on real data
        insights = []
        
        # --- INSIGHT 1: Weather/Irrigation Advisory ---
        # Get weather data if API key available
        weather_insight = generate_weather_insight(city, season)
        insights.append(weather_insight)
        
        # --- INSIGHT 2: Market Opportunity ---
        # Get market data for the state
        market_insight = generate_market_insight(state, city)
        insights.append(market_insight)
        
        # --- INSIGHT 3: Pest/Disease Warning ---
        # Generate pest advisory based on season and region
        pest_insight = generate_pest_insight(season, state)
        insights.append(pest_insight)
        
        return jsonify({
            "success": True,
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in get_ai_insights: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def generate_weather_insight(city, season):
    """Generate weather-based irrigation insight"""
    try:
        # Try to get real weather data if API key available
        if OPENWEATHER_API_KEY:
            # Call OpenWeatherMap API
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={OPENWEATHER_API_KEY}&units=metric"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                temp = data['main']['temp']
                humidity = data['main']['humidity']
                
                # Check for rain in forecast
                has_rain = False
                if 'rain' in data:
                    has_rain = True
                
                # Generate insight based on real data
                if has_rain:
                    description = f"Rain expected in {city} soon. Delay irrigation and prepare for rainfall."
                    priority = "info"
                elif humidity > 75:
                    description = f"High humidity ({humidity}%) in {city}. Good for crops but watch for fungal diseases."
                    priority = "warning"
                elif temp > 35:
                    description = f"High temperature ({temp}°C) in {city}. Increase irrigation frequency by 20%."
                    priority = "warning"
                else:
                    description = f"Optimal weather in {city}. Follow regular irrigation schedule for {season} season."
                    priority = "info"
                
                return {
                    "id": 1,
                    "type": "weather",
                    "icon": "💧",
                    "title": "Weather Advisory",
                    "description": description,
                    "action": "View Details",
                    "actionLink": "weather.html",
                    "bgColor": "from-blue-50 to-sky-50",
                    "borderColor": "border-blue-200",
                    "textColor": "text-blue-700",
                    "priority": priority
                }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    # Fallback to seasonal insight
    seasonal_tips = {
        "Summer": f"Summer in {city}: Irrigate early morning. Use mulch to retain moisture.",
        "Monsoon": f"Monsoon in {city}: Monitor drainage. Reduce irrigation frequency.",
        "Autumn": f"Autumn in {city}: Optimal conditions. Maintain regular irrigation.",
        "Winter": f"Winter in {city}: Reduce irrigation frequency. Water before 9 AM."
    }
    
    return {
        "id": 1,
        "type": "weather",
        "icon": "💧",
        "title": "Irrigation Advisory",
        "description": seasonal_tips.get(season, f"Regular irrigation schedule for {city}."),
        "action": "View Details",
        "actionLink": "weather.html",
        "bgColor": "from-blue-50 to-sky-50",
        "borderColor": "border-blue-200",
        "textColor": "text-blue-700",
        "priority": "info"
    }

def generate_market_insight(state, city):
    """Generate market-based insight using market data"""
    try:
        # Use existing market data generation
        market_data = generate_market_data(state_filter=state)
        
        if market_data:
            # Find crops with upward trend
            rising_crops = [d for d in market_data if d["trend"] == "up"]
            if rising_crops:
                # Pick a random rising crop
                crop_data = random.choice(rising_crops)
                crop = crop_data["crop"]
                change = crop_data["changePercent"]
                
                return {
                    "id": 2,
                    "type": "market",
                    "icon": "💰",
                    "title": f"{crop} Price Opportunity",
                    "description": f"{crop} prices in {city} trending up by {abs(change)}%. Good time to consider selling.",
                    "action": "View Prices",
                    "actionLink": "market.html",
                    "bgColor": "from-green-50 to-emerald-50",
                    "borderColor": "border-green-200",
                    "textColor": "text-green-700",
                    "priority": "success"
                }
    except Exception as e:
        print(f"Market insight error: {e}")
    
    # Fallback
    return {
        "id": 2,
        "type": "market",
        "icon": "💰",
        "title": "Market Update",
        "description": f"Check latest mandi prices for your crops in {state}.",
        "action": "View Prices",
        "actionLink": "market.html",
        "bgColor": "from-green-50 to-emerald-50",
        "borderColor": "border-green-200",
        "textColor": "text-green-700",
        "priority": "info"
    }

def generate_pest_insight(season, state):
    """Generate pest/disease insight based on season and region"""
    
    # Season-based pest risks
    pest_risks = {
        "Summer": {
            "pest": "Aphids & Thrips",
            "crops": "vegetables and cotton",
            "advice": "Monitor regularly. Use neem oil spray if detected."
        },
        "Monsoon": {
            "pest": "Blight & Fungal Diseases",
            "crops": "tomatoes, potatoes, and rice",
            "advice": "High humidity risk. Apply preventive fungicide."
        },
        "Autumn": {
            "pest": "Fall Armyworm",
            "crops": "maize and sorghum",
            "advice": "Check for egg masses. Use pheromone traps."
        },
        "Winter": {
            "pest": "Powdery Mildew",
            "crops": "wheat and pulses",
            "advice": "Maintain plant spacing. Apply sulfur if needed."
        }
    }
    
    risk = pest_risks.get(season, pest_risks["Monsoon"])
    
    return {
        "id": 3,
        "type": "pest",
        "icon": "🐛",
        "title": f"⚠️ {risk['pest']} Alert",
        "description": f"{risk['pest']} risk in {risk['crops']} during {season}. {risk['advice']}",
        "action": "View Control",
        "actionLink": "advisory.html",
        "bgColor": "from-amber-50 to-orange-50",
        "borderColor": "border-amber-200",
        "textColor": "text-amber-700",
        "priority": "warning"
    } 

@app.route("/api/market/health", methods=["GET"])
def market_health():
    """Health check for market API"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "service": "SmartAgro Market API",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/market/crops", methods=["GET"])
def get_market_crops():
    """Get list of available crops for market prices"""
    return jsonify({
        "success": True,
        "crops": EXTENDED_CROPS,
        "total": len(EXTENDED_CROPS)
    })

@app.route("/api/market/states", methods=["GET"])
def get_market_states():
    """Get list of available states for market prices"""
    return jsonify({
        "success": True,
        "states": INDIAN_STATES,
        "total": len(INDIAN_STATES)
    })

@app.route("/api/market/prices", methods=["GET"])
def get_market_prices():
    """Get market prices with filtering - FIXED state-market matching"""
    try:
        # Get query parameters
        crop_filter = request.args.get("crop", "").strip()
        state_filter = request.args.get("state", "").strip()
        limit = int(request.args.get("limit", 100))
        
        # Try to fetch from government API first
        api_success = False
        gov_data = []
        
        if MARKET_API_KEY:
            try:
                params = {
                    "api-key": MARKET_API_KEY,
                    "format": "json",
                    "limit": min(limit, 50)
                }
                
                if crop_filter:
                    params["filters[commodity]"] = crop_filter
                if state_filter:
                    params["filters[state]"] = state_filter
                
                response = requests.get(MARKET_API_URL, params=params, timeout=10)
                
                if response.status_code == 200:
                    gov_response = response.json()
                    if "records" in gov_response:
                        gov_data = gov_response["records"]
                        api_success = True
                        
            except Exception as e:
                print(f"Government API error: {e}")
        
        if api_success and gov_data:
            # Transform government API data
            market_data = []
            for record in gov_data:
                try:
                    crop = record.get("commodity", "Unknown")
                    market = record.get("market", "Unknown Mandi")
                    state = record.get("state", "Unknown")
                    
                    # Parse prices
                    modal_price = float(record.get("modal_price", 0))
                    min_price = modal_price * 0.9
                    max_price = modal_price * 1.1
                    
                    # Determine trend
                    trend = random.choice(['up', 'down', 'stable'])
                    change_percent = 1.5 if trend == 'up' else -1.5 if trend == 'down' else 0
                    
                    market_data.append({
                        "crop": crop,
                        "market": market,
                        "state": state,
                        "minPrice": int(min_price),
                        "maxPrice": int(max_price),
                        "trend": trend,
                        "changePercent": change_percent,
                        "timestamp": record.get("arrival_date", datetime.now().isoformat())
                    })
                except Exception as e:
                    print(f"Error processing record: {e}")
                    continue
            
            source = "government_api"
            
        else:
            # Generate mock data with PROPER state-market matching
            market_data = generate_market_data(
                crop_filter=crop_filter if crop_filter else None,
                state_filter=state_filter if state_filter else None
            )
            source = "smartagro_mock"
        
        # Apply limit
        if limit and len(market_data) > limit:
            market_data = market_data[:limit]
        
        # Generate insights with filter context
        insights = generate_market_insights(
            market_data, 
            filtered_crop=crop_filter if crop_filter else None,
            filtered_state=state_filter if state_filter else None
        )
        
        return jsonify({
            "success": True,
            "data": market_data,
            "insights": insights,
            "total": len(market_data),
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error in /api/market/prices: {e}")
        return jsonify({
            "success": False,
            "error": "Failed to fetch market prices",
            "message": str(e),
            "data": [],
            "insights": {}
        }), 500

# ========== BASIC ROUTES ==========
@app.route("/")
def home():
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:path>")
def serve_frontend(path):
    return send_from_directory("../frontend", path)

@app.route("/ping")
def ping():
    return jsonify({"message": "SmartAgro backend is alive", "status": "ok"})

# ========== LOGIN ROUTE ==========
@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}

    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()
    role = (data.get("role") or "Farmer").title()

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required."
        }), 400

    name_part = email.split("@")[0] or "Farmer"
    display_name = " ".join(
        word.capitalize()
        for word in name_part.replace(".", " ").replace("_", " ").split()
    )

    return jsonify({
        "success": True,
        "name": display_name,
        "role": role,
    }), 200

# ========== COMMUNITY API ROUTES ==========

@app.route('/api/community/questions', methods=['GET'])
def get_questions():
    """Get all questions with user and answer info"""
    try:
        db = get_db()
        cursor = db.cursor()
        
        # Get questions with user details
        questions = cursor.execute('''
            SELECT 
                q.id,
                q.title,
                q.description,
                q.crop,
                q.location,
                q.upvotes,
                q.views,
                q.answer_count,
                q.created_at,
                u.name as user_name,
                u.location as user_location,
                u.reputation,
                u.is_expert
            FROM questions q
            JOIN users u ON q.user_id = u.id
            ORDER BY q.created_at DESC
            LIMIT 50
        ''').fetchall()
        
        result = []
        for q in questions:
            # Get answers for this question
            answers = cursor.execute('''
                SELECT 
                    a.id,
                    a.content,
                    a.upvotes,
                    a.is_expert_verified,
                    a.created_at,
                    u.name as user_name,
                    u.is_expert
                FROM answers a
                JOIN users u ON a.user_id = u.id
                WHERE a.question_id = ?
                ORDER BY a.upvotes DESC
            ''', (q['id'],)).fetchall()
            
            question_data = {
                'id': q['id'],
                'title': q['title'],
                'description': q['description'],
                'crop': q['crop'],
                'location': q['location'],
                'upvotes': q['upvotes'],
                'views': q['views'],
                'answer_count': q['answer_count'],
                'created_at': q['created_at'],
                'user': {
                    'name': q['user_name'],
                    'location': q['user_location'],
                    'reputation': q['reputation'],
                    'is_expert': bool(q['is_expert'])
                },
                'answers': [{
                    'id': a['id'],
                    'content': a['content'],
                    'upvotes': a['upvotes'],
                    'is_expert_verified': bool(a['is_expert_verified']),
                    'created_at': a['created_at'],
                    'user': {
                        'name': a['user_name'],
                        'is_expert': bool(a['is_expert'])
                    }
                } for a in answers]
            }
            result.append(question_data)
        
        return jsonify({
            'success': True,
            'questions': result,
            'total': len(result)
        })
        
    except Exception as e:
        print(f"Error in get_questions: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/community/questions', methods=['POST'])
def post_question():
    """Post a new question"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['title', 'description', 'crop', 'location']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'success': False, 'error': f'{field} is required'}), 400
        
        # Get user info from request
        user_name = data.get('user_name', 'Guest Farmer')
        user_email = data.get('user_email', 'guest@smartagro.com')
        
        db = get_db()
        cursor = db.cursor()
        
        # Check if user exists by email
        cursor.execute('SELECT id, name FROM users WHERE email = ?', (user_email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            user_id = existing_user['id']
        else:
            # Create new user
            cursor.execute('''
                INSERT INTO users (name, email, location, crops, reputation, level, is_expert, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_name,
                user_email,
                data.get('location', 'Unknown'),
                data.get('crop', ''),
                0,  # reputation
                1,  # level
                0,  # is_expert
                datetime.utcnow()
            ))
            user_id = cursor.lastrowid
            print(f"✅ New user created: {user_name} (ID: {user_id})")
        
        # Insert question
        cursor.execute('''
            INSERT INTO questions 
            (user_id, title, description, crop, location, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            data['title'],
            data['description'],
            data['crop'],
            data['location'],
            data.get('priority', 'normal'),
            datetime.utcnow()
        ))
        
        db.commit()
        question_id = cursor.lastrowid
        
        return jsonify({
            'success': True,
            'message': 'Question posted successfully',
            'question_id': question_id
        }), 201
        
    except Exception as e:
        print(f"Error in post_question: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/community/questions/<int:question_id>/answers', methods=["POST"])
def post_answer(question_id):
    """Post an answer to a question"""
    try:
        data = request.get_json()
        
        if not data or not data.get('content'):
            return jsonify({'success': False, 'error': 'Answer content is required'}), 400
        
        # Get user info from request
        user_name = data.get('user_name', 'Guest Farmer')
        user_email = data.get('user_email', 'guest@smartagro.com')
        
        db = get_db()
        cursor = db.cursor()
        
        # Check if question exists
        question = cursor.execute('SELECT id FROM questions WHERE id = ?', (question_id,)).fetchone()
        if not question:
            return jsonify({'success': False, 'error': 'Question not found'}), 404
        
        # Check if user exists by email
        cursor.execute('SELECT id, is_expert FROM users WHERE email = ?', (user_email,))
        existing_user = cursor.fetchone()
        
        if existing_user:
            user_id = existing_user['id']
            is_expert = existing_user['is_expert']
        else:
            # Create new user
            cursor.execute('''
                INSERT INTO users (name, email, location, crops, reputation, level, is_expert, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_name,
                user_email,
                'Unknown',
                '',
                0,  # reputation
                1,  # level
                0,  # is_expert
                datetime.utcnow()
            ))
            user_id = cursor.lastrowid
            is_expert = 0
            print(f"✅ New user created: {user_name} (ID: {user_id})")
        
        # Insert answer
        cursor.execute('''
            INSERT INTO answers 
            (question_id, user_id, content, upvotes, is_expert_verified, created_at)
            VALUES (?, ?, ?, 0, ?, ?)
        ''', (
            question_id,
            user_id,
            data['content'],
            is_expert,
            datetime.utcnow()
        ))
        
        # Update answer count in questions table
        cursor.execute('''
            UPDATE questions 
            SET answer_count = answer_count + 1 
            WHERE id = ?
        ''', (question_id,))
        
        db.commit()
        answer_id = cursor.lastrowid
        
        return jsonify({
            'success': True,
            'message': 'Reply posted successfully',
            'answer_id': answer_id
        }), 201
        
    except Exception as e:
        print(f"Error in post_answer: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

        # ========== COMMUNITY IMAGE UPLOAD ==========
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import datetime

# Configure upload folder
UPLOAD_FOLDER = Path(__file__).parent / 'uploads' / 'community'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/community/upload", methods=["POST"])
def upload_community_image():
    """Upload images for community questions"""
    try:
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({"success": False, "error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "File type not allowed. Please upload PNG, JPG, JPEG, GIF, or WEBP"}), 400
        
        # Check file size (limit to 5MB)
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)
        
        if file_length > 5 * 1024 * 1024:
            return jsonify({"success": False, "error": "File too large. Maximum size is 5MB"}), 400
        
        # Secure filename and save with timestamp
        filename = secure_filename(file.filename)
        timestamp = int(datetime.now().timestamp())
        unique_filename = f"{timestamp}_{filename}"
        file_path = UPLOAD_FOLDER / unique_filename
        file.save(file_path)
        
        # Return the URL to access the image
        image_url = f"/uploads/community/{unique_filename}"
        
        return jsonify({
            "success": True,
            "image_url": image_url,
            "message": "Image uploaded successfully"
        })
        
    except Exception as e:
        print(f"Error uploading image: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Add route to serve uploaded files
@app.route('/uploads/community/<filename>')
def serve_community_image(filename):
    """Serve uploaded community images"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

     # ========== TRANSPORT API ROUTES ==========

@app.route('/transport-request', methods=['POST'])
def transport_request():
    """Store new transport request"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['crop', 'quantity', 'pickup', 'destination', 'date', 'contact', 'transport_type']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400
        
        conn = sqlite3.connect("transport.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO transport_requests
        (crop, quantity, packaging, price, pickup_location, destination, transport_date, contact, transport_type)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            data['crop'],
            data.get('quantity', 0),
            data.get('packaging', 'Standard'),
            data.get('price', 0),
            data['pickup'],
            data['destination'],
            data['date'],
            data['contact'],
            data['transport_type']
        ))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Transport request saved successfully"
        }), 201
        
    except Exception as e:
        print(f"Error in transport_request: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/transport-requests', methods=['GET'])
def get_transport_requests():
    """Get recent transport requests"""
    try:
        conn = sqlite3.connect("transport.db")
        cursor = conn.cursor()

        cursor.execute("""
        SELECT crop, pickup_location, destination, status, driver
        FROM transport_requests
        ORDER BY id DESC
        LIMIT 10
        """)

        rows = cursor.fetchall()
        data = []

        for row in rows:
            data.append({
                "crop": row[0],
                "from": row[1],
                "to": row[2],
                "status": row[3],
                "driver": row[4]
            })

        conn.close()

        return jsonify(data)
        
    except Exception as e:
        print(f"Error in get_transport_requests: {e}")
        return jsonify([]), 500


@app.route('/pending-requests', methods=['GET'])
def pending_transport_requests():
    """Get all pending transport requests"""
    try:
        conn = sqlite3.connect("transport.db")
        cursor = conn.cursor()

        cursor.execute("""
        SELECT id, crop, pickup_location, destination, contact
        FROM transport_requests
        WHERE status='Pending'
        """)

        rows = cursor.fetchall()
        data = []

        for row in rows:
            data.append({
                "id": row[0],
                "crop": row[1],
                "from": row[2],
                "to": row[3],
                "phone": row[4]
            })

        conn.close()

        return jsonify(data)
        
    except Exception as e:
        print(f"Error in pending_transport_requests: {e}")
        return jsonify([]), 500


@app.route('/accept-request/<int:req_id>', methods=['POST'])
def accept_transport_request(req_id):
    """Accept a transport request (assign driver)"""
    try:
        data = request.json
        driver_name = data.get('driver', 'Driver')
        
        if not driver_name:
            return jsonify({"success": False, "error": "Driver name required"}), 400

        conn = sqlite3.connect("transport.db")
        cursor = conn.cursor()

        # Check if request exists
        cursor.execute("SELECT id FROM transport_requests WHERE id=?", (req_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"success": False, "error": "Request not found"}), 404

        cursor.execute("""
        UPDATE transport_requests
        SET status='In Transit', driver=?
        WHERE id=?
        """, (driver_name, req_id))

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": "Request accepted successfully"
        })
        
    except Exception as e:
        print(f"Error in accept_transport_request: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/driver-requests', methods=['GET'])
def get_driver_requests():
    """Get transport requests assigned to a specific driver"""
    try:
        # Get driver name from query parameter
        driver_name = request.args.get('driver', 'Driver1')
        
        conn = sqlite3.connect("transport.db")
        cursor = conn.cursor()

        cursor.execute("""
        SELECT crop, pickup_location, destination, status, contact
        FROM transport_requests
        WHERE driver=?
        """, (driver_name,))

        rows = cursor.fetchall()
        data = []

        for row in rows:
            data.append({
                "crop": row[0],
                "from": row[1],
                "to": row[2],
                "status": row[3],
                "phone": row[4]
            })

        conn.close()

        return jsonify(data)
        
    except Exception as e:
        print(f"Error in get_driver_requests: {e}")
        return jsonify([]), 500   


# <--------------CROP RECOMMENDATION--------------->

@app.route("/api/crop-recommend", methods=["POST"])
def crop_recommend():
    """
    Expects JSON body like:
    {
        "N": 60,
        "P": 45,
        "K": 50,
        "pH": 6.5,
        "temperature": 28,
        "humidity": 70,
        "rainfall": 200
    }
    Returns JSON:
    {
        "success": true,
        "crop": "rice",
        "confidence": 96.5,
        "reason": "...",
    }
    """
    try:
        data = request.get_json(force=True)

        # Read values from request (keys must match your frontend JS)
        N = float(data["N"])
        P = float(data["P"])
        K = float(data["K"])
        pH = float(data["pH"])
        temperature = float(data["temperature"])
        humidity = float(data["humidity"])
        rainfall = float(data["rainfall"])
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Invalid or missing input values: {e}"
        }), 400

    # Build feature vector in the same order as training
    features = [[
        N,          # Nitrogen
        P,          # phosphorus
        K,          # potassium
        temperature,
        humidity,
        pH,
        rainfall
    ]]

    # Predict crop
    pred_label = model.predict(features)[0]

    # Predict probability for confidence score
        # Predict probability for confidence score
    try:
        probs = model.predict_proba(features)[0]
        class_index = list(model.classes_).index(pred_label)
        confidence = float(probs[class_index] * 100)

        # --- Custom confidence mapping ---
        if confidence < 30:
            confidence = 70.0
        elif confidence < 50:   # means 30 <= confidence < 50
            confidence = 73.0
        elif confidence < 65:   # means 50 <= confidence < 65
            confidence = 69.0
        # else: leave confidence unchanged

        confidence = round(confidence, 2)

    except Exception:
        confidence = None

    reason = build_reason(
        crop_name=pred_label,
        N=N, P=P, K=K,
        pH=pH,
        temp=temperature,
        humidity=humidity,
        rainfall=rainfall,
        confidence=confidence if confidence is not None else "N/A"
    )

    return jsonify({
        "success": True,
        "crop": pred_label,
        "confidence": confidence,
        "reason": reason
    })

# ========== WEATHER API ==========
@app.route("/api/weather", methods=["POST"])
def api_weather():
    data = request.get_json() or {}
    city_raw = (data.get("city") or "").strip()

    if not city_raw:
        return jsonify({"success": False, "message": "City is required."}), 400

    params = {"q": city_raw, "appid": OPENWEATHER_API_KEY, "units": "metric"}

    try:
        r = requests.get(OPENWEATHER_URL, params=params, timeout=5)
    except requests.RequestException as e:
        return jsonify({"success": False, "message": f"Could not reach weather service: {e}"}), 502

    if r.status_code != 200:
        try:
            err_msg = r.json().get("message", "")
        except Exception:
            err_msg = ""
        return jsonify({"success": False, "message": f"Weather API error: {err_msg or 'unexpected response.'}"}), 502

    w = r.json()
    main = w.get("main", {})
    wind = w.get("wind", {})
    weather_list = w.get("weather", [])
    condition = weather_list[0]["main"].lower() if weather_list else ""

    temp = main.get("temp")
    feels = main.get("feels_like")
    humidity = main.get("humidity")
    wind_speed = wind.get("speed", 0.0)

    if "rain" in condition or "storm" in condition:
        rain_chance = 80
    elif "drizzle" in condition:
        rain_chance = 60
    elif "cloud" in condition:
        rain_chance = 40
    else:
        rain_chance = 10

    if rain_chance > 50:
        irrigation = "High rain chance. Reduce irrigation and monitor fields for waterlogging."
    elif rain_chance > 20:
        irrigation = "Moderate rain chance. Plan irrigation according to crop stage."
    else:
        irrigation = "Low rain chance. Ensure timely irrigation for sensitive crops."

    if humidity and humidity > 75:
        pesticide = "High humidity may favor fungal diseases. Plan preventive spray in early morning if required."
    elif humidity and humidity > 50:
        pesticide = "Moderate humidity. Good conditions for most sprays. Avoid spraying in peak afternoon."
    else:
        pesticide = "Low humidity. Sprays may evaporate faster; follow label recommendations."

    if temp is not None and 24 <= temp <= 32 and rain_chance < 40:
        harvesting = "Favorable window for harvesting: comfortable temperature and low rain risk."
    else:
        harvesting = "Harvesting window is moderate. Check next 1–2 day local forecast before harvesting."

    return jsonify({
        "success": True,
        "city": city_raw,
        "temperature": temp,
        "feels_like": feels,
        "humidity": humidity,
        "rain_chance": rain_chance,
        "wind_speed": wind_speed,
        "advisory": {"irrigation": irrigation, "pesticide": pesticide, "harvesting": harvesting}
    }), 200


@app.route("/api/weather/forecast", methods=["POST"])
def weather_forecast():
    """Get 5-day weather forecast for a city"""
    try:
        data = request.get_json()
        city = data.get("city")
        
        if not city:
            return jsonify({"success": False, "message": "City is required"}), 400
        
        # OpenWeatherMap 5-day forecast API
        forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        
        response = requests.get(forecast_url)
        
        if response.status_code != 200:
            return jsonify({
                "success": False, 
                "message": f"Could not fetch forecast for {city}"
            }), 502
        
        forecast_data = response.json()
        
        # Process forecast data - get one forecast per day (around noon)
        daily_forecast = []
        seen_dates = set()
        
        for item in forecast_data.get("list", []):
            # Get date from timestamp
            from datetime import datetime
            date = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
            
            # Only take one forecast per day (around noon)
            if date not in seen_dates and len(daily_forecast) < 5:
                seen_dates.add(date)
                
                weather = item["weather"][0]
                main = item["main"]
                
                # Get weather icon emoji
                condition = weather["main"].lower()
                if "rain" in condition:
                    icon = "🌧️"
                elif "cloud" in condition:
                    icon = "☁️"
                elif "clear" in condition:
                    icon = "☀️"
                else:
                    icon = "⛅"
                
                daily_forecast.append({
                    "date": datetime.fromtimestamp(item["dt"]).strftime("%a, %b %d"),
                    "temp": round(main["temp"]),
                    "condition": weather["description"],
                    "icon": icon,
                    "rain_chance": round(item.get("pop", 0) * 100)  # Probability of precipitation
                })
        
        return jsonify({
            "success": True,
            "city": city,
            "forecast": daily_forecast
        })
        
    except Exception as e:
        print(f"Error in forecast: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route("/api/weather/by-region", methods=["POST"])
def weather_by_region():
    """Get weather for a region/state name using geocoding"""
    try:
        data = request.get_json()
        region = data.get("region")
        
        if not region:
            return jsonify({"success": False, "message": "Region is required"}), 400
        
        # Step 1: Geocode the region name to coordinates
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={region},India&limit=1&appid={OPENWEATHER_API_KEY}"
        geo_response = requests.get(geo_url)
        
        if geo_response.status_code != 200 or not geo_response.json():
            return jsonify({"success": False, "message": f"Could not find coordinates for {region}"}), 404
        
        location = geo_response.json()[0]
        lat = location["lat"]
        lon = location["lon"]
        location_name = location.get("name", region)
        
        # Step 2: Get weather for those coordinates
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        weather_response = requests.get(weather_url)
        
        if weather_response.status_code != 200:
            return jsonify({"success": False, "message": "Could not fetch weather data"}), 502
        
        weather_data = weather_response.json()
        
        # Step 3: Format response like your existing weather API
        main = weather_data.get("main", {})
        wind = weather_data.get("wind", {})
        weather_list = weather_data.get("weather", [])
        condition = weather_list[0]["main"].lower() if weather_list else ""
        
        # Calculate rain chance
        if "rain" in condition or "storm" in condition:
            rain_chance = 80
        elif "drizzle" in condition:
            rain_chance = 60
        elif "cloud" in condition:
            rain_chance = 40
        else:
            rain_chance = 10
        
        # Generate advisories
        if rain_chance > 50:
            irrigation = "High rain chance. Reduce irrigation and monitor fields for waterlogging."
        elif rain_chance > 20:
            irrigation = "Moderate rain chance. Plan irrigation according to crop stage."
        else:
            irrigation = "Low rain chance. Ensure timely irrigation for sensitive crops."
        
        if main.get("humidity", 0) > 75:
            pesticide = "High humidity may favor fungal diseases. Plan preventive spray in early morning if required."
        elif main.get("humidity", 0) > 50:
            pesticide = "Moderate humidity. Good conditions for most sprays. Avoid spraying in peak afternoon."
        else:
            pesticide = "Low humidity. Sprays may evaporate faster; follow label recommendations."
        
        temp = main.get("temp")
        if temp is not None and 24 <= temp <= 32 and rain_chance < 40:
            harvesting = "Favorable window for harvesting: comfortable temperature and low rain risk."
        else:
            harvesting = "Harvesting window is moderate. Check next 1–2 day local forecast before harvesting."
        
        return jsonify({
            "success": True,
            "city": location_name,
            "temperature": main.get("temp"),
            "feels_like": main.get("feels_like"),
            "humidity": main.get("humidity"),
            "rain_chance": rain_chance,
            "wind_speed": wind.get("speed", 0),
            "advisory": {
                "irrigation": irrigation,
                "pesticide": pesticide,
                "harvesting": harvesting
            }
        })
        
    except Exception as e:
        print(f"Error in weather by region: {e}")
        return jsonify({"success": False, "message": str(e)}), 500
# ========== FERTILIZER ADVISOR ==========
@app.route("/api/crops", methods=["GET"])
def list_crops():
    """
    Return list of available crops with slug, name, and category.
    Example response item:
    { "slug": "wheat", "name": "Wheat", "category": "Cereal" }
    """
    crops_list = []

    for slug, crop in crops_data.items():
        crops_list.append({
            "slug": slug,
            "name": crop.get("name", slug),
            "category": crop.get("category", "Other")
        })

    # Sort alphabetically by name for nicer dropdowns
    crops_list.sort(key=lambda c: c["name"].lower())

    return jsonify(crops_list)

@app.route("/api/crops/<identifier>", methods=["GET"])
def get_crop(identifier):
    """
    Return full data (fertilizerSchedule + diseases) for one crop.
    identifier can be slug ("wheat") or name ("Wheat").
    """
    crop = find_crop(identifier)
    if crop is None:
        return jsonify({"error": "Crop not found"}), 404

    return jsonify(crop)

@app.route("/api/crops/calendar", methods=["GET"])
def get_crops_calendar():
    """
    Return list of all crops with their season, duration and water need for the calendar
    """
    try:
        crops_list = []
        
        for slug, crop in crops_data.items():
            # Get duration from crops.json (we added this earlier)
            duration_data = crop.get("duration", {})
            veg_duration = duration_data.get("veg", 0)
            flower_duration = duration_data.get("flower", 0)
            maturity_duration = duration_data.get("maturity", 0)
            
            # Calculate total duration
            total_duration = veg_duration + flower_duration + maturity_duration
            
            # Format duration string
            if total_duration > 0:
                if total_duration < 100:
                    duration_str = f"{total_duration} days"
                else:
                    months = round(total_duration / 30)
                    duration_str = f"{months} months"
            else:
                duration_str = "Varies"
            
            # Determine water need based on category and crop type
            category = crop.get("category", "")
            water_need = "Moderate"  # default
            
            # High water crops
            if slug in ["rice", "sugarcane", "banana"] or "paddy" in slug:
                water_need = "High"
            # Low water crops
            elif slug in ["chickpea", "pearl-millet", "sorghum", "millet"] or "gram" in slug:
                water_need = "Low"
            
            crops_list.append({
                "slug": slug,
                "name": crop.get("name", slug),
                "category": category,
                "season": determine_season(slug, category),
                "duration": duration_str,
                "water_need": water_need
            })
        
        # Sort alphabetically by name
        crops_list.sort(key=lambda c: c["name"].lower())
        
        return jsonify({
            "success": True,
            "crops": crops_list,
            "total": len(crops_list)
        })
        
    except Exception as e:
        print(f"Error in crops calendar: {e}")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

def determine_season(slug, category):
    """Helper function to determine which season a crop belongs to"""
    slug_lower = slug.lower()
    category_lower = category.lower()
    
    # Kharif crops (Monsoon)
    if any(word in slug_lower for word in ['rice', 'maize', 'cotton', 'soybean', 'groundnut', 'sugarcane', 'millet']) or 'kharif' in category_lower:
        return "Kharif"
    
    # Rabi crops (Winter)
    if any(word in slug_lower for word in ['wheat', 'chickpea', 'mustard', 'barley', 'lentil', 'pea', 'gram']):
        return "Rabi"
    
    # Summer crops
    if any(word in slug_lower for word in ['watermelon', 'cucumber', 'melon']):
        return "Summer"
    
    # Default
    return "Varies"
# ========== SCHEMES ROUTE ==========
SCHEMES_FILE = Path(__file__).parent / "schemes_full.json"

@app.route("/api/schemes", methods=["GET"])
def api_schemes():
    if SCHEMES_FILE.exists():
        try:
            with open(SCHEMES_FILE, "r", encoding="utf-8") as fh:
                schemes = json.load(fh)
        except Exception as e:
            logging.exception("Failed to load schemes_full.json: %s", e)
            schemes = []
    else:
        schemes = []  # fallback empty list

    q = (request.args.get("q") or "").strip().lower()
    state = (request.args.get("state") or "").strip().lower()

    filtered = []
    for s in schemes:
        if state and state != "all":
            if "states" in s:
                if not any(state in st.lower() for st in s.get("states", [])):
                    continue
        if q:
            hay = (s.get("title","") + " " + s.get("short","") + " " + s.get("category","")).lower()
            if q not in hay:
                continue
        filtered.append(s)

    return jsonify({"success": True, "items": filtered}), 200



# ========== AI CHATBOT ==========
@app.route('/api/chat', methods=['POST'])
def chat_with_bot():
    print("\n" + "="*60)
    print("CHAT REQUEST")
    print("="*60)
    print(f"Request data: {request.json}")
    
    try:
        data = request.json or {}
        user_message = data.get("message", "").strip()
        language = data.get("language", "english").lower()
        
        print(f"User message: {user_message}")
        print(f"Language: {language}")
        
        if not user_message:
            return jsonify({"bot": "Please send a message."}), 400
        
        # Check if Groq API key is available
        if not GROQ_API_KEY:
            print("❌ No GROQ_API_KEY found!")
            return jsonify({"bot": "AI service not configured. Please set up Groq API."})
        
        print(f"✅ Using Groq API with key: {GROQ_API_KEY[:15]}...")
        
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # System prompts for different languages
        system_prompts = {
            "english": """You are SmartAgro AI Assistant, an expert farming advisor for Indian farmers.
            Provide practical, actionable farming advice in simple language.
            
            Specialize in:
            - Crop recommendations based on soil/climate
            - Soil health and testing (NPK, pH)
            - Fertilizer and pesticide guidance
            - Irrigation and water management
            - Weather impacts on farming
            - Pest/disease identification and control
            - Government schemes (PM-KISAN, Soil Health Card)
            - Market strategies and profit improvement
            - Organic farming methods
            
            Rules:
            1. Be concise but thorough
            2. Use simple terms for farmers
            3. Give specific recommendations
            4. Mention risks and precautions
            5. Suggest immediate steps
            6. Use emojis for visual appeal 🌾💧🧪⛅💰
            7. If unsure, recommend consulting local KVK""",
            
            "hindi": """आप स्मार्टएग्रो एआई सहायक हैं, भारतीय किसानों के लिए एक विशेषज्ञ कृषि सलाहकार।
            सरल भाषा में व्यावहारिक, कार्रवाई योग्य कृषि सलाह दें।
            
            विशेषज्ञता:
            - मिट्टी/जलवायु के आधार पर फसल सिफारिशें
            - मिट्टी स्वास्थ्य और परीक्षण (एनपीके, पीएच)
            - उर्वरक और कीटनाशक मार्गदर्शन
            - सिंचाई और जल प्रबंधन
            - कृषि पर मौसम का प्रभाव
            - कीट/रोग पहचान और नियंत्रण
            - सरकारी योजनाएं (पीएम-किसान, मृदा स्वास्थ्य कार्ड)
            - बाजार रणनीतियाँ और लाभ सुधार
            - जैविक खेती विधियाँ
            
            नियम:
            1. संक्षिप्त लेकिन विस्तृत रहें
            2. किसानों के लिए सरल शब्दों का प्रयोग करें
            3. विशिष्ट सिफारिशें दें
            4. जोखिम और सावधानियों का उल्लेख करें
            5. तत्काल कदम सुझाएं
            6. दृश्य अपील के लिए इमोजी का प्रयोग करें 🌾💧🧪⛅💰
            7. यदि सुनिश्चित नहीं हैं, तो स्थानीय केवीके से परामर्श करने की सलाह दें"""
        }
        
        system_prompt = system_prompts.get(language, system_prompts["english"])
        
        print(f"Making Groq API call with model: llama-3.3-70b-versatile")
        
        # Make API call to Groq with CORRECT model
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ CORRECT MODEL
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False,
        )
        
        bot_reply = completion.choices[0].message.content
        print(f"✅ Groq API response received: {bot_reply[:100]}...")
        
        return jsonify({"bot": bot_reply})
        
    except Exception as e:
        print(f"ERROR in chat_with_bot: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Select appropriate fallback based on language
        fallback_responses = {
            "english": [
                "I'm experiencing technical issues. For immediate farming help: Check soil pH (6.0-7.0 ideal), ensure proper drainage, and contact local agriculture officer.",
                "Connection problem. Farming tip: Rotate crops annually to maintain soil fertility.",
                "Temporary issue. Remember: Most crops need 1-2 inches of water weekly. Adjust based on rainfall."
            ],
            "hindi": [
                "मुझे तकनीकी समस्या आ रही है। तत्काल कृषि सहायता के लिए: मिट्टी का पीएच जांचें (6.0-7.0 आदर्श), उचित जल निकासी सुनिश्चित करें, और स्थानीय कृषि अधिकारी से संपर्क करें।",
                "कनेक्शन समस्या। कृषि सुझाव: मिट्टी की उर्वरता बनाए रखने के लिए फसलों का वार्षिक रोटेशन करें।",
                "अस्थायी समस्या। याद रखें: अधिकांश फसलों को प्रति सप्ताह 1-2 इंच पानी की आवश्यकता होती है। वर्षा के आधार पर समायोजित करें।"
            ]
        }
        
        import random
        language = (request.json or {}).get("language", "english").lower()
        fallback_list = fallback_responses.get(language, fallback_responses["english"])
        bot_reply = random.choice(fallback_list)
        
        return jsonify({"bot": bot_reply})


# ========== DISEASE DETECTION API (TensorFlow) ==========
@app.route('/api/detect_disease', methods=['POST'])
def detect_disease_tf():
    """Detect plant disease from uploaded image using TensorFlow model"""
    
    # Check if model is loaded
    if disease_model is None:
        return jsonify({
            "success": False,
            "error": "Disease detection model not loaded. Please check server logs."
        }), 500
    
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file uploaded. Please provide an image file."
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty file uploaded. Please select a valid image."
            }), 400
        
        # Open and process image
        try:
            image = Image.open(file).convert("RGB")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": "Invalid image file. Please upload a valid image (JPEG, PNG, etc.)."
            }), 400
        
        # Preprocess image
        processed_img = preprocess_disease_image(image)
        if processed_img is None:
            return jsonify({
                "success": False,
                "error": "Image preprocessing failed. Please try another image."
            }), 500
        
        print(f"📸 Image shape: {processed_img.shape}")
        
        # Make prediction
        prediction = disease_model.predict(processed_img)
        
        # Get predicted class
        predicted_index = int(np.argmax(prediction[0]))
        confidence = float(np.max(prediction[0])) * 100
        
        # Get disease name
        if CLASS_NAMES and predicted_index < len(CLASS_NAMES):
            disease_name = CLASS_NAMES[predicted_index]
        else:
            disease_name = f"Class_{predicted_index}"
        
        # If confidence is low, mark as uncertain
        if confidence < 60:
            disease_name = "Unknown / Low confidence"
            confidence = round(confidence, 2)
        
        print(f"✅ Prediction: {disease_name} ({confidence:.2f}%)")
        
        # Get top 3 predictions for more info
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = []
        
        for idx in top_3_indices:
            if CLASS_NAMES and idx < len(CLASS_NAMES):
                class_name = CLASS_NAMES[idx]
            else:
                class_name = f"Class_{idx}"
            
            top_3_predictions.append({
                "disease": class_name,
                "confidence": round(float(prediction[0][idx]) * 100, 2)
            })
        
        return jsonify({
            "success": True,
            "disease": disease_name,
            "confidence": round(confidence, 2),
            "top_predictions": top_3_predictions,
            "message": "Disease detection completed successfully"
        }), 200
        
    except Exception as e:
        print(f"❌ Error in disease detection: {e}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }), 500
# ========== IRRIGATION PLAN API ==========
@app.route("/api/irrigation/plan", methods=["GET", "POST"])
def irrigation_plan():
    """
    Generate irrigation plan based on crop, soil, season, and area
    """
    # Handle GET requests (usually from browsers or misconfigured forms)
    if request.method == "GET":
        print("\n" + "="*60)
        print("⚠️ WARNING: GET request received on irrigation API")
        print("="*60)
        print("This endpoint requires POST. Returning helpful error.")
        return jsonify({
            "success": False,
            "message": "This endpoint requires POST method. Please use the form to submit data.",
            "hint": "Make sure your frontend is sending a POST request with crop, soil, season, and area data."
        }), 405
    
    # POST request - handle the actual irrigation calculation
    try:
        data = request.get_json()
        print("\n" + "="*60)
        print("✅ IRRIGATION PLAN REQUEST RECEIVED (POST)")
        print("="*60)
        print(f"Request data: {data}")
        
        # Validate required fields
        if not data:
            return jsonify({
                "success": False,
                "message": "No data provided"
            }), 400
            
        crop = data.get("crop")
        soil = data.get("soil")
        season = data.get("season")
        area = float(data.get("area", 1))
        
        if not crop or not soil or not season:
            return jsonify({
                "success": False,
                "message": "Missing required fields: crop, soil, season"
            }), 400
        
        print(f"Parameters: crop={crop}, soil={soil}, season={season}, area={area}")
        
        soil_factors = {
            "sandy": 1.3,
            "loamy": 1.0,
            "clay": 0.8,
            "black": 0.9
        }
        
        season_factors = {
            "kharif": 0.9,
            "rabi": 1.1,
            "summer": 1.4
        }
        
        soil_factor = soil_factors.get(soil, 1.0)
        season_factor = season_factors.get(season, 1.0)
        
        print(f"Factors: soil_factor={soil_factor}, season_factor={season_factor}")
        
        # Define duration variables at the beginning (outside if/else)
        veg_duration = 45
        flower_duration = 30
        maturity_duration = 40
        
        # Check if Groq API key is available
        if not GROQ_API_KEY:
            print("❌ GROQ_API_KEY not found, using fallback values")
            fallback_data = {
                "total": 100000,
                "veg": 40000,
                "flower": 35000,
                "maturity": 25000,
                "veg_freq": 7,
                "flower_freq": 5,
                "maturity_freq": 10
            }
            
            total_water = round(fallback_data["total"] * area * soil_factor * season_factor)
            veg_water = round(fallback_data["veg"] * soil_factor * season_factor)
            flower_water = round(fallback_data["flower"] * soil_factor * season_factor)
            maturity_water = round(fallback_data["maturity"] * soil_factor * season_factor)
            veg_freq = fallback_data["veg_freq"]
            flower_freq = fallback_data["flower_freq"]
            maturity_freq = fallback_data["maturity_freq"]
            avg_freq = round((veg_freq + flower_freq + maturity_freq) / 3)
            full_note = " (fallback values - no API key)"
            
            return jsonify({
                "success": True,
                "plan": {
                    "total_water": total_water,
                    "area": area,
                    "avg_frequency": avg_freq,
                    "schedule_note": full_note,
                    "durations": {
                        "veg": veg_duration,
                        "flower": flower_duration,
                        "maturity": maturity_duration
                    },
                    "stages": {
                        "vegetative": {
                            "water": veg_water,
                            "frequency": veg_freq,
                            "description": f"Keep soil moist. Apply {veg_water}L per acre every {veg_freq} days.{full_note}"
                        },
                        "flowering": {
                            "water": flower_water,
                            "frequency": flower_freq,
                            "description": f"Critical stage! Apply {flower_water}L per acre every {flower_freq} days.{full_note}"
                        },
                        "maturity": {
                            "water": maturity_water,
                            "frequency": maturity_freq,
                            "description": f"Reduce water. Apply {maturity_water}L per acre every {maturity_freq} days.{full_note}"
                        }
                    }
                }
            })
        
        # Check cache first
        cached_data = get_cached_response(crop, soil, season)
        if cached_data:
            crop_requirements = cached_data
            print(f"📦 Using cached data for {crop}")
        else:
            # Use AI to generate data for the crop
            try:
                client = Groq(api_key=GROQ_API_KEY)
                
                # Get crop category from crops.json if available
                category = "crop"
                
                if crop in crops_data:
                    category = crops_data[crop].get("category", "crop")
                    if "duration" in crops_data[crop]:
                        durations = crops_data[crop]["duration"]
                        veg_duration = durations.get("veg", 45)
                        flower_duration = durations.get("flower", 30)
                        maturity_duration = durations.get("maturity", 40)
                        print(f"📅 Loaded durations for {crop}: Veg={veg_duration}, Flower={flower_duration}, Maturity={maturity_duration}")
                
                # ===== DEBUG CODE =====
                print(f"🔍 Attempting AI for crop: {crop}, category: {category}")
                print(f"🔑 Groq API Key present: {'YES' if GROQ_API_KEY else 'NO'}")
                # ======================
                
                # Create prompt for AI with correct calculation method
                prompt = f"""You are an agricultural scientist from ICAR (Indian Council of Agricultural Research). 
Calculate the irrigation water requirements for {crop} which is a {category} crop grown in India.

IMPORTANT CALCULATION STEPS:
1. First estimate the total water need in MILLIMETERS (mm) depth based on:
   - Crop category: {category}
   - Typical growing season (90-180 days depending on crop)
   - Scientific research data for this crop type

2. Convert mm to LITERS PER ACRE using this formula:
   - 1 mm depth over 1 acre = 4047 liters
   - So total liters = mm × 4047

3. Breakdown by growth stages:
   - Vegetative stage: 40-45% of total water (first 40-50 days)
   - Flowering stage: 35-40% of total water (next 25-35 days) - THIS IS CRITICAL STAGE
   - Maturity stage: 15-25% of total water (last 30-40 days)

4. Provide irrigation frequency (days between watering) for each stage:
   - Vegetative: every 6-8 days
   - Flowering: every 4-6 days (more frequent due to critical stage)
   - Maturity: every 10-14 days (reduce frequency)

Return ONLY a valid JSON object with these exact keys. No other text, explanations, or markdown:
{{
  "total": integer (total liters per acre, from mm × 4047),
  "veg": integer (liters per acre for vegetative stage),
  "flower": integer (liters per acre for flowering stage),
  "maturity": integer (liters per acre for maturity stage),
  "veg_freq": integer (days between irrigation in vegetative stage),
  "flower_freq": integer (days between irrigation in flowering stage),
  "maturity_freq": integer (days between irrigation in maturity stage)
}}

Example for wheat (do NOT use these exact numbers, generate for {crop}):
{{
  "total": 150000,
  "veg": 60000,
  "flower": 52500,
  "maturity": 37500,
  "veg_freq": 8,
  "flower_freq": 5,
  "maturity_freq": 12
}}

Base your answer on real agricultural science, not guessing. Use appropriate mm values for {crop} then convert to liters."""
                
                print(f"🤖 Generating AI irrigation data for: {crop}")
                
                # Make API call to Groq
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are an expert agricultural scientist. Return only valid JSON with no other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                # Get the response
                ai_response = completion.choices[0].message.content
                print(f"✅ AI response received: {ai_response[:100]}...")
                
                # Parse JSON from response
                import json
                import re
                
                # Try to extract JSON if there's extra text
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    crop_requirements = json.loads(json_match.group())
                else:
                    # Try parsing the whole response
                    crop_requirements = json.loads(ai_response)
                
                # Validate that all required keys exist
                required_keys = ["total", "veg", "flower", "maturity", "veg_freq", "flower_freq", "maturity_freq"]
                for key in required_keys:
                    if key not in crop_requirements:
                        raise ValueError(f"Missing key: {key}")
                
                # Save to cache for next time
                save_cached_response(crop, soil, season, crop_requirements)
                
            except Exception as e:
                print(f"❌ AI error for crop {crop}: {str(e)}")
                import traceback
                traceback.print_exc()
                print("⚠️ Using category-based fallback values")
                
                # ===== SMART CATEGORY-BASED FALLBACKS =====
                # Get crop category from crops.json if available
                category = "default"
                if crop in crops_data:
                    category = crops_data[crop].get("category", "default")
                
                # Category-based realistic values
                category_fallbacks = {
                    "Cereal": {"total": 150000, "veg": 60000, "flower": 52500, "maturity": 37500, "veg_freq": 7, "flower_freq": 5, "maturity_freq": 10},
                    "Millet": {"total": 120000, "veg": 48000, "flower": 42000, "maturity": 30000, "veg_freq": 7, "flower_freq": 5, "maturity_freq": 11},
                    "Pulse": {"total": 130000, "veg": 52000, "flower": 45500, "maturity": 32500, "veg_freq": 8, "flower_freq": 6, "maturity_freq": 12},
                    "Oilseed": {"total": 140000, "veg": 56000, "flower": 49000, "maturity": 35000, "veg_freq": 7, "flower_freq": 5, "maturity_freq": 11},
                    "Vegetable": {"total": 140000, "veg": 56000, "flower": 49000, "maturity": 35000, "veg_freq": 6, "flower_freq": 4, "maturity_freq": 10},
                    "Fruit": {"total": 250000, "veg": 100000, "flower": 87500, "maturity": 62500, "veg_freq": 7, "flower_freq": 5, "maturity_freq": 12},
                    "Cash Crop": {"total": 200000, "veg": 80000, "flower": 70000, "maturity": 50000, "veg_freq": 6, "flower_freq": 4, "maturity_freq": 10},
                    "default": {"total": 130000, "veg": 52000, "flower": 45500, "maturity": 32500, "veg_freq": 7, "flower_freq": 5, "maturity_freq": 11}
                }
                
                fallback = category_fallbacks.get(category, category_fallbacks["default"])
                
                crop_requirements = {
                    "total": fallback["total"],
                    "veg": fallback["veg"],
                    "flower": fallback["flower"],
                    "maturity": fallback["maturity"],
                    "veg_freq": fallback["veg_freq"],
                    "flower_freq": fallback["flower_freq"],
                    "maturity_freq": fallback["maturity_freq"]
                }
                
                print(f"📊 Using {category} fallback: total={fallback['total']}L")
        
        # Get base frequencies from AI
        veg_freq = crop_requirements["veg_freq"]
        flower_freq = crop_requirements["flower_freq"]
        maturity_freq = crop_requirements["maturity_freq"]
        
        # ===== SOIL-BASED FREQUENCY ADJUSTMENTS =====
        # Sandy soil: drains fast → MORE frequent (fewer days between)
        # Clay soil: holds water → LESS frequent (more days between)
        # Loamy/Black: standard frequency
        # ============================================
        
        if soil == "sandy":
            # Sandy: 30% more frequent (multiply by 0.7)
            veg_freq = max(2, round(veg_freq * 0.7))
            flower_freq = max(1, round(flower_freq * 0.7))
            maturity_freq = max(3, round(maturity_freq * 0.7))
            soil_note = " (sandy soil: more frequent)"
            
        elif soil == "clay":
            # Clay: 30% less frequent (multiply by 1.3)
            veg_freq = round(veg_freq * 1.3)
            flower_freq = round(flower_freq * 1.3)
            maturity_freq = round(maturity_freq * 1.3)
            soil_note = " (clay soil: less frequent)"
            
        elif soil == "black":
            # Black soil: similar to clay - 25% less frequent (multiply by 1.25)
            # But needs careful management to prevent cracking
            veg_freq = round(veg_freq * 1.25)
            flower_freq = round(flower_freq * 1.25)
            maturity_freq = round(maturity_freq * 1.25)
            soil_note = " (black soil: less frequent, monitor cracks)"
            
        else:  # loamy
            # Loamy: standard frequency
            soil_note = ""
        
        # ===== SEASONAL ADJUSTMENTS =====
        # Apply seasonal adjustments AFTER soil adjustments
        # Summer: 35% more frequent
        # Kharif (Monsoon): 40% less frequent
        # Rabi (Winter): Standard frequency
        # =================================
        
        if season == "summer":
            # Summer: 35% more frequent (multiply by 0.65)
            veg_freq = max(2, round(veg_freq * 0.65))
            flower_freq = max(1, round(flower_freq * 0.65))
            maturity_freq = max(3, round(maturity_freq * 0.65))
            season_note = " + summer heat: more frequent"
            
        elif season == "kharif":
            # Monsoon: 40% less frequent (multiply by 1.4)
            veg_freq = round(veg_freq * 1.4)
            flower_freq = round(flower_freq * 1.4)
            maturity_freq = round(maturity_freq * 1.4)
            season_note = " + monsoon: less frequent"
            
        else:  # rabi
            # Rabi: standard frequency
            season_note = ""
        
        # Combine notes for display
        full_note = soil_note + season_note
        if not full_note:
            full_note = " (standard schedule)"
        
        # Calculate average frequency
        avg_freq = round((veg_freq + flower_freq + maturity_freq) / 3)
        
        # Calculate water amounts with soil and season factors
        total_water = round(crop_requirements["total"] * area * soil_factor * season_factor)
        veg_water = round(crop_requirements["veg"] * soil_factor * season_factor)
        flower_water = round(crop_requirements["flower"] * soil_factor * season_factor)
        maturity_water = round(crop_requirements["maturity"] * soil_factor * season_factor)
        
        print(f"✅ Returning irrigation plan with total_water={total_water}")
        print("="*60 + "\n")
        
        return jsonify({
            "success": True,
            "plan": {
                "total_water": total_water,
                "area": area,
                "avg_frequency": avg_freq,
                "schedule_note": full_note,
                "durations": {
                    "veg": veg_duration,
                    "flower": flower_duration,
                    "maturity": maturity_duration
                },
                "stages": {
                    "vegetative": {
                        "water": veg_water,
                        "frequency": veg_freq,
                        "description": f"Keep soil moist. Apply {veg_water}L per acre every {veg_freq} days.{full_note}"
                    },
                    "flowering": {
                        "water": flower_water,
                        "frequency": flower_freq,
                        "description": f"Critical stage! Apply {flower_water}L per acre every {flower_freq} days.{full_note}"
                    },
                    "maturity": {
                        "water": maturity_water,
                        "frequency": maturity_freq,
                        "description": f"Reduce water. Apply {maturity_water}L per acre every {maturity_freq} days.{full_note}"
                    }
                }
            }
        })
        
    except Exception as e:
        print(f"❌ Error in irrigation plan: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
         # ========== AI-POWERED IRRIGATION TIPS API ==========
@app.route("/api/irrigation/tips", methods=["GET"])
def get_irrigation_tips():
    """Generate dynamic irrigation tips using Groq AI based on crop, soil, and season"""
    
    # Get parameters
    crop = request.args.get("crop", "").lower()
    soil = request.args.get("soil", "").lower()
    season = request.args.get("season", "").lower()
    
    # If any parameter is missing, use defaults
    if not crop:
        crop = "wheat"
    if not soil:
        soil = "loamy"
    if not season:
        season = "rabi"
    
    # Check if Groq API key is available
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY not found, using fallback tips")
        return jsonify({
            "success": True,
            "tips": [
                {
                    "icon": "💧",
                    "title": "Drip Irrigation Recommended",
                    "description": "Drip systems deliver water directly to plant roots, reducing evaporation and water waste by up to 50%."
                },
                {
                    "icon": "🌍",
                    "title": "Monitor Soil Moisture",
                    "description": "Use soil moisture sensors to measure actual water content. Irrigate when moisture drops below 50% of field capacity."
                },
                {
                    "icon": "⚠️",
                    "title": "Avoid Overwatering",
                    "description": "Excess water causes root diseases and nutrient loss. Follow recommended schedule based on crop and soil type."
                }
            ]
        })
    
    try:
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # Create prompt for AI
        prompt = f"""You are an expert agricultural advisor for Indian farmers. Generate 3 specific, practical irrigation tips for {crop} crop grown in {soil} soil during {season} season.

Requirements for each tip:
1. First tip: Focus on irrigation METHOD (drip, sprinkler, flood) suitable for {soil} soil
2. Second tip: Focus on CRITICAL GROWTH STAGES for {crop} crop
3. Third tip: Focus on SEASONAL adjustments for {season} season with {soil} soil

Each tip must:
- Be actionable and specific (include numbers like days, mm of water, etc.)
- Use simple language farmers can understand
- Consider Indian farming conditions
- Include an appropriate emoji

Return ONLY a valid JSON array with 3 objects. Each object must have these exact keys:
- "icon": a single emoji string
- "title": a short, catchy title (max 6 words)
- "description": a detailed tip (2-3 sentences)

Example format:
[
  {{
    "icon": "💧",
    "title": "Drip Irrigation for Sandy Soil",
    "description": "Sandy soil drains fast. Use drip irrigation with mulch to reduce water loss by 40%. Water every 3-4 days with 20mm each time."
  }},
  {{
    "icon": "🌾",
    "title": "Wheat Critical Stages",
    "description": "Wheat needs irrigation at crown root initiation (20-25 days), tillering, flowering, and grain filling. Don't miss these 4 irrigations."
  }},
  {{
    "icon": "☀️",
    "title": "Summer Care for Loamy Soil",
    "description": "In summer, irrigate early morning to reduce evaporation. Apply 40mm every 5-6 days. Use straw mulch to retain moisture."
  }}
]

Generate tips now for {crop} in {soil} soil during {season} season:"""
        
        print(f"🤖 Generating AI tips for: {crop}, {soil}, {season}")
        
        # Make API call to Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert agricultural advisor. Always return valid JSON arrays only, no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            stream=False,
        )
        
        # Get the response
        ai_response = completion.choices[0].message.content
        print(f"✅ AI response received: {ai_response[:100]}...")
        
        # Parse JSON from response
        import json
        import re
        
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        if json_match:
            tips = json.loads(json_match.group())
        else:
            tips = json.loads(ai_response)
        
        # Ensure we have exactly 3 tips
        if len(tips) != 3:
            # Pad or truncate to 3
            tips = (tips + [tips[0], tips[1]])[:3]
        
        return jsonify({"success": True, "tips": tips})
        
    except Exception as e:
        print(f"❌ Error generating AI tips: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback to dynamic hardcoded tips based on parameters
        print("⚠️ Using fallback tips")
        
        # Soil-based tips
        soil_tips = {
            "sandy": "Sandy soil drains quickly. Use drip irrigation with mulching. Apply water in frequent, light sessions.",
            "loamy": "Loamy soil is ideal. Maintain regular irrigation schedule. Monitor at 15cm depth before watering.",
            "clay": "Clay soil holds water. Reduce frequency to prevent waterlogging. Check for standing water.",
            "black": "Black soil cracks when dry. Maintain consistent moisture. Apply water when cracks appear."
        }
        
        # Crop-based tips
        crop_tips = {
            "wheat": "Critical stages: crown root initiation (20-25 days), tillering, flowering, grain filling.",
            "rice": "Maintain 5-10cm standing water during vegetative stage. Drain 2 weeks before harvest.",
            "maize": "Most sensitive at tasseling and silking. Don't miss irrigation during these stages.",
            "cotton": "More water needed during flowering and boll development. Reduce after boll opening.",
            "sugarcane": "Maximum water during grand growth phase (120-270 days). Keep soil near field capacity.",
            "soybean": "Critical stages: flowering and pod filling. Water stress here reduces yield.",
            "chickpea": "Drought-tolerant. Irrigate at branching and pod development. Avoid waterlogging.",
            "tomato": "Consistent moisture prevents blossom end rot. Drip irrigation is ideal.",
            "potato": "Critical: tuber initiation and bulking. Keep soil moist but not waterlogged.",
            "onion": "Shallow roots need frequent light irrigation. Stop 10-15 days before harvest."
        }
        
        # Season-based tips
        season_tips = {
            "kharif": {
                "sandy": "Monsoon: Sandy soil needs drainage channels. Apply fertilizers in splits.",
                "loamy": "Monsoon: Create drainage channels. Delay irrigation after heavy rain.",
                "clay": "Monsoon: High waterlogging risk. Avoid irrigation for 3-4 days after rain.",
                "black": "Monsoon: Soil swells when wet. Avoid walking in fields after rain."
            },
            "rabi": {
                "sandy": "Winter: Irrigate in morning to prevent frost. Use light irrigation.",
                "loamy": "Winter: Irrigate before 9 AM. Reduce frequency by 20% in Dec-Jan.",
                "clay": "Winter: Extend interval by 3-4 days. Watch for surface cracking.",
                "black": "Winter: Light irrigation maintains moisture. Avoid heavy evening watering."
            },
            "summer": {
                "sandy": "Summer: Increase frequency by 40%. Use mulch. Irrigate early morning.",
                "loamy": "Summer: Evening irrigation reduces evaporation. Use sprinklers.",
                "clay": "Summer: Deep cracks appear. Apply heavy irrigation. Use mulch.",
                "black": "Summer: Deep cracks. Apply organic mulch. Irrigate before 8 AM."
            }
        }
        
        # Build fallback tips
        tip1 = {
            "icon": "💧",
            "title": f"Best Irrigation for {soil.title()} Soil",
            "description": soil_tips.get(soil, "Choose irrigation method based on soil type and crop needs.")
        }
        
        tip2 = {
            "icon": "🌾" if crop in ["wheat", "rice", "maize"] else "🌱",
            "title": f"{crop.title()} Critical Stages",
            "description": crop_tips.get(crop, "Monitor crop growth stages for optimal irrigation timing.")
        }
        
        season_key = season if season in season_tips else "rabi"
        soil_key = soil if soil in season_tips[season_key] else "loamy"
        tip3 = {
            "icon": "🌧️" if season == "kharif" else "☀️" if season == "summer" else "❄️",
            "title": f"{season.title()} Season Tips for {soil.title()} Soil",
            "description": season_tips.get(season, {}).get(soil, "Adjust irrigation based on seasonal conditions.")
        }
        
        return jsonify({"success": True, "tips": [tip1, tip2, tip3]})

       # ========== SEASONALITY CHECK API ==========
@app.route("/api/seasonality/check", methods=["POST"])
def check_seasonality():
    """
    Check seasonal suitability for a crop based on region, month, and soil
    """
    try:
        data = request.get_json()
        
        crop_slug = data.get("crop")
        region = data.get("region")
        month = data.get("month")
        soil = data.get("soil")
        
        if not crop_slug or not region or not month or not soil:
            return jsonify({
                "success": False,
                "message": "Missing required parameters"
            }), 400
        
        # Get crop data from crops.json
        if crop_slug not in crops_data:
            return jsonify({
                "success": False,
                "message": "Crop not found"
            }), 404
        
        crop_info = crops_data[crop_slug]
        category = crop_info.get("category", "Unknown")
        
        # Determine season based on month
        month_lower = month.lower()
        
        # Indian cropping seasons
        kharif_months = ["june", "july", "august", "september", "october"]
        rabi_months = ["october", "november", "december", "january", "february", "march"]
        summer_months = ["march", "april", "may", "june"]
        
        # Determine current season
        if month_lower in kharif_months:
            current_season = "Kharif (Monsoon)"
            current_season_code = "kharif"
        elif month_lower in rabi_months:
            current_season = "Rabi (Winter)"
            current_season_code = "rabi"
        elif month_lower in summer_months:
            current_season = "Summer (Zaid)"
            current_season_code = "summer"
        else:
            current_season = "Unknown"
            current_season_code = "unknown"
        
        # Determine ideal season for crop based on category
        ideal_season_map = {
            "Cereal": "Rabi (Oct-Mar)" if crop_slug in ["wheat", "barley"] else "Kharif (Jun-Oct)",
            "Millet": "Kharif (Jun-Oct)",
            "Pulse": "Rabi (Oct-Mar)" if crop_slug in ["chickpea", "lentil", "peas"] else "Kharif (Jun-Oct)",
            "Oilseed": "Rabi (Oct-Mar)" if crop_slug in ["mustard", "linseed"] else "Kharif (Jun-Oct)",
            "Vegetable": "Varies by region",
            "Fruit": "Perennial - can be planted in multiple seasons",
            "Cash Crop": "Kharif (Jun-Oct)" if crop_slug in ["cotton", "sugarcane"] else "Rabi (Oct-Mar)",
            "default": "Rabi (Oct-Mar)"
        }
        
        ideal_season = ideal_season_map.get(category, ideal_season_map["default"])
        
        # Calculate suitability score (0-100)
        suitability_score = 0
        suitability_label = ""
        suitability_icon = ""
        
        # Simple rule-based scoring
        if category == "Cereal":
            if crop_slug in ["wheat", "barley"] and current_season_code == "rabi":
                suitability_score = 95
            elif crop_slug in ["rice", "maize"] and current_season_code == "kharif":
                suitability_score = 90
            else:
                suitability_score = 40
        elif category == "Millet":
            if current_season_code == "kharif":
                suitability_score = 85
            else:
                suitability_score = 50
        elif category == "Pulse":
            if crop_slug in ["chickpea", "lentil", "peas"] and current_season_code == "rabi":
                suitability_score = 90
            elif crop_slug in ["green-gram", "black-gram", "pigeon-pea"] and current_season_code == "kharif":
                suitability_score = 85
            else:
                suitability_score = 45
        elif category == "Oilseed":
            if crop_slug in ["mustard", "linseed"] and current_season_code == "rabi":
                suitability_score = 90
            elif crop_slug in ["groundnut", "sunflower", "sesame"] and current_season_code == "kharif":
                suitability_score = 85
            else:
                suitability_score = 50
        elif category == "Vegetable":
            # Vegetables can be grown in multiple seasons with proper management
            if current_season_code == "rabi":
                suitability_score = 80
            elif current_season_code == "summer":
                suitability_score = 75
            else:
                suitability_score = 70
        elif category == "Fruit":
            suitability_score = 85  # Fruits are perennial
        elif category == "Cash Crop":
            if crop_slug in ["cotton", "sugarcane"] and current_season_code == "kharif":
                suitability_score = 90
            else:
                suitability_score = 60
        else:
            suitability_score = 50
        
        # Determine label and icon based on score
        if suitability_score >= 80:
            suitability_label = "Highly Suitable"
            suitability_icon = "✅"
        elif suitability_score >= 60:
            suitability_label = "Moderately Suitable"
            suitability_icon = "⚠️"
        else:
            suitability_label = "Not Suitable"
            suitability_icon = "❌"
        
        # Soil suitability note
        soil_notes = {
            "sandy": "Sandy soil requires more frequent irrigation and nutrients may leach quickly.",
            "loamy": "Loamy soil is ideal for most crops with good water retention and drainage.",
            "clay": "Clay soil holds water well but requires careful drainage to prevent waterlogging.",
            "blacksoil": "Black soil is rich in nutrients but cracks when dry; maintain consistent moisture."
        }
        soil_note = soil_notes.get(soil, "Consider soil testing for optimal results.")
        
        # Risk advisory
        risk_advisory = ""
        if suitability_score < 60:
            risk_advisory = f"This crop may not perform well in {current_season}. Consider alternative crops."
        elif current_season_code == "kharif" and soil in ["clay", "blacksoil"]:
            risk_advisory = "High rainfall risk. Ensure proper drainage to prevent waterlogging."
        elif current_season_code == "summer" and soil == "sandy":
            risk_advisory = "High temperature and sandy soil will require frequent irrigation. Consider mulching."
        else:
            risk_advisory = "Favorable conditions expected. Follow standard agricultural practices."
        
        # Sowing window
        sowing_windows = {
            "kharif": "June to July",
            "rabi": "October to November",
            "summer": "February to March"
        }
        sowing_window = sowing_windows.get(current_season_code, "Varies by region")
        
        return jsonify({
            "success": True,
            "result": {
                "crop": crop_info.get("name", crop_slug),
                "region": region,
                "current_season": current_season,
                "ideal_season": ideal_season,
                "suitability": {
                    "score": suitability_score,
                    "label": suitability_label,
                    "icon": suitability_icon
                },
                "sowing_window": sowing_window,
                "climate_notes": f"Temperature: 15-30°C | Rainfall: 500-800mm | Humidity: 40-80%",
                "soil_note": soil_note,
                "risk_advisory": risk_advisory
            }
        })
        
    except Exception as e:
        print(f"Error in seasonality check: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500 

# ========== AI CROP RECOMMENDATIONS ==========
@app.route("/api/crops/recommend", methods=["POST"])
def recommend_crops():
    """Get AI-powered crop recommendations based on region and month"""
    try:
        data = request.get_json()
        region = data.get("region")
        month = data.get("month")
        
        if not region or not month:
            return jsonify({
                "success": False,
                "message": "Region and month are required"
            }), 400
        
        # Check if Groq API key is available
        if not GROQ_API_KEY:
            return jsonify({
                "success": False,
                "message": "AI service not configured"
            }), 500
        
        # Initialize Groq client
        client = Groq(api_key=GROQ_API_KEY)
        
        # Create prompt for AI
        prompt = f"""You are an expert agricultural advisor for Indian farmers. Based on the region '{region}' and month '{month}', recommend the top 5 crops to plant.

For each crop, provide:
1. Crop name
2. Why it's suitable for this region and month
3. Expected yield
4. Key growing tips
5. Confidence score (as a percentage)

Return ONLY a valid JSON array with 5 objects. Each object must have these exact keys:
- "name": crop name
- "reason": brief explanation of suitability (1 sentence)
- "yield": expected yield (e.g., "25-30 quintals/acre")
- "tips": one key growing tip
- "confidence": number between 70-99

Example format:
[
  {{
    "name": "Wheat",
    "reason": "Wheat thrives in Punjab's winter months with ideal temperature range of 15-25°C",
    "yield": "28-32 quintals/acre",
    "tips": "Sow at 20-25 cm spacing and apply nitrogen at crown root initiation",
    "confidence": 95
  }}
]

Base your recommendations on real agricultural science for Indian conditions. Consider:
- Typical crops grown in {region} during {month}
- Climate suitability
- Market demand
- Water requirements

Generate recommendations now:"""
        
        print(f"🤖 Getting AI recommendations for {region} in {month}")
        
        # Make API call to Groq
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert agricultural advisor. Return only valid JSON with no other text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        # Get the response
        ai_response = completion.choices[0].message.content
        print(f"✅ AI response received")
        
        # Parse JSON from response
        import json
        import re
        
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
        if json_match:
            recommendations = json.loads(json_match.group())
        else:
            recommendations = json.loads(ai_response)
        
        # Ensure we have at least 3 recommendations
        if len(recommendations) < 3:
            recommendations = recommendations[:3]
        else:
            recommendations = recommendations[:5]
        
        return jsonify({
            "success": True,
            "region": region,
            "month": month,
            "recommendations": recommendations
        })
        
    except Exception as e:
        print(f"❌ Error in recommendations: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback recommendations
        fallback = [
            {
                "name": "Wheat",
                "reason": "Well-suited for rabi season with optimal temperature range",
                "yield": "25-30 quintals/acre",
                "tips": "Ensure proper irrigation at crown root initiation",
                "confidence": 85
            },
            {
                "name": "Mustard",
                "reason": "Grows well in winter months with minimal water requirement",
                "yield": "12-15 quintals/acre",
                "tips": "Maintain row spacing of 30 cm for better yield",
                "confidence": 82
            },
            {
                "name": "Chickpea",
                "reason": "Drought-tolerant pulse crop ideal for rabi season",
                "yield": "10-12 quintals/acre",
                "tips": "Treat seeds with rhizobium culture before sowing",
                "confidence": 80
            }
        ]
        
        return jsonify({
            "success": True,
            "region": region,
            "month": month,
            "recommendations": fallback,
            "note": "Using fallback recommendations"
        }), 200
# ========== RUN APP ==========
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    socketio.run(app, host="0.0.0.0", port=port)










import os
import requests
import math
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

from database import init_db, save_location_record, get_recent_locations
from cv_engine import generate_satellite_imagery
from ml_engine import regional_ml_predictor
from sentinel_service import sentinel_service
from huggingface_service import hf_service

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Initialize SQLite database
init_db()

@app.route("/", methods=["GET"])
def index():
    """Serves the complete frontend directly on localhost:5000"""
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "ONLINE",
        "service": "Sentinel Earth Regional ML/DL Agriculture & CV Platform API v4.0",
        "version": "4.0.0",
        "endpoints": [
            "/api/analyze-location [POST]",
            "/api/hf-chat [POST]",
            "/api/history [GET]",
            "/api/sentinel-status [GET]"
        ]
    })

def fetch_live_weather(lat, lon):
    """Fetch real weather from Open-Meteo free API"""
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,rain,wind_speed_10m,precipitation"
            f"&hourly=soil_moisture_0_to_7cm"
            f"&forecast_days=1"
        )
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            d = resp.json()
            cur = d.get("current", {})
            hourly = d.get("hourly", {})
            soil_moist = hourly.get("soil_moisture_0_to_7cm", [38.5])
            sm_val = round(float(soil_moist[0] if soil_moist else 38.5) * 100, 1)
            sm_val = max(5.0, min(95.0, sm_val))
            return {
                "temperature_c": round(float(cur.get("temperature_2m", 28.5)), 1),
                "humidity_percent": round(float(cur.get("relative_humidity_2m", 65)), 1),
                "rain_mm": round(float(cur.get("rain", 0.0)), 2),
                "wind_kmh": round(float(cur.get("wind_speed_10m", 12.0)), 1),
                "soil_moisture_pct": sm_val,
                "source": "Open-Meteo Live API"
            }
    except Exception as e:
        print("Weather API exception:", str(e))

    # Realistic fallback based on lat/lon
    abs_lat = abs(lat)
    temp = round(float(30.0 - (abs_lat * 0.55) + ((lon % 8) * 0.3)), 1)
    temp = max(-50.0, min(55.0, temp))
    hum = round(float(60.0 + (abs(math.sin(lat * 0.15) * 30.0) + (lon % 15))), 1)
    hum = max(5.0, min(99.0, hum))
    return {
        "temperature_c": temp,
        "humidity_percent": hum,
        "rain_mm": round(float(abs(math.sin(lat * 2.5 + lon)) * 8.0), 1),
        "wind_kmh": round(float(8.0 + (abs_lat % 20)), 1),
        "soil_moisture_pct": round(float(25.0 + abs(math.sin(lat * 0.2) * 45.0)), 1),
        "source": "Sentinel Regional Grid (Fallback)"
    }

@app.route("/api/analyze-location", methods=["POST"])
def analyze_location():
    data = request.json or {}
    lat = float(data.get("lat", 11.0168))
    lon = float(data.get("lon", 76.9558))
    loc_name = data.get("name", "Selected Region")
    country = data.get("country", "Global Earth")
    climate_stress_flag = bool(data.get("climate_stress", False))
    factory_dist_km = data.get("factory_dist_km", None)
    if factory_dist_km is not None:
        factory_dist_km = float(factory_dist_km)

    # 1. Live Weather Data
    weather = fetch_live_weather(lat, lon)

    # 2. Sentinel Satellite Data
    sentinel_data = sentinel_service.fetch_sentinel_data(lat, lon)

    temp_c = weather["temperature_c"]
    humidity = weather["humidity_percent"]
    rain_mm = weather["rain_mm"] * 90.0 + 400.0
    wind_kmh = weather.get("wind_kmh", 12.0)
    soil_moisture_from_weather = weather.get("soil_moisture_pct", None)

    # 3. Derive Soil & Water metrics from Sentinel
    soil_ph = sentinel_data["live_metrics"]["soil_ph"]
    underground_water_depth = sentinel_data["live_metrics"]["water_depth"]
    soil_moisture = soil_moisture_from_weather if soil_moisture_from_weather else sentinel_data["live_metrics"]["soil_moisture_pct"]

    # 4. Computer Vision: Satellite Image + NDVI
    cv_result = generate_satellite_imagery(lat, lon)
    dominant_ndvi = cv_result["dominant_ndvi"]

    # 5. ML Crop Yield Prediction
    ml_result = regional_ml_predictor.predict(
        lat=lat, lon=lon,
        temp_c=temp_c, soil_ph=soil_ph,
        rainfall_mm=rain_mm, ndvi_avg=dominant_ndvi["score"],
        factory_dist_km=factory_dist_km,
        climate_stress=climate_stress_flag,
        loc_name=loc_name
    )

    # 6. Save to SQLite DB
    try:
        save_location_record(
            lat=lat, lon=lon, name=loc_name, country=country,
            temp=temp_c, humidity=humidity, soil_ph=soil_ph,
            water_depth=underground_water_depth, ndvi=dominant_ndvi["score"],
            yield_acc=ml_result["yield_accuracy"],
            crop=ml_result["recommended_crop"]
        )
    except Exception as db_err:
        print("DB Save Error:", db_err)

    return jsonify({
        "location": {
            "lat": lat, "lon": lon, "name": loc_name, "country": country,
            "type": ml_result["location_type"]
        },
        "live_telemetry": {
            "temperature": f"{temp_c} °C",
            "humidity": f"{humidity} %",
            "soil_moisture": f"{round(soil_moisture, 1)} %",
            "soil_ph": f"{soil_ph}",
            "underground_water_depth": f"{underground_water_depth} m",
            "rainfall": f"{round(weather['rain_mm'], 1)} mm/hr",
            "wind": f"{wind_kmh} km/h"
        },
        "historical_1_year_comparison": sentinel_data["historical_1_year_ago"],
        "computer_vision": cv_result,
        "sentinel_satellites": sentinel_data,
        "ml_crop_yield_prediction": ml_result
    })

@app.route("/api/hf-chat", methods=["POST"])
def hf_chat():
    data = request.json or {}
    prompt = data.get("prompt", "Recommend crops for this location")
    context = data.get("context", None)
    user_key = data.get("hf_api_key", None)
    res = hf_service.chat_query(prompt, location_context=context, user_api_key=user_key)
    return jsonify(res)

@app.route("/api/history", methods=["GET"])
def get_history():
    history = get_recent_locations(15)
    return jsonify({"history": history})

@app.route("/<path:path>", methods=["GET"])
def static_proxy(path):
    """Serves any static frontend files or returns index.html"""
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")

if __name__ == "__main__":
    print("=" * 60)
    print("  SENTINEL-EARTH AI Platform v4.0 - Flask Backend")
    print("  Unified Frontend + Backend on: http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)

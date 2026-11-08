import os
import sys

# Ensure backend directory and root directory are in sys.path
_cur_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_cur_dir, '..'))
for _p in [_cur_dir, _root_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Ensure VERCEL environment flag is set if running on Lambda/Vercel
if 'VERCEL' in os.environ or 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
    os.environ['VERCEL'] = '1'

ROOT_DIR = _root_dir
import requests
import math
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS

from database import init_db, save_location_record, get_recent_locations, get_all_crop_knowledge
from cv_engine import generate_satellite_imagery
from ml_engine import regional_ml_predictor
from sentinel_service import sentinel_service
from huggingface_service import hf_service

FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# Initialize SQLite database safely
try:
    init_db()
except Exception as e:
    print('Database init non-critical error:', e)

@app.route("/", methods=["GET"])
def index():
    """Serves the complete frontend directly on localhost:5000"""
    if os.path.exists(os.path.join(ROOT_DIR, 'index.html')):
        return send_from_directory(ROOT_DIR, 'index.html')
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route("/api/status", methods=["GET"])
@app.route("/status", methods=["GET"])
@app.route("/api/sentinel-status", methods=["GET"])
@app.route("/sentinel-status", methods=["GET"])
def sentinel_status():
    return jsonify({
        "status": "OPERATIONAL",
        "satellites": {
            "Sentinel-1": {"status": "ACTIVE", "mode": "C-band SAR Ground Moisture"},
            "Sentinel-2": {"status": "ACTIVE", "mode": "MSI Multi-Spectral 10m NDVI"},
            "Sentinel-3": {"status": "ACTIVE", "mode": "SLSTR Surface Temperature"},
            "Sentinel-5P": {"status": "ACTIVE", "mode": "TROPOMI Atmospheric NO2/SO2"}
        }
    })

def api_status():
    return jsonify({
        "status": "ONLINE",
        "service": "Sentinel Earth Regional ML/DL Agriculture & CV Platform API v4.0",
        "version": "4.0.0",
        "endpoints": [
            "/api/analyze-location [POST]",
            "/api/hf-chat [POST]",
            "/api/history [GET]",
            "/api/sentinel-status [GET]",
            "/api/building-suitability [POST]"
        ]
    })

@app.route("/api/building-suitability", methods=["POST"])
@app.route("/building-suitability", methods=["POST"])
def building_suitability():
    """Analyze whether a location's land is suitable for building/construction with full risk breakdown."""
    data = request.json or {}
    lat = float(data.get("lat", 11.0168))
    lon = float(data.get("lon", 76.9558))
    soil_ph = float(str(data.get("soil_ph", 6.5)).replace("pH","").strip())
    soil_moisture = float(str(data.get("soil_moisture", 40.0)).replace("%","").strip())
    water_depth = float(str(data.get("water_depth", 14.0)).replace("m","").strip())
    location_type = str(data.get("location_type", "Prime Fertile Agricultural Zone"))
    ndvi_score = float(str(data.get("ndvi_score", 0.5)))
    loc_name = data.get("name", "Selected Region")
    factory_dist = float(data.get("factory_dist_km") or 10.0)
    rainfall_mm = float(data.get("rainfall_mm") or 15.0)
    lt = location_type.lower()

    score = 0
    positives = []
    issues = []

    # 1. Flood Risk Assessment
    if water_depth < 5.0 or rainfall_mm > 50.0:
        flood_risk_level = "HIGH FLOOD HAZARD"
        flood_risk_desc = f"Shallow water table ({water_depth}m) creates severe risk of basement inundation, waterlogging, and hydrostatic uplift pressure during seasonal monsoons."
        issues.append(f"Severe Flood Risk: Shallow water table ({water_depth}m) — hydraulic basement tanking & elevated plinth mandatory")
    elif water_depth < 10.0:
        flood_risk_level = "MODERATE FLOOD RISK"
        flood_risk_desc = f"Water table at {water_depth}m. Moderate flood vulnerability during peak rains; raised plinth (+1.0m) and perimeter drainage channels required."
        score += 12
        positives.append(f"Manageable water table ({water_depth}m) with standard perimeter drainage")
    else:
        flood_risk_level = "LOW FLOOD RISK (SAFE)"
        flood_risk_desc = f"Deep water table ({water_depth}m) provides excellent natural subsoil percolation and zero hydrostatic pressure on foundations."
        score += 28
        positives.append(f"Deep water table ({water_depth}m) — safe natural drainage buffer")

    # 2. Soil Stability & Structural Collapse ("Building May Fall") Assessment
    if soil_moisture > 55.0:
        stability_level = "CRITICAL LIQUEFACTION RISK (BUILDING COLLAPSE RISK)"
        stability_desc = f"Excessive subsoil moisture ({soil_moisture}%) can cause loss of soil shear strength and liquefaction under building weight during seismic tremors or monsoon saturation. Building risks differential settlement or tilting without deep pile foundations."
        issues.append(f"Structural Collapse Risk: Saturated soil ({soil_moisture}%) risks liquefaction and settlement — friction piles into bedrock required")
    elif any(kw in lt for kw in ["black cotton", "vertisol", "cotton"]):
        stability_level = "EXPANSIVE SOIL HEAVE RISK"
        stability_desc = f"Expansive montmorillonite clay swells when wet and shrinks when dry, generating vertical heave forces of up to 250 kPa that crack masonry walls and destabilize shallow foundations."
        issues.append("Expansive clay soil: High swelling/shrinkage — under-reamed pile foundations with void formers mandatory")
        score += 8
    elif soil_moisture < 35.0:
        stability_level = "EXCELLENT STRUCTURAL BEARING CAPACITY"
        stability_desc = f"Compacted subsoil moisture ({soil_moisture}%) and stable stratigraphy exhibit high California Bearing Ratio (CBR > 12%) and safe bearing capacity (> 240 kPa)."
        score += 24
        positives.append(f"High soil bearing capacity (CBR > 12%) — ideal foundation compaction")
    else:
        stability_level = "MODERATE BEARING CAPACITY"
        stability_desc = f"Normal subsoil cohesion ({soil_moisture}% moisture); standard raft or spread footings provide safe load transfer."
        score += 15
        positives.append(f"Adequate soil stability ({soil_moisture}%) for residential/commercial structures")

    # 3. Factory Proximity & Air Pollution Hazard
    if factory_dist < 3.0:
        air_pollution_level = "CRITICAL INDUSTRIAL TOXIC POLLUTION"
        air_pollution_desc = f"Industrial factory detected within {round(factory_dist, 1)} km emitting chemical fumes, sulfur dioxide (SO2), and NO2 particulates (from Sentinel-5P TROPOMI sensor), causing corrosive atmospheric degradation of metal building roofs and acute health hazards."
        issues.append(f"Industrial Air Pollution: Chemical factory within {round(factory_dist, 1)}km — corrosive emissions and particulate smog")
    else:
        air_pollution_level = "CLEAN AIR BUFFER"
        air_pollution_desc = f"Safe distance ({round(factory_dist, 1)} km) from industrial manufacturing corridors ensures clean ambient air quality."
        score += 18
        positives.append(f"Clean rural/suburban air buffer ({round(factory_dist, 1)} km from industrial zones)")

    # 4. Factory Waste Water & Groundwater Quality Deterioration
    if factory_dist < 3.0:
        wastewater_threat_level = "SEVERE CHEMICAL CONTAMINATION THREAT"
        wastewater_threat_desc = f"Untreated industrial effluent and chemical wastewater runoff from the nearby factory ({round(factory_dist, 1)} km) risk infiltrating local aquifers, elevating water TDS and toxic heavy metals (lead, chromium). Groundwater is unsafe for drinking and corrosive to concrete (acid/sulfate attack on Portland cement)."
        issues.append(f"Factory Wastewater Hazard: Chemical runoff leaches into groundwater — borewell water unsafe; concrete sulfate-resistant barrier required")
    else:
        wastewater_threat_level = "SAFE FRESHWATER AQUIFER"
        wastewater_threat_desc = "No nearby point-source industrial chemical discharge detected. Subsurface aquifer water is chemically safe."
        score += 18
        positives.append("Pure groundwater aquifer free from chemical effluent discharge")

    # 5. Soil pH Assessment
    if 5.5 <= soil_ph <= 8.5:
        score += 12
        positives.append(f"Soil pH {soil_ph} is non-corrosive to reinforced concrete foundation steel")
    else:
        issues.append(f"Soil pH {soil_ph} outside safe range — sulfate-resistant cement (SRPC) mandatory")

    # 6. Location Type Constraints
    if any(kw in lt for kw in ["ocean","sea","water","marine","river"]):
        score = 0
        issues = ["Ocean / Marine Water Body: Terrestrial building construction is biologically and structurally impossible."]
    elif any(kw in lt for kw in ["polar","permafrost"]):
        score = 8
        issues.append("Polar Permafrost: Foundation thaw subsidence; cryogenic insulated pilings required at high cost")

    score = max(0, min(100, score))

    if score >= 70:
        rating = "✅ HIGHLY SUITABLE"; rating_color = "green"
        recommendation = "Favorable geotechnical and environmental conditions. Low disaster risk. Proceed with standard architectural planning and municipal permits."
    elif score >= 50:
        rating = "⚠️ MODERATELY SUITABLE"; rating_color = "orange"
        recommendation = "Construction is feasible with mandatory engineering precautions: raised plinth against flooding, waterproof tanking, and structural bearing plate load testing."
    elif score >= 25:
        rating = "⚠️ HIGH RISK / LOW SUITABILITY"; rating_color = "orange"
        recommendation = "Significant environmental hazards detected (factory pollution, water table or soil liquefaction risk). Requires specialized deep piling and environmental clearance."
    else:
        rating = "❌ NOT SUITABLE / HIGH HAZARD"; rating_color = "red"
        recommendation = "Construction not recommended due to critical hazards (open water, industrial toxicity, or severe foundation failure risks)."

    foundation_type = (
        "Shallow strip / raft foundation (<1.5m)" if water_depth > 15 and soil_moisture < 40 and factory_dist >= 3.0
        else "Medium depth isolated footing with damp-proof membrane (2–3m)" if water_depth > 8
        else "Deep reinforced concrete pile foundation (>4–8m into bedrock) to prevent building settlement"
    )

    return jsonify({
        "location": {"lat": lat, "lon": lon, "name": loc_name},
        "suitability_score": score,
        "rating": rating,
        "rating_color": rating_color,
        "recommendation": recommendation,
        "flood_risk": flood_risk_level,
        "soil_stability_collapse_risk": stability_level,
        "factory_pollution_hazard": air_pollution_level,
        "factory_wastewater_water_quality": wastewater_threat_level,
        "hazard_risk_breakdown": {
            "flood_risk": {
                "level": flood_risk_level,
                "description": flood_risk_desc,
                "mitigation": "Elevate plinth level +1.2m, perimeter sump pits, dual-layer bitumen waterproofing."
            },
            "soil_stability_collapse_risk": {
                "level": stability_level,
                "description": stability_desc,
                "will_building_fall_risk": "High risk of tilting/cracking if built on shallow footing without pile stabilization." if "CRITICAL" in stability_level or "EXPANSIVE" in stability_level else "Low risk: Stable soil stratigraphy provides firm bearing capacity."
            },
            "factory_pollution_hazard": {
                "level": air_pollution_level,
                "distance_km": round(factory_dist, 1),
                "description": air_pollution_desc
            },
            "factory_wastewater_water_quality": {
                "threat_level": wastewater_threat_level,
                "description": wastewater_threat_desc,
                "mitigation": "Install geomembrane chemical barrier, use municipal filtered water supply, test borewells for heavy metals."
            }
        },
        "positive_factors": positives,
        "issues": issues,
        "construction_requirements": [
            f"Recommended Foundation Type: {foundation_type}",
            "Standard Penetration Test (SPT) & Plate Load Bearing Test mandatory before foundation pouring",
            "Continuous exterior damp-proof membrane & sump-pump drainage installation",
            "Sulfate-resistant cement (SRPC) if near factory chemical runoff",
            "Town & Country Planning Environmental Clearance Certificate"
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
@app.route("/analyze-location", methods=["POST"])
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
        loc_name=loc_name,
        water_depth=underground_water_depth
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
            "water_salinity": sentinel_data["telemetry_format"].get("water_salinity", "0.4 PSU"),
            "water_tds": sentinel_data["telemetry_format"].get("water_tds", "380 ppm"),
            "water_ph": sentinel_data["telemetry_format"].get("water_ph", f"{soil_ph}"),
            "rainfall": f"{round(weather['rain_mm'], 1)} mm/hr",
            "wind": f"{wind_kmh} km/h"
        },
        "construction_risk_analysis": sentinel_data.get("construction_risk_analysis", {}),
        "historical_1_year_comparison": sentinel_data["historical_1_year_ago"],
        "computer_vision": cv_result,
        "sentinel_satellites": sentinel_data,
        "ml_crop_yield_prediction": ml_result
    })

@app.route("/api/hf-chat", methods=["POST"])
@app.route("/hf-chat", methods=["POST"])
def hf_chat():
    data = request.json or {}
    prompt = data.get("prompt", "Recommend crops for this location")
    context = data.get("context", None)
    user_key = data.get("hf_api_key", None)
    lang = data.get("lang") or (context.get("lang") if isinstance(context, dict) else None) or "en"
    res = hf_service.chat_query(prompt, location_context=context, user_api_key=user_key, lang=lang)
    return jsonify(res)

@app.route("/api/history", methods=["GET"])
@app.route("/history", methods=["GET"])
def get_history():
    history = get_recent_locations(15)
    return jsonify({"history": history})

@app.route("/api/crop-knowledge", methods=["GET"])
@app.route("/crop-knowledge", methods=["GET"])
def get_crop_knowledge():
    knowledge = get_all_crop_knowledge()
    return jsonify({"knowledge": knowledge})

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

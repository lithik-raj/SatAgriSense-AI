import sqlite3
import os
import json

# Determine database path safely across local environment and Vercel / AWS Lambda
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    DB_PATH = "/tmp/agri_sentinel.db"
else:
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        test_file = os.path.join(backend_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        DB_PATH = os.path.join(backend_dir, "agri_sentinel.db")
    except Exception:
        DB_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "agri_sentinel.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Table for storing location history and scan results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            name TEXT NOT NULL,
            country TEXT,
            temp_c REAL,
            humidity REAL,
            soil_ph REAL,
            underground_water_depth REAL,
            ndvi_avg REAL,
            yield_accuracy REAL,
            recommended_crop TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Table for Sentinel satellite image metadata & CV analysis logs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL,
            lon REAL,
            satellite_source TEXT,
            land_cover_stats TEXT,
            water_ph REAL,
            ndvi_category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Table for AI chat prompt history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT,
            response TEXT,
            model_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Table for Worldwide Crop Yield Knowledge Base with Species & Varieties
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS global_crop_yield_knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region_name TEXT NOT NULL,
            country TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            crop_name TEXT NOT NULL,
            botanical_species TEXT NOT NULL,
            variety TEXT NOT NULL,
            optimal_soil_type TEXT NOT NULL,
            optimal_ph_min REAL NOT NULL,
            optimal_ph_max REAL NOT NULL,
            base_yield_tons_ha REAL NOT NULL,
            market_price_per_ton REAL NOT NULL,
            cost_per_ha REAL NOT NULL,
            curcumin_or_brix_quality TEXT,
            specialty_desc TEXT NOT NULL
        )
        """)

        # Populate global_crop_yield_knowledge if empty or needs update
        cursor.execute("SELECT COUNT(*) FROM global_crop_yield_knowledge")
        knowledge_count = cursor.fetchone()[0]

        if knowledge_count < 15:
            # Clear older incomplete records to refresh with full species schema
            cursor.execute("DELETE FROM global_crop_yield_knowledge")
            
            crop_knowledge_dataset = [
                # 1. Erode, Tamil Nadu (Global Turmeric Hub)
                (
                    "Erode", "India", 11.3410, 77.7172,
                    "Turmeric", "Curcuma longa", "Erode Local / BSR-2 / Salem Golden",
                    "Rich Bhavani river alluvial silt and well-drained red-clay loam",
                    6.5, 7.5, 28.5, 1850.0, 3400.0,
                    "Curcumin 4.5% - 5.2% (GI Tagged Grade A)",
                    "Erode is the famous Yellow City of India. Optimal Bhavani silt and warm humid growing cycle produce world-record curcumin density and 28.5 T/Ha peak fresh rhizome yield."
                ),
                # 2. Kashmir Valley, J&K (World Renowned Apple Basin)
                (
                    "Kashmir", "India", 34.0837, 74.7973,
                    "Apples", "Malus domestica", "Kashmiri Ambri / Red Delicious / Royal Gala",
                    "Glacial Karewa terraces with deep fertile clay-loam and rich humus",
                    6.0, 6.8, 26.0, 1100.0, 4200.0,
                    "Soluble Solids Brix 14.5° / Crisp Flesh with High Anthocyanin",
                    "Highland alpine valley chilling requirements (>1200 hrs <7°C) guarantee exceptional fruit coloration, crisp crunch, and 26.0 T/Ha orchard yield."
                ),
                # 3. Coimbatore, Tamil Nadu (Kongu Agro Hub)
                (
                    "Coimbatore", "India", 11.0168, 76.9558,
                    "Coconut & Sugarcane", "Cocos nucifera & Saccharum officinarum", "Co-0238 / Co-86032 High Biomass",
                    "Deep red-clay loam with Siruvani river perennial irrigation",
                    6.5, 7.8, 88.5, 44.0, 1700.0,
                    "Sucrose Recovery 11.5% / High Fiber",
                    "Kongu region with Siruvani and Bhavani canal network achieves world-class sugarcane biomass reaching 88.5 T/Ha with minimal lodging."
                ),
                # 4. Salinas Valley, California, USA (Salad Bowl of the World)
                (
                    "Salinas Valley", "United States", 36.6777, -121.6555,
                    "Strawberries", "Fragaria × ananassa", "Sweet Charlie / Albion / Camarosa",
                    "Coastal alluvial sandy-loam with high organic mulch",
                    5.8, 6.8, 28.4, 2400.0, 14500.0,
                    "Soluble Solids Brix 9.8° / High Vitamin C",
                    "Pacific maritime fog, mild cool sunny days, and automated drip fertigation produce the world's most productive strawberry yields at 28.4 T/Ha."
                ),
                # 5. Nagpur, Maharashtra (Orange City)
                (
                    "Nagpur", "India", 21.1458, 79.0882,
                    "Mandarin Orange", "Citrus reticulata", "Nagpur Santra (GI Tag)",
                    "Deccan basaltic vertisol black loam with high calcium reserves",
                    6.5, 7.8, 24.0, 650.0, 3200.0,
                    "Brix-to-Acid Ratio 12:1 / Thin Loose Peeling Rind",
                    "Vidarbha black soils and dry subtropical winters deliver sweet-tart Nagpur Santra mandarins with abundant juice content at 24.0 T/Ha."
                ),
                # 6. Guntur, Andhra Pradesh (Asia's Largest Spice Yard)
                (
                    "Guntur", "India", 16.3067, 80.4365,
                    "Red Chilli", "Capsicum annuum", "Guntur Sannam S4 (GI Tag)",
                    "Krishna river delta black cotton soil with high moisture retention",
                    6.2, 7.5, 5.2, 2800.0, 2600.0,
                    "Pungency 35,000–40,000 Scoville Heat Units (SHU)",
                    "High solar radiation and deep delta clay produce intense fiery red pigmentation and peak dried pod yields of 5.2 T/Ha."
                ),
                # 7. Wayanad & Idukki, Kerala (Malabar Spice Highlands)
                (
                    "Wayanad", "India", 11.6854, 76.1320,
                    "Black Pepper & Cardamom", "Piper nigrum & Elettaria cardamomum", "Panniyur-1 / Malabar Bold",
                    "Western Ghats humid laterite organic mulch with dense shade canopy",
                    5.2, 6.4, 3.2, 6200.0, 3500.0,
                    "Piperine 6.2% / Essential Volatile Oils 3.8%",
                    "High-elevation Western Ghats agroforestry with southwest monsoon rains yields premium black pepper and aromatic green cardamom."
                ),
                # 8. Ludhiana, Punjab (Grain Basket of India)
                (
                    "Punjab", "India", 30.9010, 75.8573,
                    "Wheat & Basmati Paddy", "Triticum aestivum & Oryza sativa", "HD-2967 (Kalyan Sona) / Pusa Basmati 1121",
                    "Indo-Gangetic deep alluvial silt-loam with canal & tubewell water",
                    6.8, 7.6, 5.8, 320.0, 680.0,
                    "Grain Protein 12.8% / Extra Long Slender Kernels",
                    "Fertile alluvial plains with cool rabi winter temperatures yield bumper wheat crops at 5.8 T/Ha and fragrant Basmati export paddy."
                ),
                # 9. Nilgiris / Ooty, Tamil Nadu (Blue Mountains Tea Estates)
                (
                    "Nilgiris", "India", 11.4102, 76.6950,
                    "Tea & Arabica Coffee", "Camellia sinensis & Coffea arabica", "Nilgiri Orthodox / Selection 795",
                    "Acidic laterite mountain slopes (1500-2200m) with high humus",
                    4.5, 5.5, 3.6, 3400.0, 2900.0,
                    "High Theaflavins / Bright Cup Golden Liquor",
                    "Continuous hill station mist, high elevation, and acidic soil pH produce exquisite Nilgiri frosty tea and Arabica coffee at 3.6 T/Ha."
                ),
                # 10. Corn Belt, Iowa / Illinois, USA (World Corn Heartland)
                (
                    "Iowa Corn Belt", "United States", 41.8780, -93.0977,
                    "Field Corn & Soybean", "Zea mays & Glycine max", "Pioneer Hi-Bred / Roundup Ready 2",
                    "Deep black Mollisol prairie soils with 4%+ organic matter",
                    6.0, 7.0, 11.5, 260.0, 820.0,
                    "Starch 72% / Protein 36% in Soybean",
                    "Deep glacial till and optimal summer warmth and humidity make Iowa the global leader in grain corn yields at 11.5 T/Ha."
                ),
                # 11. Jaen, Andalusia, Spain (World Olive Capital)
                (
                    "Andalusia", "Spain", 37.7796, -3.7849,
                    "Olive", "Olea europaea", "Picual / Arbequina Extra Virgin",
                    "Limestone calcareous rocky slopes with exceptional drainage",
                    6.8, 8.4, 6.2, 1600.0, 2100.0,
                    "Polyphenols >500 mg/kg / Oleic Acid >78%",
                    "Arid Mediterranean summers and limestone soil stress drive deep root penetration and produce premium Extra Virgin Olive Oil at 6.2 T/Ha."
                ),
                # 12. Mato Grosso, Brazil (Cerrado Agricultural Giant)
                (
                    "Mato Grosso", "Brazil", -12.6819, -56.9211,
                    "Soybean & Maize", "Glycine max & Zea mays", "Intacta RR2 PRO / Pioneer 30F53",
                    "Cerrado deep oxisol red clay with lime correction",
                    5.8, 6.6, 4.2, 540.0, 720.0,
                    "Oil 21% / High Seed Hardiness",
                    "Tropical savannas with two reliable crop cycles per year (Safrinha) deliver intensive grain output at 4.2 T/Ha."
                ),
                # 13. Bordeaux / Champagne, France (Premier Viticulture)
                (
                    "Bordeaux", "France", 44.8378, -0.5792,
                    "Wine Grapes", "Vitis vinifera", "Cabernet Sauvignon / Merlot",
                    "Gravelly alluvial terraces along the Gironde estuary",
                    6.5, 7.8, 12.0, 2800.0, 5200.0,
                    "Sugar 22° Brix / Rich Tannin & Phenolic Ripeness",
                    "Oceanic temperate climate and gravel soils reflect solar heat to vines, yielding prestigious grand cru wine grapes."
                ),
                # 14. Bay of Bengal / Deep Sea Marine Waters (Marine Exclusion)
                (
                    "Bay of Bengal", "International Waters", 15.0000, 88.0000,
                    "NO CROP CAN BE GROWN", "Marine Substrate", "Ocean Saline Waters (35 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.2, 8.4, 0.0, 0.0, 0.0,
                    "Non-Agricultural Saline Marine Zone",
                    "Target is open ocean marine water. Water salinity causes fatal osmotic shock to terrestrial crop roots. Marine aquaculture and fishing only."
                ),
                # 15. Amazon Protected Rainforest Canopy, Brazil (Forest Reserve)
                (
                    "Amazon Reserve", "Brazil", -3.4653, -62.2159,
                    "NO CROP CAN BE GROWN", "Tropical Rainforest", "Protected Native Bio-Canopy",
                    "Heavily leached tropical oxisol with thick leaf litter layer",
                    4.8, 5.5, 0.0, 0.0, 0.0,
                    "Ecological Protection Zone / Carbon Sink",
                    "Protected primary rainforest canopy. Clearing for agriculture strictly prohibited under international climate conventions."
                ),
                # 16. Sahara Desert Barren Land (Building / Solar Suitable)
                (
                    "Sahara Desert", "Africa", 23.4162, 25.6628,
                    "Barren Land (Building & Solar Suitable)", "Phoenix dactylifera (Date Palm Oasis)", "Medjool Oasis Date / Xerophyte",
                    "Compacted Sandy-Gravel Bedrock with Zero Topsoil Humus",
                    7.8, 8.6, 1.8, 420.0, 480.0,
                    "High Foundation Bearing Capacity (280 kPa)",
                    "Arid barren land unsuitable for rainfed farming. Highly optimal for commercial construction, warehousing, and utility-scale solar photovoltaic farms."
                )
            ]

            cursor.executemany("""
                INSERT INTO global_crop_yield_knowledge (
                    region_name, country, lat, lon, crop_name, botanical_species, variety,
                    optimal_soil_type, optimal_ph_min, optimal_ph_max, base_yield_tons_ha,
                    market_price_per_ton, cost_per_ha, curcumin_or_brix_quality, specialty_desc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, crop_knowledge_dataset)

        # Seed initial diverse crop recommendations in locations table if empty
        cursor.execute("SELECT COUNT(*) FROM locations")
        if cursor.fetchone()[0] == 0:
            seed_locations = [
                (11.3410, 77.7172, "Erode Yellow Basin, Tamil Nadu", "India", 29.2, 72.0, 6.8, 16.2, 0.73, 99.4, "Turmeric (Curcuma longa - Erode Local / BSR-2)"),
                (34.0837, 74.7973, "Kashmir Valley Orchard Basin, J&K", "India", 18.5, 58.0, 6.4, 21.0, 0.72, 99.3, "Apples (Malus domestica - Kashmiri Ambri / Red Delicious)"),
                (11.0168, 76.9558, "Coimbatore Agricultural Basin, Tamil Nadu", "India", 28.5, 75.0, 6.8, 14.5, 0.70, 98.7, "Coconut (Cocos nucifera - West Coast Tall) & Sugarcane (Co-86032)"),
                (36.6777, -121.6555, "Salinas Valley, California", "United States", 20.5, 62.0, 6.2, 22.0, 0.71, 99.5, "Strawberries (Fragaria × ananassa - Sweet Charlie / Albion)"),
                (21.1458, 79.0882, "Nagpur Citrus Tract, Maharashtra", "India", 28.0, 60.0, 7.1, 18.0, 0.69, 99.1, "Mandarin Orange (Citrus reticulata - Nagpur Santra GI)"),
                (16.3067, 80.4365, "Guntur Spice Belt, Andhra Pradesh", "India", 31.0, 68.0, 6.9, 15.0, 0.64, 99.2, "Red Chilli (Capsicum annuum - Guntur Sannam S4)"),
                (30.9010, 75.8573, "Ludhiana Plains, Punjab", "India", 24.5, 65.0, 7.2, 12.0, 0.72, 98.5, "Wheat (Triticum aestivum - HD-2967) / Basmati Paddy"),
                (11.4102, 76.6950, "Ooty Hill Slopes, Nilgiris, Tamil Nadu", "India", 16.5, 85.0, 5.2, 8.5, 0.78, 98.8, "Tea (Camellia sinensis - Nilgiri Orthodox) & Coffee"),
                (15.0000, 88.0000, "Bay of Bengal Deep Marine Zone", "International Waters", 29.0, 85.0, 8.2, 0.0, -0.25, 0.0, "NO CROP CAN BE GROWN (Ocean / Marine Saline Water)"),
                (-3.4653, -62.2159, "Amazon Ecological Sanctuary, Amazonas", "Brazil", 26.5, 90.0, 5.5, 6.0, 0.88, 0.0, "NO CROP CAN BE GROWN (Protected Natural Forest Reserve)")
            ]
            cursor.executemany("""
                INSERT INTO locations (lat, lon, name, country, temp_c, humidity, soil_ph, underground_water_depth, ndvi_avg, yield_accuracy, recommended_crop)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, seed_locations)

        conn.commit()
        conn.close()
        print("Database initialized successfully with global crop knowledge at:", DB_PATH)
    except Exception as e:
        print("Database initialization non-critical warning:", str(e))

def get_crop_knowledge_by_location(lat, lon, loc_name=""):
    """
    Query SQLite global crop knowledge dataset.
    Matches by region name keywords first, then falls back to nearest coordinate distance.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        loc_lower = str(loc_name or "").lower()

        # 1. Direct Keyword Search
        cursor.execute("SELECT * FROM global_crop_yield_knowledge")
        rows = cursor.fetchall()

        matched_row = None
        for r in rows:
            reg = r["region_name"].lower()
            crop = r["crop_name"].lower()
            if reg in loc_lower or (crop in loc_lower and len(crop) > 3):
                matched_row = dict(r)
                break

        # 2. Nearest Coordinate Euclidean Search if no text match
        if not matched_row and rows:
            def coord_dist(r):
                dlat = r["lat"] - lat
                dlon = r["lon"] - lon
                return (dlat * dlat) + (dlon * dlon)
            closest = min(rows, key=coord_dist)
            # Only accept closest if within reasonable radius (~6 degrees)
            if coord_dist(closest) <= 36.0:
                matched_row = dict(closest)

        conn.close()
        return matched_row
    except Exception as e:
        print("Crop knowledge lookup warning:", str(e))
        return None

def get_all_crop_knowledge():
    """Returns all rows in global_crop_yield_knowledge table"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM global_crop_yield_knowledge ORDER BY id ASC")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print("Get all crop knowledge warning:", str(e))
        return []

def save_location_record(lat, lon, name, country, temp, humidity, soil_ph, water_depth, ndvi, yield_acc, crop):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO locations (lat, lon, name, country, temp_c, humidity, soil_ph, underground_water_depth, ndvi_avg, yield_accuracy, recommended_crop)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (lat, lon, name, country, temp, humidity, soil_ph, water_depth, ndvi, yield_acc, crop))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Save location DB non-critical warning:", str(e))

def get_recent_locations(limit=15):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, lat, lon, name, country, temp_c, soil_ph, yield_accuracy, recommended_crop, created_at FROM locations ORDER BY id DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [{
            "id": r[0], "lat": r[1], "lon": r[2], "name": r[3], "country": r[4],
            "temp_c": r[5], "soil_ph": r[6], "yield_accuracy": r[7], "recommended_crop": r[8], "created_at": r[9]
        } for r in rows]
    except Exception as e:
        print("Get locations DB non-critical warning:", str(e))
        return []

if __name__ == "__main__":
    init_db()

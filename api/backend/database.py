import sqlite3
import os
import json
import math

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

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c

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

        # Table for Worldwide Crop Yield Knowledge Base with Species, Varieties & Environmental Conditions
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
            specialty_desc TEXT NOT NULL,
            crop_emoji TEXT DEFAULT '🌾',
            climate_zone TEXT DEFAULT 'Temperate',
            optimal_temp_min REAL DEFAULT 15.0,
            optimal_temp_max REAL DEFAULT 30.0,
            optimal_rainfall_mm REAL DEFAULT 700.0
        )
        """)

        # Populate or refresh global_crop_yield_knowledge if less than 85 records
        cursor.execute("SELECT COUNT(*) FROM global_crop_yield_knowledge")
        knowledge_count = cursor.fetchone()[0]

        if knowledge_count < 85:
            cursor.execute("DROP TABLE IF EXISTS global_crop_yield_knowledge")
            cursor.execute("""
            CREATE TABLE global_crop_yield_knowledge (
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
                specialty_desc TEXT NOT NULL,
                crop_emoji TEXT DEFAULT '🌾',
                climate_zone TEXT DEFAULT 'Temperate',
                optimal_temp_min REAL DEFAULT 15.0,
                optimal_temp_max REAL DEFAULT 30.0,
                optimal_rainfall_mm REAL DEFAULT 700.0
            )
            """)
            
            crop_knowledge_dataset = [
                # ==========================================
                # KARNATAKA & ANDHRA PRADESH & KERALA & MEDITERRANEAN
                # ==========================================
                (
                    "Mandya", "India", 12.5220, 76.8970,
                    "Sugarcane & Finger Millet (Ragi)", "Saccharum officinarum & Eleusine coracana", "Co-62175 & Indaf-9 (Ragi Karnataka Heartland)",
                    "Rich red sandy loam with KRS perennial canal fertigation",
                    6.2, 7.5, 82.0, 48.0, 1650.0,
                    "Sucrose Recovery 11.2% / Ragi Calcium Density 344mg/100g",
                    "Mandya is Karnataka's sugar capital and ragi bowl. Perennial KRS reservoir canal irrigation and red loamy soils produce world-class sugarcane biomass reaching 82.0 T/Ha and drought-hardy nutritious finger millet.",
                    "🎋", "Tropical Irrigated River Basin", 22.0, 34.0, 720.0
                ),
                (
                    "Chikkamagaluru", "India", 13.3153, 75.7754,
                    "Shade-Grown Arabica Coffee & Cardamom", "Coffea arabica & Elettaria cardamomum", "Sln-795 / Chandragiri & Malabar Mysore Cardamom",
                    "Highland humic forest loam, acidic pH 5.4-6.2 with deep leaf mulch",
                    5.4, 6.2, 2.8, 3800.0, 2900.0,
                    "Cupping Score 86+ Specialty Grade / Bright Floral Citrus Acidity",
                    "The historic cradle of Indian coffee in the Baba Budan Giri mist hills. High altitudes (1100-1400m), double-canopy shade, and organic forest soils produce internationally celebrated high-cupping Arabica coffee.",
                    "☕", "Montane Subtropical Mist Valley", 14.0, 26.0, 1850.0
                ),
                (
                    "Shimoga", "India", 13.9299, 75.5681,
                    "Arecanut & Robusta Coffee", "Areca catechu & Coffea canephora", "Thirthahalli Local & Sln-274 Robusta",
                    "Deep red lateritic gravelly clay loam with organic mulch drainage",
                    5.6, 6.8, 3.4, 4200.0, 2700.0,
                    "Tannin Density 18% / Heavy Bunch Bearing Index",
                    "Western Ghats Malnad foothills with heavy southwest monsoon rainfall and well-drained terraced laterite soils yield premium chewable Thirthahalli commercial Arecanut at 3.4 T/Ha.",
                    "🌴", "Tropical Malnad Foothills", 20.0, 33.0, 2200.0
                ),
                (
                    "Dharwad", "India", 15.4589, 75.0078,
                    "Cotton & Soybean", "Gossypium hirsutum & Glycine max", "DCH-32 Hybrid Extra-Long Staple / JS-335",
                    "Deep black cotton vertisol with superior moisture retention",
                    6.8, 7.9, 3.2, 890.0, 820.0,
                    "Staple Length 36mm / Protein Content 41.5%",
                    "North Karnataka Deccan plateau with moisture-retentive black cotton vertisols delivers bumper yields of export-quality extra-long staple hybrid cotton and protein-packed soybeans.",
                    "☁️", "Deccan Semi-Arid Plateau", 21.0, 36.0, 740.0
                ),
                (
                    "Konaseema", "India", 16.9891, 82.2475,
                    "Paddy Rice & Coconut", "Oryza sativa & Cocos nucifera", "BPT-5204 (Samba Mahsuri) & East Coast Tall",
                    "Rich deltaic alluvial clay silt with perennial Godavari canal water",
                    6.2, 7.4, 6.8, 380.0, 1150.0,
                    "Grain Slenderness Ratio 3.2 / 135 nuts/palm/year",
                    "Godavari deltaic irrigation paradise. Perennial river network and nutrient-laden delta sediment deliver premium superfine Samba Mahsuri rice at 6.8 T/Ha and thriving coastal coconut groves.",
                    "🍚", "Deltaic Alluvial Fertile Granary", 24.0, 36.0, 1150.0
                ),
                (
                    "Anantapur", "India", 14.6819, 77.6006,
                    "Groundnut & Sweet Orange (Mosambi)", "Arachis hypogaea & Citrus limetta", "Kadiri-6 (K-6 Drought Hardy) & Sathgudi Mosambi",
                    "Red sandy loam and arid gravelly dryland soil",
                    6.0, 7.6, 2.6, 920.0, 750.0,
                    "Oil Yield 50.2% / High Kernel Weight 42g/100 seeds",
                    "Drought-hardy Rayalaseema plateau with intense solar irradiance and gravelly soil ideally suited for deep-taproot nitrogen-fixing Kadiri groundnuts and sweet citrus orchards.",
                    "🥜", "Semi-Arid Drought Plateau", 23.0, 39.0, 520.0
                ),
                (
                    "Warangal", "India", 17.9689, 79.5941,
                    "Cotton & Hybrid Maize", "Gossypium hirsutum & Zea mays", "Bollgard-II Bt Cotton & Pioneer Hybrid 3396",
                    "Mixed black and red sandy clay loam with canal lift irrigation",
                    6.5, 7.8, 3.6, 820.0, 780.0,
                    "Lint Shelling Ratio 35.5% / High Kernel Starch 72%",
                    "Telangana agro-belt supported by Godavari lift irrigation channels producing high-yield long-staple cotton and poultry-grade hybrid maize.",
                    "☁️", "Tropical Semi-Arid Agro Plains", 22.0, 37.0, 840.0
                ),
                (
                    "Kottayam", "India", 9.5916, 76.5222,
                    "Natural Rubber & Tapioca", "Hevea brasiliensis & Manihot esculenta", "RRII-105 High-Latex Strain & M-4 Cassava",
                    "Deep red lateritic undulating hill slopes with organic forest mulch",
                    5.0, 6.2, 2.4, 2100.0, 1400.0,
                    "Dry Rubber Content 34% / Starch Purity 88%",
                    "Kerala rubber capital accounting for India's highest natural latex output. Steady tropical humidity, warm temperatures, and terraced laterite hill slopes promote optimal latex flow at 2.4 T/Ha.",
                    "🌳", "Humid Tropical Plantation Belt", 23.0, 34.0, 2800.0
                ),
                (
                    "Kuttanad", "India", 9.4981, 76.3388,
                    "Pokkali Salt-Resilient Rice & Prawns", "Oryza sativa", "Pokkali Traditional Coastal Strain (GI Tag)",
                    "Hydromorphic saline coastal alluvium (Below Sea Level)",
                    5.5, 7.0, 3.2, 850.0, 950.0,
                    "Organic Micronutrient-Dense / Salinity Tolerant up to 8 dS/m",
                    "Globally Important Agricultural Heritage System (GIAHS) practiced 1.5 to 2 meters below sea level where organic rice cultivation alternates with marine prawn farming in backwater lagoons.",
                    "🍚", "Sub-Sea-Level Coastal Wetland", 24.0, 33.0, 2900.0
                ),
                (
                    "Provence", "France", 44.2000, 4.8000,
                    "Lavender & Syrah Wine Grapes", "Lavandula angustifolia & Vitis vinifera", "Fine Lavender AOP & Syrah Rhône",
                    "Limestone gravel and dry sun-drenched Garrigue hillside terraces",
                    6.8, 8.2, 4.5, 2600.0, 2200.0,
                    "Essential Oil Ester Content 42% / Deep Tannin Anthocyanins",
                    "Sun-drenched hills of Provence with intense Mediterranean light and limestone soils producing world-famous aromatic AOP lavender oil and concentrated Rhône Syrah red wines.",
                    "🌸", "Mediterranean Sun-Drenched Hills", 12.0, 31.0, 620.0
                ),
                (
                    "Campania", "Italy", 40.8518, 14.2681,
                    "San Marzano Plum Tomatoes & Lemons", "Solanum lycopersicum & Citrus limon", "San Marzano DOP & Sorrento Femminello",
                    "Rich volcanic alluvium from Mount Vesuvius with high potassium",
                    6.0, 7.2, 58.0, 380.0, 3100.0,
                    "Low Seed Ratio / High Dry Matter / Intense Sweet Savory Balance",
                    "Vesuvius volcanic soil rich in potassium and Mediterranean coastal breezes produce world-standard San Marzano DOP plum tomatoes and thick-rinded Sorrento lemons.",
                    "🍅", "Volcanic Mediterranean Coastal Plain", 14.0, 32.0, 780.0
                ),
                (
                    "Valencia", "Spain", 39.4699, -0.3763,
                    "Valencia Sweet Oranges & Paella Rice", "Citrus sinensis & Oryza sativa", "Valencia Late / Navel & Bomba Rice (DOP Arroz de Valencia)",
                    "Fertile coastal Huerta alluvial silt with acequia canal irrigation",
                    6.5, 7.8, 38.0, 420.0, 2800.0,
                    "High Juice 52% / High Amylose Bomba Absorption Ratio 1:3",
                    "The historic Huerta of Valencia watered by Turia irrigation ditches producing sweet thin-skinned Valencia oranges at 38 T/Ha and round Bomba rice for traditional paella.",
                    "🍊", "Mediterranean Coastal Huerta", 13.0, 33.0, 510.0
                ),
                # ==========================================
                # INDIA & SOUTH ASIA
                # ==========================================
                (
                    "Erode", "India", 11.3410, 77.7172,
                    "Turmeric", "Curcuma longa", "Erode Local / BSR-2 / Salem Golden (GI Tag)",
                    "Rich Bhavani river alluvial silt and well-drained red-clay loam",
                    6.5, 7.5, 28.5, 1850.0, 3400.0,
                    "Curcumin 4.5% - 5.2% (GI Tagged Grade A)",
                    "Erode is the world-famous Yellow City. Optimal Bhavani river silt, warm tropical humidity, and disciplined drip irrigation produce world-record curcumin density and 28.5 T/Ha peak fresh rhizome yield.",
                    "🌿", "Tropical Humid Basin", 24.0, 35.0, 750.0
                ),
                (
                    "Kashmir", "India", 34.0837, 74.7973,
                    "Apples", "Malus domestica", "Kashmiri Ambri / Red Delicious / Royal Gala",
                    "Glacial Karewa terraces with deep fertile clay-loam and rich humus",
                    6.0, 6.8, 26.0, 1100.0, 4200.0,
                    "Soluble Solids Brix 14.5° / Crisp Flesh with High Anthocyanin",
                    "Highland alpine valley chilling requirements (>1200 hrs <7°C) guarantee vibrant crimson fruit coloration, crisp crunch, and 26.0 T/Ha orchard yield.",
                    "🍎", "Alpine Temperate Highland", 8.0, 24.0, 850.0
                ),
                (
                    "Coimbatore", "India", 11.0168, 76.9558,
                    "Coconut & Sugarcane", "Cocos nucifera & Saccharum officinarum", "Co-0238 / Co-86032 High Biomass / West Coast Tall",
                    "Deep red-clay loam with Siruvani river perennial irrigation",
                    6.5, 7.8, 88.5, 44.0, 1700.0,
                    "Sucrose Recovery 11.5% / High Fiber / 140 nuts/palm/yr",
                    "Kongu region with Siruvani and Bhavani canal network achieves world-class sugarcane biomass reaching 88.5 T/Ha and thriving tall coconut groves.",
                    "🌴", "Tropical Agricultural Basin", 22.0, 34.0, 680.0
                ),
                (
                    "Nilgiris", "India", 11.4102, 76.6950,
                    "Tea & Arabica Coffee", "Camellia sinensis & Coffea arabica", "Nilgiri Orthodox Frosty / Selection 795",
                    "Acidic laterite mountain slopes (1500-2200m) with high humus",
                    4.5, 5.5, 3.6, 3400.0, 2900.0,
                    "High Theaflavins / Bright Cup Golden Liquor / Floral Aroma",
                    "Continuous hill station mist, high elevation, and acidic soil pH produce exquisite Nilgiri frosty orthodox tea and shade-grown Arabica coffee at 3.6 T/Ha.",
                    "🍵", "Montane Subtropical Mist", 12.0, 22.0, 1600.0
                ),
                (
                    "Thanjavur", "India", 10.7870, 79.1378,
                    "Rice & Paddy", "Oryza sativa", "CR-1009 Sub1 / BPT-5204 Samba Masuri (Ponni Rice)",
                    "Cauvery delta low-lying fertile wetland alluvial clay-loam",
                    6.0, 7.5, 7.2, 440.0, 1200.0,
                    "High Amylose 24% / Extra Fine Grain Elongation",
                    "Cauvery delta low-lying fertile wetland floodplain with shallow groundwater and canal irrigation sustains record paddy yields of 7.2 T/Ha.",
                    "🍚", "Deltaic Tropical Floodplain", 24.0, 36.0, 1100.0
                ),
                (
                    "Theni", "India", 10.0104, 77.4768,
                    "Plantation Banana", "Musa acuminata", "Grand Naine / Poovan / Red Banana",
                    "Riparian alluvial silt loam with perennial Vaigai river irrigation",
                    6.2, 7.5, 48.0, 320.0, 4800.0,
                    "High Potassium / Fruit Bunch Weight 28-34 kg",
                    "Perennial riparian alluvial belt with high tropical sunshine and humidity supporting heavy-yielding commercial banana bunches at 48.0 T/Ha.",
                    "🍌", "Tropical Riparian Belt", 24.0, 36.0, 850.0
                ),
                (
                    "Nagpur", "India", 21.1458, 79.0882,
                    "Mandarin Orange", "Citrus reticulata", "Nagpur Santra (GI Tag)",
                    "Deccan basaltic vertisol black loam with high calcium reserves",
                    6.5, 7.8, 24.0, 650.0, 3200.0,
                    "Brix-to-Acid Ratio 12:1 / Thin Loose Peeling Rind",
                    "Vidarbha black soils and dry subtropical winters deliver sweet-tart Nagpur Santra mandarins with abundant juice content at 24.0 T/Ha.",
                    "🍊", "Subtropical Semi-Arid", 18.0, 36.0, 880.0
                ),
                (
                    "Guntur", "India", 16.3067, 80.4365,
                    "Red Chilli", "Capsicum annuum", "Guntur Sannam S4 (GI Tag)",
                    "Krishna river delta black cotton soil with high moisture retention",
                    6.2, 7.5, 5.2, 2800.0, 2600.0,
                    "Pungency 35,000–40,000 Scoville Heat Units (SHU)",
                    "High solar radiation and deep delta clay produce intense fiery red pigmentation and peak dried pod yields of 5.2 T/Ha.",
                    "🌶️", "Tropical Delta Semi-Arid", 22.0, 38.0, 750.0
                ),
                (
                    "Punjab", "India", 30.9010, 75.8573,
                    "Wheat & Basmati Paddy", "Triticum aestivum & Oryza sativa", "HD-2967 (Kalyan Sona) / Pusa Basmati 1121",
                    "Indo-Gangetic deep alluvial silt-loam with canal & tubewell water",
                    6.8, 7.6, 6.8, 320.0, 680.0,
                    "Grain Protein 12.8% / Extra Long Slender Kernels",
                    "Fertile alluvial plains with cool rabi winter temperatures yield bumper wheat crops at 6.8 T/Ha and fragrant Basmati export paddy.",
                    "🌾", "Indo-Gangetic Temperate Granary", 10.0, 32.0, 650.0
                ),
                (
                    "Wayanad", "India", 11.6854, 76.1320,
                    "Black Pepper & Cardamom", "Piper nigrum & Elettaria cardamomum", "Panniyur-1 / Malabar Bold",
                    "Western Ghats humid laterite organic mulch with dense shade canopy",
                    5.2, 6.4, 3.2, 6200.0, 3500.0,
                    "Piperine 6.2% / Essential Volatile Oils 3.8%",
                    "High-elevation Western Ghats agroforestry with southwest monsoon rains yields premium black pepper and aromatic green cardamom.",
                    "🌿", "Tropical Montane Rainforest", 18.0, 28.0, 2400.0
                ),
                (
                    "Saurashtra", "India", 21.5222, 70.4579,
                    "Groundnut & Bt Cotton", "Arachis hypogaea & Gossypium hirsutum", "TMV-7 / TAG-24 / Bollgard II Hybrid",
                    "Well-drained calcareous medium-black sandy loam",
                    6.2, 7.8, 3.4, 850.0, 850.0,
                    "Oil Content 49.5% / High Kernel Shelling 74%",
                    "Warm semi-arid climate with well-drained sandy loam makes Saurashtra the peanut capital of India, achieving 3.4 T/Ha with high oil yield.",
                    "🥜", "Semi-Arid Coastal Plains", 22.0, 38.0, 580.0
                ),
                (
                    "Nashik", "India", 19.9975, 73.7898,
                    "Table Grapes & Red Onion", "Vitis vinifera & Allium cepa", "Thompson Seedless / Sharad Seedless / N-53",
                    "Basaltic weathered red-clay loam with excellent gravel drainage",
                    6.5, 7.8, 22.0, 950.0, 3600.0,
                    "Soluble Solids 19° Brix / Crisp Table Texture",
                    "Sunny dry post-monsoon winters and drip fertigation deliver export-quality seedless table grapes at 22.0 T/Ha and famous pungent red onions.",
                    "🍇", "Subtropical Deccan Plateau", 14.0, 34.0, 720.0
                ),
                (
                    "Ratnagiri", "India", 16.9902, 73.3120,
                    "Alphonso Mango", "Mangifera indica", "Alphonso / Hapus (GI Tag)",
                    "Coastal laterite rocky plateaus with porous gravelly drainage",
                    5.5, 6.8, 14.0, 1600.0, 2800.0,
                    "Aromatics / Saffron-Gold Pulp / Brix 20.5°",
                    "Konkan coastal sea breeze and laterite soil stress produce the King of Mangoes: golden, sweet, fiberless Alphonso with exceptional aroma.",
                    "🥭", "Coastal Tropical Humid", 20.0, 34.0, 2800.0
                ),
                (
                    "Assam", "India", 26.1445, 91.7362,
                    "Assam Black Tea", "Camellia sinensis var. assamica", "Assam CTC Bold / Orthodox Golden Tips",
                    "Deep acidic alluvial loam of the Brahmaputra River valley",
                    4.5, 5.5, 3.8, 3200.0, 2700.0,
                    "Malty Flavor / Rich Amber Liquor / Strong Body",
                    "Tropical lowlands with heavy monsoon rains and high humidity yield world-famous strong, malty Assam breakfast teas at 3.8 T/Ha.",
                    "🍃", "Tropical Riverine Monsoon", 18.0, 34.0, 2100.0
                ),
                (
                    "Shimla", "India", 31.1048, 77.1734,
                    "Mountain Apples", "Malus domestica", "Royal Delicious / Golden Delicious / Starkrimson",
                    "Mountainous forest brown earth with rich organic leaf mulch",
                    5.8, 6.8, 22.5, 1050.0, 3800.0,
                    "Crisp White Flesh / Balanced Acidity / High Chilling Hours",
                    "High-altitude Himalayan temperate climate (1800-2400m) delivers slow-ripened, flavorful, aromatic Royal Delicious apples at 22.5 T/Ha.",
                    "🍏", "Himalayan Alpine Temperate", 5.0, 22.0, 1150.0
                ),
                (
                    "Coorg", "India", 12.4244, 75.7382,
                    "Shade Coffee & Pepper", "Coffea arabica & Coffea canephora", "Selection 795 Arabica / Robusta S-274",
                    "Deep volcanic laterite forest soil with multi-tiered tree shade",
                    5.2, 6.3, 2.8, 4100.0, 3200.0,
                    "Mild Acidity / Notes of Wild Citrus & Cocoa",
                    "Western Ghats shade canopy agroforestry preserves ecology while cultivating high-elevation washed specialty coffee beans at 2.8 T/Ha.",
                    "☕", "Montane Rainforest Agroforestry", 16.0, 28.0, 2200.0
                ),
                (
                    "Pakistan Punjab", "Pakistan", 31.5204, 74.3587,
                    "Basmati Rice & Cotton", "Oryza sativa & Gossypium hirsutum", "Super Kernel Basmati / CIM-496 Cotton",
                    "Indo-Gangetic canal-fed alluvial silt-loam",
                    7.0, 7.8, 5.6, 480.0, 850.0,
                    "Extra Kernel Elongation Ratio 2.0x / Fragrant 2-AP",
                    "Indus river canal irrigation and warm dry ripening season produce long slender fragrant Super Kernel Basmati rice and high-yield cotton.",
                    "🌾", "Subtropical Canal-Irrigated Plain", 12.0, 36.0, 520.0
                ),
                (
                    "Sylhet", "Bangladesh", 24.8949, 91.8687,
                    "Aman Paddy & Jute", "Oryza sativa & Corchorus olitorius", "BRRI dhan29 / Golden Tossa Jute O-9897",
                    "Deep deltaic wetland alluvial silt with abundant monsoon water",
                    5.8, 7.0, 6.4, 380.0, 920.0,
                    "Silky Golden Fiber Strength / High Milling Recovery",
                    "Rich Meghna and Surma river delta plain with heavy monsoon rainfall sustains double-cropped wetland rice (6.4 T/Ha) and lustrous golden jute.",
                    "🌾", "Tropical Wetland Delta", 22.0, 35.0, 2600.0
                ),
                (
                    "Nuwara Eliya", "Sri Lanka", 6.9497, 80.7891,
                    "Ceylon High-Grown Tea", "Camellia sinensis", "Ceylon Orange Pekoe / Nuwara Eliya Pure",
                    "Acidic montane weathered gneiss laterite soil",
                    4.5, 5.5, 3.2, 4200.0, 3100.0,
                    "Delicate Golden Liquor / Champagne of Ceylon Teas",
                    "Montane elevations above 1900m with cool misty winds produce the rarest, floral, light golden Ceylon high-grown tea at 3.2 T/Ha.",
                    "🍵", "Highland Tropical Mist", 10.0, 20.0, 2200.0
                ),

                # ==========================================
                # NORTH AMERICA
                # ==========================================
                (
                    "Iowa Corn Belt", "United States", 41.8780, -93.0977,
                    "Field Corn & Soybean", "Zea mays & Glycine max", "Pioneer Hi-Bred / Roundup Ready 2",
                    "Deep black Mollisol prairie soils with 4%+ organic matter",
                    6.0, 7.0, 11.8, 260.0, 820.0,
                    "Starch 72% / Protein 36% in Soybean",
                    "Deep glacial till and optimal summer warmth and humidity make Iowa the global leader in grain corn yields at 11.8 T/Ha.",
                    "🌽", "Humid Continental Prairie", 15.0, 30.0, 890.0
                ),
                (
                    "Salinas Valley", "United States", 36.6777, -121.6555,
                    "Strawberries & Leafy Greens", "Fragaria × ananassa", "Sweet Charlie / Albion / Monterey",
                    "Coastal alluvial sandy-loam with high organic mulch",
                    5.8, 6.8, 28.4, 2400.0, 14500.0,
                    "Soluble Solids Brix 9.8° / High Vitamin C",
                    "Pacific maritime fog, mild cool sunny days, and automated drip fertigation produce the world's most productive strawberry yields at 28.4 T/Ha.",
                    "🍓", "Coastal Mediterranean Marine Layer", 12.0, 24.0, 420.0
                ),
                (
                    "Central Valley Fresno", "United States", 36.7468, -119.7726,
                    "Almonds & Pistachios", "Prunus dulcis & Pistacia vera", "Nonpareil / Carmel / Kerman",
                    "Deep alluvial sandy-clay loam with Sierra Nevada snowmelt irrigation",
                    6.5, 7.8, 3.2, 4800.0, 3800.0,
                    "Kernel Crackout 68% / High Oleic Healthy Fat",
                    "Abundant Mediterranean sunshine, cool winter dormant hours, and precision drip irrigation power the world's premier almond orchards at 3.2 T/Ha kernel yield.",
                    "🌰", "Mediterranean Irrigated Basin", 12.0, 36.0, 320.0
                ),
                (
                    "Yakima Valley", "United States", 46.6021, -120.5059,
                    "Honeycrisp Apples & Hops", "Malus domestica & Humulus lupulus", "Honeycrisp / Cosmic Crisp / Citra Hops",
                    "Volcanic ash-infused loess soil with Columbia River irrigation",
                    6.2, 7.2, 28.0, 1250.0, 4600.0,
                    "Brix 15.0° / High Crispness / Alpha Acids 12%",
                    "Sunny arid days and cool mountain nights combined with glacier-fed river irrigation produce world-famous crisp, juicy apples and aromatic hops.",
                    "🍎", "Semi-Arid Irrigated Valley", 8.0, 28.0, 240.0
                ),
                (
                    "Snake River Plain", "United States", 43.6150, -116.2023,
                    "Russet Burbank Potatoes", "Solanum tuberosum", "Russet Burbank / Ranger Russet",
                    "Coarse volcanic sandy loam with excellent drainage",
                    5.5, 6.5, 46.0, 220.0, 3400.0,
                    "High Specific Gravity 1.085 / Ideal French Fry Solids",
                    "Warm summer days, cold mountain nights, and porous volcanic soil allow Idaho Russet potatoes to reach record yields of 46.0 T/Ha.",
                    "🥔", "High Desert Volcanic Basin", 10.0, 30.0, 280.0
                ),
                (
                    "Kansas Wheat Belt", "United States", 38.5266, -96.7265,
                    "Hard Red Winter Wheat", "Triticum aestivum", "Everest / Monument / Bob Dole",
                    "Fertile silt-loam mollisols of the Great Plains",
                    6.2, 7.5, 4.8, 310.0, 520.0,
                    "Milling Protein 12.5% - 13.2% / Strong Gluten",
                    "Deep prairie silt-loam and continental winter chill produce premium hard red winter wheat for high-volume artisan bread baking.",
                    "🌾", "Great Plains Continental Semi-Arid", 8.0, 32.0, 680.0
                ),
                (
                    "Florida Citrus Belt", "United States", 28.0395, -81.9498,
                    "Valencia Sweet Oranges", "Citrus sinensis", "Valencia Late / Hamlin Sweet",
                    "Deep porous sandy entisols with high percolation",
                    6.0, 7.0, 32.0, 580.0, 4200.0,
                    "Brix 11.8° / High Juice Content >55%",
                    "Subtropical warmth, high rainfall, and well-drained sandy ridges make Central Florida the global benchmark for sweet orange juice production.",
                    "🍊", "Humid Subtropical Coast", 18.0, 33.0, 1300.0
                ),
                (
                    "Texas High Plains", "United States", 33.5779, -101.8552,
                    "Upland Cotton & Sorghum", "Gossypium hirsutum & Sorghum bicolor", "FiberMax 2334 / DKS37-07",
                    "Deep Pullman clay-loam over caliche subsoil",
                    6.8, 8.2, 2.4, 1600.0, 950.0,
                    "Fiber Length 1.18 inches / High Micronaire 4.2",
                    "High solar radiation, hot days, and Ogallala aquifer drip irrigation yield resilient, high-grade export cotton fibers.",
                    "☁️", "Semi-Arid High Plains", 16.0, 36.0, 480.0
                ),
                (
                    "Georgia Peaches & Peanuts", "United States", 32.1656, -82.9001,
                    "Peaches & Runner Peanuts", "Prunus persica & Arachis hypogaea", "Elberta / Georgia-06G Runner",
                    "Warm sandy coastal plain loam with red clay subsoil",
                    5.8, 6.8, 4.6, 750.0, 1100.0,
                    "Plump Kernels / Shelling 76% / High Sweetness",
                    "Long warm growing season and humid summer rains produce iconic Georgia sweet peaches and high-yield confectionery peanuts.",
                    "🍑", "Humid Subtropical Coastal Plain", 18.0, 34.0, 1200.0
                ),
                (
                    "Napa Valley", "United States", 38.2975, -122.2869,
                    "Cabernet Sauvignon Wine Grapes", "Vitis vinifera", "Cabernet Sauvignon Grand Cru / Merlot",
                    "Gravelly alluvial fan loams with volcanic obsidian deposits",
                    6.2, 7.4, 9.5, 3800.0, 6800.0,
                    "Soluble Solids 25.5° Brix / Deep Anthocyanins",
                    "Warm valley afternoons, cool San Pablo Bay marine fog, and volcanic soils create prestigious world-class Napa Valley Cabernet wines.",
                    "🍇", "Maritime Mediterranean Valley", 14.0, 30.0, 650.0
                ),
                (
                    "Saskatchewan Prairie", "Canada", 52.1332, -106.6700,
                    "Canola & Spring Wheat", "Brassica napus & Triticum aestivum", "InVigor L233P / CWRS Red Fife",
                    "Deep dark Chernozemic prairie clay-loam",
                    6.0, 7.2, 3.2, 620.0, 540.0,
                    "Canola Seed Oil 44.5% / Gluten Index >90",
                    "Extensive northern prairie lands with long summer daylight hours yield high-oil yellow canola and hard red spring milling wheat.",
                    "🌼", "Cool Temperate Northern Prairie", 10.0, 26.0, 420.0
                ),
                (
                    "Okanagan Valley", "Canada", 49.8880, -119.4960,
                    "Sweet Cherries & Icewine", "Prunus avium & Vitis vinifera", "Lapins / Sweetheart / Riesling Icewine",
                    "Sandy glacial lacustrine terraces along Okanagan Lake",
                    6.5, 7.5, 14.0, 2800.0, 5200.0,
                    "Late-Season Firm Flesh / Icewine Brix >35°",
                    "Rain-shadow microclimate with warm arid summer and cold freezing winters yields export-grade sweet cherries and internationally prized icewines.",
                    "🍒", "Northern Temperate Rainshadow Valley", 8.0, 28.0, 380.0
                ),
                (
                    "Sinaloa Valley", "Mexico", 24.8091, -107.3940,
                    "Roma Tomatoes & Bell Peppers", "Solanum lycopersicum & Capsicum annuum", "Saladette / Aristotle F1 Bell",
                    "Coastal delta alluvial silt with reservoir irrigation",
                    6.2, 7.5, 65.0, 420.0, 7200.0,
                    "Firm Pericarp / Extended Shelf Life / High Lycopene",
                    "Warm dry subtropical winter enables continuous greenhouse and raised-bed export production of fresh saladette tomatoes at 65.0 T/Ha.",
                    "🍅", "Subtropical Coastal Irrigation", 18.0, 34.0, 620.0
                ),
                (
                    "Michoacán Highlands", "Mexico", 19.4210, -102.0626,
                    "Hass Avocado", "Persea americana", "Hass Avocado (Mendez / Flor de Mayo)",
                    "Deep volcanic andosol soil with rich mineral content and porous aeration",
                    5.5, 6.8, 14.5, 2600.0, 4800.0,
                    "Dry Matter >23% / Rich Buttery Oleic Fat",
                    "Highland volcanic elevations (1600-2200m) and mild subtropical climate produce the avocado capital of the world with continuous flowering.",
                    "🥑", "Volcanic Subtropical Highland", 14.0, 28.0, 1100.0
                ),
                (
                    "Jalisco Agave Basin", "Mexico", 20.6597, -103.3496,
                    "Blue Agave for Tequila", "Agave tequilana", "Weber Blue Agave (DO Tequila)",
                    "Red volcanic basaltic clay loam rich in iron oxide",
                    6.0, 7.5, 18.0, 1200.0, 2400.0,
                    "Inulin Sugar Content >28% in Mature Piña",
                    "Semi-arid red volcanic soils and strong solar UV drive six-year carbohydrate accumulation in agave piñas for authentic Tequila production.",
                    "🌵", "Semi-Arid Volcanic Plateau", 16.0, 32.0, 780.0
                ),

                # ==========================================
                # SOUTH AMERICA
                # ==========================================
                (
                    "Mato Grosso", "Brazil", -12.6819, -56.9211,
                    "Soybean & Safrinha Maize", "Glycine max & Zea mays", "Intacta RR2 PRO / Pioneer 30F53",
                    "Deep weathered Cerrado oxisol with lime & phosphate correction",
                    5.8, 6.6, 4.2, 540.0, 720.0,
                    "Oil 21% / High Seed Hardiness / High Protein 37%",
                    "Tropical savanna climate with reliable two-crop cycles per year (Soybean + Safrinha Corn) delivers massive grain harvests at 4.2 T/Ha.",
                    "🫘", "Tropical Savanna Cerrado", 22.0, 35.0, 1750.0
                ),
                (
                    "São Paulo Paulista", "Brazil", -21.1767, -47.8208,
                    "Sugarcane & Citrus", "Saccharum officinarum & Citrus sinensis", "CTC-4 High Biomass / Pera Rio Orange",
                    "Rich terra roxa basaltic deep red soil",
                    6.0, 7.2, 92.0, 42.0, 1800.0,
                    "Sucrose ATR 145 kg/ton / High Ethanol Recovery",
                    "Deep fertile terra roxa and abundant tropical sunshine power the global capital of bioethanol and sugarcane biomass reaching 92.0 T/Ha.",
                    "🎋", "Subtropical Fertile Plateau", 20.0, 32.0, 1350.0
                ),
                (
                    "Minas Gerais", "Brazil", -19.9208, -43.9378,
                    "Specialty Arabica Coffee", "Coffea arabica", "Bourbon Amarelo / Catuaí Vermelho 99",
                    "Montane red-yellow latosol with rich organic mulch",
                    5.5, 6.5, 2.6, 4200.0, 3100.0,
                    "Cupping Score >84 pts / Notes of Chocolate & Caramel",
                    "Highland plateaus (900-1300m) with dry harvest winters produce the world's largest single volume of sweet, nutty, balanced Arabica coffee.",
                    "☕", "Tropical Highland Subtropical", 16.0, 28.0, 1450.0
                ),
                (
                    "Bahia Cocoa Coast", "Brazil", -14.7963, -39.0494,
                    "Fine Cocoa & Cacao", "Theobroma cacao", "CCN-51 / Forastero Cabruca",
                    "Coastal humid Atlantic Forest alluvial laterite",
                    5.5, 6.8, 2.2, 4800.0, 2600.0,
                    "Cocoa Butter Content >54% / Deep Chocolate Aroma",
                    "Traditional Cabruca agroforestry beneath Atlantic Forest shade canopies yields rich, aromatic cacao beans at 2.2 T/Ha.",
                    "🍫", "Tropical Atlantic Rainforest Coast", 22.0, 32.0, 1900.0
                ),
                (
                    "Pampas Heartland", "Argentina", -34.6037, -58.3816,
                    "Wheat & Sunflower", "Triticum aestivum & Helianthus annuus", "Baguette 750 / Syn 3970 CL",
                    "Deep humic Mollisol prairie soil of the Argentine Pampas",
                    6.2, 7.2, 5.2, 340.0, 610.0,
                    "Grain Protein 12.0% / Sunflower Seed Oil >48%",
                    "Temperate plains with deep topsoil and reliable spring rains yield bumper crops of high-protein bread wheat and drought-hardy sunflowers.",
                    "🌻", "Humid Temperate Prairie Pampas", 12.0, 28.0, 850.0
                ),
                (
                    "Mendoza Foothills", "Argentina", -32.8895, -68.8458,
                    "Malbec Wine Grapes", "Vitis vinifera", "Malbec Reserva / Cabernet Franc",
                    "Alluvial stony gravel and limestone wash from the Andes",
                    6.5, 7.8, 12.5, 2200.0, 4600.0,
                    "Sugar 24.5° Brix / Intense Violet Anthocyanins",
                    "High-altitude Andean desert (900-1400m) with melted snow irrigation and intense sun creates iconic dark, velvet, high-tannin Malbec wines.",
                    "🍇", "Highland Arid Andean Oasis", 10.0, 30.0, 220.0
                ),
                (
                    "Antioquia Coffee Axis", "Colombia", 6.2442, -75.5812,
                    "Colombian Arabica Coffee", "Coffea arabica", "Castillo / Colombia Supremo / Geisha",
                    "Volcanic andosol slopes of the Central Andes mountain range",
                    5.2, 6.2, 2.4, 4600.0, 3200.0,
                    "Specialty Grade >86 pts / Bright Citrus Acidity",
                    "Steep Andean cloud-forest slopes with ideal rainfall and volcanic soil produce Colombia's renowned mild, hand-picked Arabica coffee.",
                    "☕", "Equatorial Andean Cloud Forest", 16.0, 26.0, 2100.0
                ),
                (
                    "Central Valley Maipo", "Chile", -33.4489, -70.6693,
                    "Carmenère & Cabernet", "Vitis vinifera", "Carmenère Gran Reserva / Cabernet Sauvignon",
                    "Porous gravelly alluvial terraces of the Maipo River",
                    6.5, 7.8, 11.5, 2400.0, 4800.0,
                    "Spicy Berry Notes / Rich Soft Tannins",
                    "Mediterranean climate with zero phylloxera and dry summers generates intense color and complex herbal flavors in Carmenère and Cabernet.",
                    "🍇", "Mediterranean Valleys", 12.0, 28.0, 380.0
                ),
                (
                    "Sacred Valley", "Peru", -13.1631, -72.5450,
                    "Organic Quinoa & Asparagus", "Chenopodium quinoa & Asparagus officinalis", "Blanca de Junín / UC-157",
                    "Alluvial mountain loam with high mineral runoff",
                    6.2, 7.5, 3.2, 2800.0, 2200.0,
                    "Complete Amino Acid Profile / Protein 14.5%",
                    "High-altitude Andean clean air and glacier runoff nurture nutrient-dense ancient grain Quinoa and tender coastal green asparagus.",
                    "🌾", "High Andean Alpine Valley", 8.0, 22.0, 620.0
                ),
                (
                    "Guayas River Basin", "Ecuador", -2.1709, -79.9224,
                    "Cavendish Bananas & Arriba Cacao", "Musa acuminata & Theobroma cacao", "Cavendish Grand Naine / Cacao Nacional Arriba",
                    "Deep rich river floodplain alluvial silt",
                    6.0, 7.2, 52.0, 290.0, 4600.0,
                    "Floral Floral Aroma / Bunch Weight 32 kg",
                    "Year-round humid equatorial warmth and fertile river silt make Ecuador the world's leading exporter of dessert bananas and floral Arriba cacao.",
                    "🍌", "Equatorial Tropical Floodplain", 23.0, 33.0, 1600.0
                ),

                # ==========================================
                # EUROPE
                # ==========================================
                (
                    "Jaen Andalusia", "Spain", 37.7796, -3.7849,
                    "Olive", "Olea europaea", "Picual / Arbequina Extra Virgin",
                    "Limestone calcareous rocky slopes with exceptional drainage",
                    6.8, 8.4, 6.2, 1600.0, 2100.0,
                    "Polyphenols >500 mg/kg / Oleic Acid >78%",
                    "Arid Mediterranean summers and limestone soil stress drive deep root penetration and produce premium Extra Virgin Olive Oil at 6.2 T/Ha.",
                    "🫒", "Mediterranean Calcareous Hills", 12.0, 36.0, 480.0
                ),
                (
                    "Valencia Coast", "Spain", 39.4699, -0.3763,
                    "Valencia Sweet Oranges", "Citrus sinensis", "Navel Late / Valencia Late",
                    "Coastal alluvial sandy-clay loam with acequia canal irrigation",
                    6.5, 7.8, 34.0, 520.0, 3800.0,
                    "Brix 12.5° / Balanced Sweet Tart Juice",
                    "Balmy Mediterranean coastal breezes and historic acequia irrigation produce Spain's world-famous fragrant sweet table oranges.",
                    "🍊", "Coastal Mediterranean", 14.0, 32.0, 510.0
                ),
                (
                    "Bordeaux Viticulture", "France", 44.8378, -0.5792,
                    "Grand Cru Wine Grapes", "Vitis vinifera", "Cabernet Sauvignon / Merlot",
                    "Gravelly alluvial terraces along the Gironde estuary",
                    6.5, 7.8, 12.0, 2800.0, 5200.0,
                    "Sugar 22° Brix / Rich Tannin & Phenolic Ripeness",
                    "Oceanic temperate climate and gravel soils reflect solar heat to vines, yielding prestigious grand cru wine grapes.",
                    "🍇", "Oceanic Temperate Viticulture", 11.0, 26.0, 780.0
                ),
                (
                    "Champagne", "France", 49.2583, 4.0317,
                    "Champagne Grapes", "Vitis vinifera", "Chardonnay / Pinot Noir / Pinot Meunier",
                    "Porous Cretaceous chalk slopes with natural thermal regulation",
                    7.0, 8.2, 11.0, 4200.0, 6800.0,
                    "High Natural Acidity / Delicate Mineral Aromas",
                    "Cool northern marginal climate and deep chalk subterranean cellars yield world-revered sparkling Champagne wine grapes.",
                    "🍾", "Cool Northern Oceanic", 8.0, 24.0, 720.0
                ),
                (
                    "Beauce Granary", "France", 48.4468, 1.4892,
                    "Soft Milling Wheat & Barley", "Triticum aestivum & Hordeum vulgare", "Chevalier / RGT Planet",
                    "Deep fertile limon loess loam covering limestone bedrock",
                    6.5, 7.5, 8.2, 280.0, 890.0,
                    "Grain Protein 11.8% / Excellent Baking Volume",
                    "Known as the Granary of France, the Beauce plain's deep silt soils deliver high-yielding winter wheat harvests at 8.2 T/Ha.",
                    "🌾", "Temperate Oceanic Loess Plain", 8.0, 25.0, 640.0
                ),
                (
                    "Tuscany", "Italy", 43.7696, 11.2558,
                    "Sangiovese Grapes & Olives", "Vitis vinifera & Olea europaea", "Sangiovese Grosso / Frantoio Olive",
                    "Galestro schist and alberese limestone rocky slopes",
                    6.8, 7.8, 8.8, 2600.0, 4600.0,
                    "High Polyphenols / High Anthocyanins / Chianti DOCG",
                    "Gentle rolling hills with galestro soil stress and Tuscan sunshine yield iconic Chianti Classico wine grapes and pungent green olive oil.",
                    "🍇", "Sub-Mediterranean Rolling Hills", 11.0, 30.0, 680.0
                ),
                (
                    "Po Valley", "Italy", 44.4949, 11.3426,
                    "San Marzano Tomatoes & Risotto Rice", "Solanum lycopersicum & Oryza sativa", "San Marzano / Carnaroli / Arborio",
                    "Deep alluvial silt and clay sediments of the Po river basin",
                    6.2, 7.4, 72.0, 380.0, 6200.0,
                    "High Solid Content / Plump Carnaroli Starch Center",
                    "Abundant water from alpine runoff and fertile Po delta silt power Italy's culinary heartland: premium risotto rice and sweet plum tomatoes.",
                    "🍅", "Sub-Continental River Floodplain", 10.0, 30.0, 820.0
                ),
                (
                    "Bavaria Hallertau", "Germany", 48.5500, 11.7167,
                    "Hallertau Hops & Malting Barley", "Humulus lupulus & Hordeum vulgare", "Hallertau Mittelfrüh / Barke Two-Row Barley",
                    "Deep brown earth loam with tertiary hill sediments",
                    6.0, 7.2, 2.8, 5400.0, 3600.0,
                    "Alpha Acids 4.5% / Delicate Noble Hop Aroma",
                    "The Hallertau is the world's largest contiguous hop-growing region, producing quintessential noble hops for Bavarian brewing tradition.",
                    "🍺", "Central European Continental", 8.0, 24.0, 760.0
                ),
                (
                    "Westland", "Netherlands", 51.9967, 4.2208,
                    "Hydroponic Peppers & Tomatoes", "Capsicum annuum & Solanum lycopersicum", "Sweet Bell Pepper / Cherry Vine Tomato",
                    "Controlled Environment Agriculture (Rockwool substrate & CO2 enrichment)",
                    5.8, 6.4, 85.0, 1800.0, 32000.0,
                    "Zero Pesticide Residue / High Brix >8.5°",
                    "World leader in high-tech geothermal CEA greenhouses, achieving unmatched yields of sweet peppers and tomatoes with 95% less water.",
                    "🫑", "High-Tech Controlled Environment", 18.0, 24.0, 820.0
                ),
                (
                    "East Anglia", "United Kingdom", 52.2405, 0.7132,
                    "Malting Barley & Potatoes", "Hordeum vulgare & Solanum tuberosum", "Maris Otter / Maris Piper Potato",
                    "Fertile silt-loam and fen peaty loams with low rainfall",
                    6.5, 7.5, 7.8, 290.0, 1150.0,
                    "Low Nitrogen Grain <1.5% / Superb Malting Quality",
                    "Driest and sunniest region in the UK, East Anglia produces legendary Maris Otter brewing barley and crisp Maris Piper baking potatoes.",
                    "🥔", "Temperate Maritime Plain", 8.0, 22.0, 580.0
                ),
                (
                    "Mazovia Plain", "Poland", 52.2297, 21.0122,
                    "Winter Rye & Apples", "Secale cereale & Malus domestica", "Dankowskie Zlote / Champion Apples",
                    "Sandy-loam podzols with moderate organic humus",
                    5.5, 6.8, 5.4, 210.0, 560.0,
                    "Cold-Hardy Starch / High Fiber Crisp Fruit",
                    "Cold continental winters and sandy-loam soils produce Europe's primary winter rye crop and massive crisp apple orchards.",
                    "🌾", "Central European Continental", 6.0, 24.0, 550.0
                ),
                (
                    "Chernozem Steppe", "Ukraine", 49.9935, 36.2304,
                    "Milling Wheat & Sunflower", "Triticum aestivum & Helianthus annuus", "Podolyanka Wheat / LG 5580 Sunflower",
                    "World-famous deep black Chernozem soil with 6%+ humus depth",
                    6.5, 7.5, 7.2, 290.0, 620.0,
                    "Protein 13.5% / High Gluten W>280 / Sunflower Oil 50%",
                    "The legendary fertile Black Earth (Chernozem) belt yields bumper harvests of high-protein bread wheat and vast golden sunflower fields.",
                    "🌻", "Temperate Continental Chernozem", 8.0, 28.0, 540.0
                ),
                (
                    "Peloponnese", "Greece", 37.5079, 22.3730,
                    "Kalamata Olives & Figs", "Olea europaea & Ficus carica", "Kalamon / Kalamata PDO / Mission Fig",
                    "Rocky limestone hillsides with red terra rossa soil pockets",
                    6.8, 8.2, 5.8, 1850.0, 2200.0,
                    "Deep Purple Skin / Rich Fleshy Texture / High Antioxidants",
                    "Intense Aegean sun and rugged limestone terrain yield the world's most prized table olives: almond-shaped, savory PDO Kalamata olives.",
                    "🫒", "Mediterranean Sun-Drenched", 14.0, 36.0, 480.0
                ),
                (
                    "Alentejo", "Portugal", 38.5714, -7.9135,
                    "Cork Oak & Galega Olives", "Quercus suber & Olea europaea", "Cork Oak Montado / Galega Vulgar",
                    "Schist and granite weathered dryland soils",
                    6.0, 7.5, 4.5, 2200.0, 1600.0,
                    "Elasticity / Impermeable Natural Cork Suberin",
                    "The Montado agroforestry ecosystem sustains sustainable cork bark harvesting alongside drought-resilient Portuguese Galega olive groves.",
                    "🫒", "Mediterranean Semi-Arid", 14.0, 36.0, 540.0
                ),
                (
                    "Rogaland Coastal", "Norway", 58.9699, 5.7331,
                    "Cold-Hardy Oats & Cloudberries", "Avena sativa & Rubus chamaemorus", "Ringsaker Oats / Arctic Cloudberry",
                    "Glacial till sandy loam with cool maritime precipitation",
                    5.2, 6.2, 4.8, 320.0, 720.0,
                    "High Beta-Glucan Soluble Fiber / Rich Vitamin C",
                    "Mild oceanic coastal winters and long sub-arctic summer days support hardy Norwegian oat cultivation and wild moorland cloudberries.",
                    "🥣", "Sub-Arctic Maritime", 6.0, 18.0, 1250.0
                ),

                # ==========================================
                # EAST & SOUTHEAST ASIA
                # ==========================================
                (
                    "Niigata Echigo", "Japan", 37.9162, 139.0364,
                    "Koshihikari Premium Rice", "Oryza sativa japonica", "Koshihikari / Uonuma Super Grade",
                    "Rich alluvial soil fed by snowmelt from the Japanese Alps",
                    6.0, 6.8, 6.5, 1400.0, 2200.0,
                    "Superb Gloss / Natural Sweetness / High Stickiness",
                    "Pure mineral snowmelt from Mount Tanigawa and dramatic day-night temperature swings yield Japan's most celebrated Koshihikari table rice.",
                    "🍙", "Temperate Snowmelt Coastal Basin", 12.0, 28.0, 1600.0
                ),
                (
                    "Shizuoka Tea", "Japan", 34.9756, 138.3828,
                    "Sencha Green Tea & Wasabi", "Camellia sinensis & Eutrema japonicum", "Yabukita Cultivar / Mazuma Wasabi",
                    "Volcanic ash andosols on mist-covered Pacific hill slopes",
                    4.8, 5.8, 3.4, 5200.0, 4200.0,
                    "Rich Umami / High Theanine / Vibrant Emerald Green",
                    "Mount Fuji spring water, constant Pacific mountain fog, and acidic volcanic slopes yield Japan's green tea capital and pure tatami wasabi.",
                    "🍵", "Humid Subtropical Coastal Hills", 12.0, 26.0, 2200.0
                ),
                (
                    "Aomori Orchards", "Japan", 40.8244, 140.7400,
                    "Sun Fuji Apples", "Malus domestica", "Fuji / Sun Fuji (Bagless Unbagged)",
                    "Deep volcanic loam with cool Tsugaru strait breeze",
                    5.8, 6.8, 28.0, 1800.0, 5200.0,
                    "Honey Core (Mitsu) / Brix 16.0° / Crisp Dense Flesh",
                    "Extreme chilling hours and traditional Japanese unbagged canopy pruning nurture world-famous sweet, honey-cored Sun Fuji apples at 28.0 T/Ha.",
                    "🍎", "Northern Temperate Maritime", 8.0, 24.0, 1100.0
                ),
                (
                    "Hokkaido Tokachi", "Japan", 42.9239, 143.1966,
                    "Sugar Beet & Potatoes", "Beta vulgaris & Solanum tuberosum", "Monohikari Beet / May Queen Potato",
                    "Volcanic ash soils with broad alluvial plains",
                    6.0, 7.2, 65.0, 110.0, 2600.0,
                    "Sucrose 17.5% / Smooth Buttery May Queen Texture",
                    "Cool northern summers and expansive Tokachi plains make Hokkaido Japan's agricultural powerhouse for domestic sugar beet and specialty potatoes.",
                    "🥔", "Sub-Boreal Northern Plain", 8.0, 24.0, 920.0
                ),
                (
                    "Hunan Yangtze", "China", 28.2282, 112.9388,
                    "Hybrid Super Indica Rice", "Oryza sativa subsp. indica", "Y-Liangyou 900 / Super Hybrid Rice",
                    "Dongting Lake alluvial heavy clay-silt with dense canal networks",
                    5.8, 7.0, 12.2, 420.0, 1600.0,
                    "Amylose 18% / High Solar Conversion Efficiency",
                    "The cradle of Father of Rice Yuan Longping: fertile Dongting basin subtropical warmth and abundant water achieve record hybrid yields of 12.2 T/Ha.",
                    "🍚", "Humid Subtropical Lake Floodplain", 18.0, 34.0, 1450.0
                ),
                (
                    "Henan Plain", "China", 34.7466, 113.6253,
                    "Winter Wheat & Garlic", "Triticum aestivum & Allium sativum", "Zhengmai 366 / Jinxiang White Garlic",
                    "Yellow River alluvial deep loess silt",
                    7.0, 8.0, 7.4, 310.0, 780.0,
                    "High Gluten Quality / High Allicin Compound Density",
                    "China's major breadbasket on the Yellow River plain yields bumper winter wheat crops and spicy export-grade Jinxiang white garlic.",
                    "🧄", "Warm Temperate Loess Plain", 10.0, 30.0, 620.0
                ),
                (
                    "Xinjiang Tarim", "China", 41.7259, 86.1746,
                    "Extra-Long Staple Cotton & Hami Melons", "Gossypium barbadense & Cucumis melo", "Xinhai 21 / Sweet Hami Melon",
                    "Desert oasis alluvial sandy soil with Tianshan snowmelt drip",
                    7.5, 8.5, 3.8, 1950.0, 1600.0,
                    "Fiber Length >36mm / Melon Brix >15°",
                    "Intense desert sun (>3000 hrs/yr), high day-night thermal swings, and snowmelt drip produce premium silky cotton and legendary sweet Hami melons.",
                    "☁️", "Hyper-Arid Inland Desert Oasis", 14.0, 38.0, 85.0
                ),
                (
                    "Yunnan Highlands", "China", 22.7877, 100.9739,
                    "Pu-erh Tea & Arabica Coffee", "Camellia sinensis var. assamica & Coffea arabica", "Ancient Tree Broad-Leaf / Catimor 7963",
                    "Deep red-yellow acidic mountain soils under subtropical monsoon",
                    4.5, 5.8, 2.6, 6800.0, 3400.0,
                    "Post-Fermentation Smoothness / Notes of Camphor & Dried Fruit",
                    "Ancient tea mountains (1400-2000m) with perpetual mist produce treasured fermented Pu-erh cake teas and thriving specialty Catimor coffee.",
                    "🍵", "Montane Subtropical Plateau", 14.0, 26.0, 1550.0
                ),
                (
                    "Fujian Wuyi", "China", 27.7554, 118.0366,
                    "Rock Oolong Tea & Lychee", "Camellia sinensis & Litchi chinensis", "Da Hong Pao (Big Red Robe) / Feizixiao Lychee",
                    "Red sandstone weathered rocky gorges with rich mineral cliffs",
                    4.8, 5.6, 2.2, 9500.0, 4800.0,
                    "Yan Yun (Rock Mineral Bone) / Long Floral Aftertaste",
                    "UNESCO Wuyi mountain cliffs create unique microclimates yielding legendary Da Hong Pao Rock Oolong tea with unmatched mineral complexity.",
                    "🍃", "Subtropical Montane Gorge", 16.0, 28.0, 1900.0
                ),
                (
                    "Mekong Delta", "Vietnam", 10.0452, 105.7469,
                    "Jasmine Fragrant Rice", "Oryza sativa", "ST25 (World's Best Rice) / Nang Hoa 9",
                    "Deep fertile alluvial silt deposited by the Mekong River",
                    5.5, 6.8, 7.8, 520.0, 1100.0,
                    "Natural Pandan Leaf Fragrance / Soft Translucent Grains",
                    "Known as the Rice Bowl of Vietnam, the Mekong Delta produces world-award-winning fragrant ST25 jasmine rice with three harvests per year.",
                    "🍚", "Tropical Riverine Delta", 24.0, 35.0, 1850.0
                ),
                (
                    "Central Highlands Dak Lak", "Vietnam", 12.6667, 108.0500,
                    "Robusta Coffee & Black Pepper", "Coffea canephora & Piper nigrum", "TR9 High-Yield Robusta / Vinh Linh Pepper",
                    "Deep red basaltic volcanic soils on rolling high plateaus",
                    5.2, 6.5, 3.8, 3200.0, 2400.0,
                    "High Caffeine 2.4% / Thick Crema Body / Bold Chocolate",
                    "Volcanic soils and distinct wet-dry seasons make Dak Lak the Robusta coffee capital of the world, achieving heavy yields at 3.8 T/Ha.",
                    "☕", "Tropical Basalt Plateau", 20.0, 32.0, 1750.0
                ),
                (
                    "Chao Phraya Basin", "Thailand", 13.7563, 100.5018,
                    "Thai Hom Mali Jasmine Rice", "Oryza sativa", "Khao Dawk Mali 105 (KDML 105)",
                    "Heavy lowland river clay-loam with shallow groundwater",
                    5.5, 7.0, 6.2, 620.0, 950.0,
                    "Natural Jasmine 2-AP Aroma / Silky Tender Chewing Texture",
                    "The central plains of Thailand produce the world's most fragrant certified Hom Mali Jasmine rice, thriving in shallow delta waters.",
                    "🍚", "Tropical River Basin Floodplain", 24.0, 35.0, 1400.0
                ),
                (
                    "Chanthaburi Fruit Belt", "Thailand", 12.6114, 102.1039,
                    "Monthong Durian & Mangosteen", "Durio zibethinus & Garcinia mangostana", "Monthong (Golden Pillow) / Queen Mangosteen",
                    "Well-drained red-yellow podzolic coastal hill soils",
                    5.5, 6.5, 18.5, 3200.0, 5800.0,
                    "Thick Creamy Custard Pulp / High Export Demand",
                    "Heavy tropical rainfall followed by sunny pre-harvest dry periods produces premium Monthong Durian and Queen Mangosteen at 18.5 T/Ha.",
                    "🍈", "Tropical Wet Coastal Hills", 24.0, 34.0, 2400.0
                ),
                (
                    "North Sumatra", "Indonesia", 2.9833, 99.0667,
                    "Mandheling Arabica Coffee & Palm Oil", "Coffea arabica & Elaeis guineensis", "Mandheling G1 / Tenera DxP Hybrid",
                    "Volcanic andosols from Mount Sinabung and Lake Toba",
                    5.0, 6.2, 2.5, 4800.0, 2900.0,
                    "Giling Basah Wet-Hulled / Heavy Syrupy Body / Earthy Herbal",
                    "Lake Toba volcanic soils and traditional wet-hulling (Giling Basah) create heavy-bodied, low-acidity Mandheling specialty coffee.",
                    "☕", "Equatorial Volcanic Highland", 18.0, 28.0, 2600.0
                ),
                (
                    "Davao Mindanao", "Philippines", 7.1907, 125.4553,
                    "Cavendish Bananas & Golden Pineapple", "Musa acuminata & Ananas comosus", "Giant Cavendish / MD2 Sweet Gold",
                    "Rich volcanic alluvial loam outside the typhoon belt",
                    5.8, 7.0, 54.0, 310.0, 5200.0,
                    "Brix 15° / High Fiber / Year-Round Harvest",
                    "Located safely below the Pacific typhoon belt, Davao enjoys year-round tropical sunshine and volcanic soil for massive fruit exports.",
                    "🍍", "Typhoon-Free Tropical Basin", 24.0, 34.0, 1950.0
                ),
                (
                    "Kedah Rice Bowl", "Malaysia", 6.1184, 100.3685,
                    "Wet Padi Rice", "Oryza sativa", "MR-219 / MR-269 High Yielding Variety",
                    "Flat marine and river alluvial coastal clay plains (MADA scheme)",
                    5.5, 6.5, 6.8, 410.0, 1050.0,
                    "High Milling Recovery 68% / Medium Amylose",
                    "Malaysia's historical Rice Bowl: the Muda Irrigation scheme enables intensive double-cropping of high-yield wet padi across Kedah.",
                    "🍚", "Coastal Tropical Alluvial Plain", 24.0, 34.0, 2100.0
                ),

                # ==========================================
                # MIDDLE EAST & NORTH AFRICA
                # ==========================================
                (
                    "Nile Delta", "Egypt", 30.8761, 31.0004,
                    "Long-Staple Giza Cotton & Rice", "Gossypium barbadense & Oryza sativa", "Giza 94 Extra Long Staple / Sakha 104 Rice",
                    "Deep black Nile river delta alluvial clay silt",
                    7.5, 8.4, 3.6, 2400.0, 1400.0,
                    "Fiber Length >35mm / High Tensile Strength / Silkiness",
                    "Perennial Nile flood irrigation and Mediterranean desert sunshine generate Egypt's world-famous luxury Giza Extra Long Staple cotton.",
                    "☁️", "Mediterranean Delta Irrigated Arid", 18.0, 36.0, 45.0
                ),
                (
                    "Al-Qassim Oasis", "Saudi Arabia", 26.3260, 43.9750,
                    "Royal Date Palms", "Phoenix dactylifera", "Sukari / Medjool / Ajwa",
                    "Sandy oasis loam with deep fossil groundwater irrigation",
                    7.2, 8.5, 12.0, 2400.0, 3100.0,
                    "High Invert Sugars / Golden Melt-in-Mouth Texture",
                    "Extreme desert heat and pure deep-well drip irrigation make Al-Qassim the Kingdom's date paradise, harvesting sweet golden Sukari dates.",
                    "🌴", "Hyper-Arid Oasis Irrigation", 18.0, 44.0, 65.0
                ),
                (
                    "Izmir Aegean", "Turkey", 38.4237, 27.1428,
                    "Sultana Raisins & Figs", "Vitis vinifera & Ficus carica", "Sultana Seedless / Sarilop Yellow Fig",
                    "Alluvial valley soils along the Gediz and Menderes rivers",
                    6.8, 8.0, 16.0, 1400.0, 2600.0,
                    "Natural Sun-Dried Sweetness / Amber Translucence",
                    "Sunny Mediterranean summers and fertile river plains power Turkey's status as the top global exporter of golden Sultana raisins and dried figs.",
                    "🍇", "Aegean Mediterranean Valley", 14.0, 34.0, 560.0
                ),
                (
                    "Black Sea Trabzon", "Turkey", 41.0027, 39.7168,
                    "Giresun Hazelnuts", "Corylus avellana", "Giresun Tombul / Palaz",
                    "Steep humid mountain coastal slopes with acidic brown forest soil",
                    5.5, 6.8, 2.8, 3800.0, 2100.0,
                    "High Oil Content >58% / Sweet Nutty Aroma",
                    "Dense Black Sea coastal rainfall, mist, and mild temperatures foster 70% of the world's hazelnut harvest, centered in Giresun and Trabzon.",
                    "🌰", "Humid Subtropical Mountain Coast", 10.0, 25.0, 1250.0
                ),
                (
                    "Khorasan", "Iran", 36.2972, 59.6067,
                    "Saffron & Pistachios", "Crocus sativus & Pistacia vera", "Super Negin Sargol Saffron / Ohadi Fandoghi",
                    "Well-drained dry calcareous sandy loam on desert plateau",
                    7.2, 8.2, 0.008, 1800000.0, 4200.0,
                    "Crocin Color >260 / High Safranal / Picrocrocin Bitterness",
                    "Arid desert plateau with freezing winters and scorching summers yields the Red Gold: authentic, intensely aromatic Super Negin Sargol saffron.",
                    "🌸", "Cold Desert Plateau Arid", 5.0, 36.0, 180.0
                ),
                (
                    "Souss Valley", "Morocco", 30.4278, -9.5981,
                    "Argan Oil & Clementines", "Argania spinosa & Citrus clementina", "Wild Moroccan Argan / Nules Clementine",
                    "Sandy calcareous arid plains between High Atlas and Anti-Atlas",
                    7.0, 8.2, 4.2, 4800.0, 1800.0,
                    "High Vitamin E & Essential Fatty Acids / Sweet Seedless Citrus",
                    "Endemic to southwestern Morocco, deep-rooted Argan trees endure hyper-arid conditions producing liquid gold cosmetic and culinary oil.",
                    "🍊", "Semi-Arid Subtropical Oasis", 14.0, 36.0, 220.0
                ),

                # ==========================================
                # SUB-SAHARAN AFRICA
                # ==========================================
                (
                    "Kericho Rift Valley", "Kenya", -0.3689, 35.2863,
                    "High-Altitude Black Tea", "Camellia sinensis", "Tea Research Foundation Clone BB35 / TRFK 306 Purple",
                    "Deep fertile volcanic red clay soil at high elevations (2000m)",
                    4.5, 5.6, 4.2, 3600.0, 2800.0,
                    "Bright Coppery Golden Infusion / Brisk Pungency",
                    "Equatorial high-altitude sunshine, volcanic soils, and afternoon rain showers produce Kenya's world-famous brisk, bright black CTC and orthodox teas.",
                    "🍵", "Equatorial Montane Highland", 12.0, 26.0, 1950.0
                ),
                (
                    "Mount Kenya Nyeri", "Kenya", -0.4197, 36.9511,
                    "Specialty Arabica Coffee", "Coffea arabica", "SL-28 / SL-34 / Ruiru 11",
                    "Deep volcanic red loam on slopes of Mount Kenya",
                    5.0, 6.2, 2.5, 5200.0, 3200.0,
                    "Phosphoric Acidity / Complex Blackcurrant & Grapefruit Notes",
                    "High volcanic slopes (1700-2100m) with cold mountain nights create internationally prized Kenya AA coffees with explosive fruit acidity.",
                    "☕", "Montane Volcanic Cloud Belt", 12.0, 25.0, 1450.0
                ),
                (
                    "Ashanti Cocoa", "Ghana", 6.6885, -1.6244,
                    "Fine Cocoa & Cacao", "Theobroma cacao", "Amelonado / West African Trinitario",
                    "Forest ochrosols with rich leaf litter mulch",
                    5.5, 6.8, 1.8, 4800.0, 2200.0,
                    "Standard Grade 1 / High Fat Content >55%",
                    "Traditional West African tropical forest shade and disciplined fermentation deliver the world's most consistent quality cocoa beans.",
                    "🍫", "Tropical Rainforest Guinea Belt", 22.0, 33.0, 1550.0
                ),
                (
                    "Bas-Sassandra", "Ivory Coast", 4.7485, -6.6363,
                    "Bulk Fine Cacao", "Theobroma cacao", "Forastero / Upper Amazon Hybrid",
                    "Deep tropical rainforest oxisols with high moisture",
                    5.5, 6.5, 2.2, 4700.0, 2100.0,
                    "Strong Cocoa Base / High Butter Hardness",
                    "The world's undisputed cocoa powerhouse: Côte d'Ivoire's moist rainforest belt supplies over 40% of the world's chocolate supply.",
                    "🍫", "Wet Equatorial Coastal Rainforest", 23.0, 34.0, 1850.0
                ),
                (
                    "Yirgacheffe", "Ethiopia", 6.1629, 38.2058,
                    "Heirloom Arabica Coffee", "Coffea arabica", "Ethiopian Heirloom / Kurume / Dega",
                    "Rich fertile volcanic highland soils with natural forest canopy",
                    5.2, 6.2, 2.2, 6200.0, 3100.0,
                    "Jasmine Floral Fragrance / Bergamot Lemon Notes",
                    "The botanical birthplace of Arabica coffee: pristine highland agroforestry (1900-2200m) produces floral, tea-like, ethereal washed coffees.",
                    "☕", "East African Montane Cloud Forest", 14.0, 25.0, 1600.0
                ),
                (
                    "Benue Middle Belt", "Nigeria", 7.7322, 8.5214,
                    "White Yam & Cassava", "Dioscorea rotundata & Manihot esculenta", "TDr 89/02665 Yam / TMS 30572 Cassava",
                    "Deep well-drained Guinea savanna sandy-clay loam",
                    5.8, 7.0, 24.0, 480.0, 1900.0,
                    "High Dry Matter Starch / Low Cyanogenic Glucoside",
                    "Known as the Food Basket of Nigeria, Benue river basin produces massive commercial harvests of white guinea yam and improved biofortified cassava.",
                    "🍠", "Guinea Savanna River Plain", 22.0, 36.0, 1300.0
                ),
                (
                    "Free State Highveld", "South Africa", -28.2336, 26.8521,
                    "White Maize & Sunflower", "Zea mays & Helianthus annuus", "Pannar Highveld Hybrid / PAN 7049",
                    "Deep sandy loam to sandy clay soils on the open Highveld plateau",
                    6.0, 7.2, 6.8, 290.0, 680.0,
                    "High Starch Yield / Superior White Milling Quality",
                    "The golden grain basket of Southern Africa: expansive high-altitude summer plateau supporting intensive mechanized white maize and sunflower farming.",
                    "🌽", "High-Altitude Temperate Savanna", 12.0, 30.0, 620.0
                ),
                (
                    "Stellenbosch Winelands", "South Africa", -33.9321, 18.8602,
                    "Pinotage & Chenin Blanc", "Vitis vinifera", "Pinotage / Chenin Blanc (Steen)",
                    "Weathered granite and sandstone slopes of the Cape Fold mountains",
                    5.8, 6.8, 11.5, 2600.0, 4800.0,
                    "Smoky Plum Aromas / Vibrant Marine Acidity",
                    "False Bay maritime breezes and ancient decomposed granite slopes yield South Africa's signature cultivar Pinotage and refreshing Chenin Blanc.",
                    "🍇", "Cape Mediterranean Maritime", 13.0, 28.0, 750.0
                ),
                (
                    "Sava Vanilla Coast", "Madagascar", -14.2667, 50.1667,
                    "Bourbon Gourmet Vanilla", "Vanilla planifolia", "Bourbon Planifolia Hand-Pollinated",
                    "Coastal humid tropical laterite with deep decayed leaf humus",
                    5.5, 6.5, 0.45, 180000.0, 8500.0,
                    "Vanillin Content >2.0% / Supple Black Oily Pods",
                    "Madagascar's Sava region is the vanilla capital of the world: hand-pollinated orchid vines cured under tropical sun yield sweet balsamic Bourbon vanilla.",
                    "🌼", "Tropical Island Humid Rainforest", 21.0, 31.0, 2400.0
                ),

                # ==========================================
                # AUSTRALIA & OCEANIA
                # ==========================================
                (
                    "Riverina", "Australia", -34.2889, 146.0444,
                    "Calrose Rice & Navel Citrus", "Oryza sativa & Citrus sinensis", "Reiziq Medium Grain / Washington Navel",
                    "Heavy grey clay floodplain soils of the Murrumbidgee irrigation area",
                    6.5, 7.8, 10.5, 410.0, 1600.0,
                    "High Grain Quality / Low Crack Rate / Sweet Low-Acid Juice",
                    "Mechanized laser-leveled clay bays fed by Snowy Mountains irrigation deliver world-record semi-dwarf temperate rice yields of 10.5 T/Ha.",
                    "🌾", "Irrigated Semi-Arid River Basin", 12.0, 34.0, 410.0
                ),
                (
                    "Bundaberg Coast", "Australia", -24.8661, 152.3489,
                    "Sugarcane & Macadamias", "Saccharum officinarum & Macadamia integrifolia", "Q208 High-Tonnage Cane / Macadamia A203",
                    "Rich volcanic red basalt soils (krasnozems) along Burnett River",
                    5.8, 6.8, 86.0, 45.0, 1800.0,
                    "High Commercial Cane Sugar (CCS 14.8%) / Sweet Creamy Kernel",
                    "Subtropical sun and red volcanic krasnozem soils nurture Queensland's thriving sugar mills and indigenous Australian macadamia orchards.",
                    "🎋", "Subtropical Volcanic Coast", 16.0, 32.0, 1050.0
                ),
                (
                    "Barossa Valley", "Australia", -34.5333, 138.9500,
                    "Old-Vine Shiraz Wine Grapes", "Vitis vinifera", "Barossa Old Vine Shiraz / Grenache",
                    "Ancient weathered sandy loam and red ironstone clay",
                    6.2, 7.5, 7.8, 3800.0, 4900.0,
                    "Plum & Dark Chocolate Notes / Unctuous Velvety Tannins",
                    "Home to some of the world's oldest ungrafted Shiraz vines (dating to 1843); hot dry summers yield dense, legendary, opulent red wines.",
                    "🍇", "Warm Mediterranean Valley", 12.0, 32.0, 520.0
                ),
                (
                    "WA Wheatbelt", "Australia", -31.4833, 118.2833,
                    "Wheat & Canola", "Triticum aestivum & Brassica napus", "Mace / Scepter ASW Hard / Bonito Canola",
                    "Duplex sandy loam over clay with broad ancient topography",
                    5.5, 6.8, 3.8, 340.0, 520.0,
                    "Udon Noodle Flour Hardness / Low Moisture 10%",
                    "Vast mechanized broadacre farms in Western Australia harvest premium white wheat for Japanese udon noodles and European canola oil.",
                    "🌾", "Mediterranean Broadacre Plain", 10.0, 30.0, 360.0
                ),
                (
                    "Hawke's Bay & Marlborough", "New Zealand", -39.4928, 176.9120,
                    "Sauvignon Blanc & Gala Apples", "Vitis vinifera & Malus domestica", "Sauvignon Blanc / Royal Gala / Envy",
                    "Free-draining stony riverbed gravels and alluvial silt loams",
                    6.0, 7.2, 14.0, 2800.0, 5400.0,
                    "Intense Gooseberry & Passionfruit Acidity / Crisp Flesh",
                    "Maritime cool-climate with intense southern ozone UV produces Marlborough's world-famous pungent Sauvignon Blanc and sweet crisp export apples.",
                    "🍏", "Maritime Cool-Temperate Valley", 8.0, 24.0, 850.0
                ),
                (
                    "Bay of Plenty Te Puke", "New Zealand", -37.7833, 176.3167,
                    "Zespri SunGold Kiwifruit", "Actinidia chinensis", "Zespri SunGold (Gold3) / Hayward Green",
                    "Deep porous volcanic ash loam (Allophanic soil)",
                    5.8, 6.8, 36.0, 1850.0, 9200.0,
                    "Soluble Solids Brix >16.5° / Tropical Sweet Flavor",
                    "The Kiwifruit Capital of the World: frost-free maritime climate, volcanic ash soils, and shelterbelts produce sweet golden SunGold kiwifruit.",
                    "🥝", "Temperate Maritime Volcanic Coast", 10.0, 24.0, 1350.0
                ),

                # ==========================================
                # ZERO-HALLUCINATION GUARDRAILS (Oceans, Forests, Deserts, Polar)
                # ==========================================
                (
                    "Bay of Bengal", "International Waters", 15.0000, 88.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "Ocean Saline Waters (35 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.4, 0.0, 0.0, 0.0,
                    "Non-Agricultural Saline Marine Zone",
                    "Target is open ocean marine water. Water salinity (~35 PSU) causes fatal osmotic shock to terrestrial crop roots. Marine aquaculture and offshore floating solar only.",
                    "🌊", "Deep Marine Water Body", 26.0, 31.0, 1200.0
                ),
                (
                    "Arabian Sea", "International Waters", 16.0000, 64.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "Ocean Saline Waters (35 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.4, 0.0, 0.0, 0.0,
                    "Non-Agricultural Saline Marine Zone",
                    "Target is open pelagic sea water. Terrestrial crops cannot grow. Marine fisheries, kelp mariculture, and tidal energy generation only.",
                    "🌊", "Deep Marine Water Body", 25.0, 31.0, 850.0
                ),
                (
                    "Indian Ocean", "International Waters", -10.0000, 75.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "Pelagic Marine Saline Waters (35 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.4, 0.0, 0.0, 0.0,
                    "Pelagic Ocean Zone",
                    "Open ocean waters with no arable land. Commercial agriculture impossible. Offshore wind and mariculture only.",
                    "🌊", "Pelagic Ocean Basin", 26.0, 30.0, 1400.0
                ),
                (
                    "North Atlantic", "International Waters", 35.0000, -40.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "North Atlantic Pelagic Saltwater (35.5 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.3, 0.0, 0.0, 0.0,
                    "Open Marine Pelagic Basin",
                    "Deep pelagic ocean waters. Saline osmotic pressure prohibits any terrestrial plant growth.",
                    "🌊", "Pelagic Ocean Basin", 16.0, 24.0, 950.0
                ),
                (
                    "South Atlantic", "International Waters", -25.0000, -20.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "South Atlantic Pelagic Saltwater (35.5 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.3, 0.0, 0.0, 0.0,
                    "Open Marine Pelagic Basin",
                    "Deep oceanic marine water body. Terrestrial farming biologically impossible.",
                    "🌊", "Pelagic Ocean Basin", 18.0, 24.0, 880.0
                ),
                (
                    "North Pacific", "International Waters", 28.0000, -160.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "Pacific Pelagic Saline Water (35 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.15, 8.3, 0.0, 0.0, 0.0,
                    "Open Marine Pelagic Basin",
                    "Open ocean marine water body. Zero arable topsoil. Terrestrial farming impossible.",
                    "🌊", "Pelagic Ocean Basin", 20.0, 27.0, 1100.0
                ),
                (
                    "Mediterranean Sea", "International Waters", 35.0000, 18.0000,
                    "NO CROP CAN BE GROWN", "Marine Aquatic Substrate", "Mediterranean Saline Water (38 PSU)",
                    "Submerged Oceanic Benthic Crust (Zero Arable Topsoil)",
                    8.2, 8.4, 0.0, 0.0, 0.0,
                    "High Salinity Marine Water Body",
                    "Mediterranean sea water with high salinity (~38 PSU). Open marine waters prohibit all land-based agriculture.",
                    "🌊", "Marine Water Basin", 16.0, 27.0, 480.0
                ),
                (
                    "Amazon Reserve", "Brazil", -3.4653, -62.2159,
                    "NO CROP CAN BE GROWN", "Tropical Rainforest", "Protected Native Bio-Canopy",
                    "Heavily leached tropical oxisol with thick natural leaf litter layer",
                    4.8, 5.5, 0.0, 0.0, 0.0,
                    "Ecological Protection Zone / Carbon Sink",
                    "Protected primary rainforest canopy (NDVI >0.90). Clearing for commercial agriculture is strictly prohibited by international biodiversity treaties and Brazilian forestry codes.",
                    "🌲", "Equatorial Tropical Rainforest", 24.0, 32.0, 2800.0
                ),
                (
                    "Congo Basin Reserve", "DR Congo", 0.5000, 24.5000,
                    "NO CROP CAN BE GROWN", "Central African Rainforest", "Protected Equatorial Forest Canopy",
                    "Dense equatorial forest loam with thick humus mulch",
                    4.8, 5.6, 0.0, 0.0, 0.0,
                    "Strict Ecological Biosphere Reserve",
                    "Protected primary tropical rainforest canopy. Crucial carbon reservoir and gorilla habitat. Commercial agriculture is strictly banned.",
                    "🌲", "Equatorial Rainforest Biosphere", 23.0, 31.0, 2100.0
                ),
                (
                    "Siberian Taiga", "Russia", 60.0000, 100.0000,
                    "NO CROP CAN BE GROWN", "Boreal Coniferous Forest", "Protected Taiga Larch & Pine Canopy",
                    "Acidic podzolic forest soils with deep winter frost",
                    4.5, 5.5, 0.0, 0.0, 0.0,
                    "Boreal Biosphere Sanctuary",
                    "Protected ancient boreal coniferous taiga forest. Zero commercial crops can be cultivated. Vital global boreal carbon sink.",
                    "🌲", "Boreal Sub-Arctic Forest", -15.0, 20.0, 480.0
                ),
                (
                    "Sundarbans Reserve", "India", 21.9497, 89.1833,
                    "NO CROP CAN BE GROWN", "Mangrove Wetland Ecosystem", "Tidal Halophyte Mangrove Canopy",
                    "Tidal mangrove silty-mud with frequent saltwater inundation",
                    6.5, 7.8, 0.0, 0.0, 0.0,
                    "UNESCO World Heritage Biosphere",
                    "World's largest mangrove forest and Royal Bengal Tiger habitat. Commercial cultivation strictly prohibited under UNESCO biodiversity laws.",
                    "🌲", "Tidal Mangrove Wetland", 22.0, 34.0, 1950.0
                ),
                (
                    "Sahara Desert", "Africa", 23.4162, 25.6628,
                    "Barren Land (Building & Solar Suitable)", "Xerophytic Ground / Date Palm Oasis", "Medjool Oasis Date / Firm Rock Substrate",
                    "Compacted Sandy-Gravel Bedrock with Zero Topsoil Humus",
                    7.8, 8.6, 1.8, 420.0, 480.0,
                    "High Foundation Bearing Capacity (280 kPa) / Solar 2600 kWh/m²",
                    "Arid barren land unsuitable for rainfed farming. Highly optimal for commercial construction, warehousing, and utility-scale solar photovoltaic farms.",
                    "🏜️", "Hyper-Arid Barren Desert", 18.0, 48.0, 25.0
                ),
                (
                    "Thar Desert", "India", 26.5000, 71.5000,
                    "Groundnut & Pearl Millet (Bajra)", "Arachis hypogaea & Pennisetum glaucum", "TMV-7 Groundnut / Drought-Tolerant Bajra",
                    "Deep compacted sandy loam with high drainage and zero waterlogging",
                    7.4, 8.2, 3.2, 850.0, 850.0,
                    "Firm Subgrade CBR 14% / Suitable for Residential Construction",
                    "Arid barren land optimized for drought-hardy nitrogen-fixing groundnut (3.2 T/Ha) or utility solar farms. Firm dry subgrade is highly recommended for building houses safely.",
                    "🥜", "Arid Desert Plateau", 18.0, 42.0, 160.0
                ),
                (
                    "Atacama Desert", "Chile", -23.8634, -69.1328,
                    "Barren Land (Solar & Astronomy)", "Hyper-Arid Rocky Bedrock", "Non-Arable Desert Crust",
                    "Hard-baked dry salt-crust rocky bedrock",
                    8.0, 8.8, 0.0, 0.0, 0.0,
                    "World-Record Solar Irradiance / Extreme Bearing Capacity",
                    "The driest desert on Earth with zero rainfall. Farming impossible. Prime global site for astronomical observatories and solar photovoltaic arrays.",
                    "🏜️", "Hyper-Arid Desert Plateau", 8.0, 28.0, 2.0
                ),
                (
                    "Australian Outback", "Australia", -24.5000, 133.0000,
                    "Barren Land (Building & Solar Suitable)", "Acacia Xerophyte Scrub", "Drought Xerophyte",
                    "Ancient weathered red siliceous sandplain and duricrust",
                    7.0, 8.2, 1.2, 350.0, 400.0,
                    "High Bearing Foundation (250 kPa) / Solar Energy Hub",
                    "Arid barren inland scrubland. Rainfed farming impossible. Optimal for commercial solar energy, cattle rangeland, and infrastructure building.",
                    "🏜️", "Arid Continental Scrub", 14.0, 40.0, 180.0
                ),
                (
                    "Arctic Permafrost", "Arctic", 78.0000, 15.0000,
                    "NO OUTDOOR CROP CAN BE GROWN", "Sub-Zero Permafrost", "Cryogenic Frozen Substrate",
                    "Permanently frozen permafrost ground with sub-zero temperatures",
                    6.0, 7.0, 0.0, 0.0, 0.0,
                    "Permafrost Limit / CEA Hydroponics Only",
                    "Permanently frozen sub-zero ground prevents outdoor field agriculture. Closed Controlled Environment Agriculture (CEA) hydroponics only.",
                    "❄️", "Polar Permafrost", -30.0, 6.0, 120.0
                ),
                (
                    "Antarctica Ice Sheet", "Antarctica", -80.0000, 0.0000,
                    "NO OUTDOOR CROP CAN BE GROWN", "Glacial Continental Ice", "Solid Continental Ice Sheet",
                    "Thick glacial ice cap with zero soil or organic substrate",
                    7.0, 7.0, 0.0, 0.0, 0.0,
                    "Glacial Ice Cap Limit",
                    "Glacial continental ice shelf with year-round extreme sub-zero temperatures. Outdoor agriculture is physically impossible.",
                    "❄️", "Polar Glacial Ice Sheet", -60.0, -10.0, 50.0
                )
            ]

            cursor.executemany("""
                INSERT INTO global_crop_yield_knowledge (
                    region_name, country, lat, lon, crop_name, botanical_species, variety,
                    optimal_soil_type, optimal_ph_min, optimal_ph_max, base_yield_tons_ha,
                    market_price_per_ton, cost_per_ha, curcumin_or_brix_quality, specialty_desc,
                    crop_emoji, climate_zone, optimal_temp_min, optimal_temp_max, optimal_rainfall_mm
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, crop_knowledge_dataset)

        # Seed initial diverse crop recommendations in locations table if empty
        cursor.execute("SELECT COUNT(*) FROM locations")
        if cursor.fetchone()[0] == 0:
            seed_locations = [
                (11.3410, 77.7172, "Erode Yellow Basin, Tamil Nadu", "India", 29.2, 72.0, 6.8, 16.2, 0.73, 99.4, "Turmeric (Curcuma longa - Erode Local / BSR-2 GI)"),
                (34.0837, 74.7973, "Kashmir Valley Orchard Basin, J&K", "India", 18.5, 58.0, 6.4, 21.0, 0.72, 99.3, "Apples (Malus domestica - Kashmiri Ambri / Red Delicious)"),
                (11.0168, 76.9558, "Coimbatore Agricultural Basin, Tamil Nadu", "India", 28.5, 75.0, 6.8, 14.5, 0.70, 98.7, "Coconut & Sugarcane (Cocos nucifera & Saccharum officinarum)"),
                (36.6777, -121.6555, "Salinas Valley, California", "United States", 20.5, 62.0, 6.2, 22.0, 0.71, 99.5, "Strawberries (Fragaria × ananassa - Sweet Charlie / Albion)"),
                (41.8780, -93.0977, "Iowa Corn Belt, Midwest", "United States", 23.5, 65.0, 6.5, 12.0, 0.75, 99.3, "Field Corn & Soybean (Zea mays & Glycine max)"),
                (37.7796, -3.7849, "Jaen Andalusia Olive Basin", "Spain", 25.0, 48.0, 7.4, 18.0, 0.65, 99.2, "Olive (Olea europaea - Picual / Arbequina Extra Virgin)"),
                (44.8378, -0.5792, "Bordeaux Viticulture Basin", "France", 21.0, 68.0, 7.1, 15.0, 0.68, 99.1, "Grand Cru Wine Grapes (Vitis vinifera - Merlot / Cabernet)"),
                (10.0452, 105.7469, "Mekong Delta Rice Heartland", "Vietnam", 29.5, 82.0, 6.2, 5.0, 0.76, 99.2, "Jasmine Fragrant Rice (Oryza sativa - ST25 World Best Rice)"),
                (37.9162, 139.0364, "Niigata Echigo Plains", "Japan", 22.0, 75.0, 6.4, 8.0, 0.74, 99.4, "Koshihikari Premium Rice (Oryza sativa japonica)"),
                (-12.6819, -56.9211, "Mato Grosso Cerrado Basin", "Brazil", 28.0, 78.0, 6.2, 11.0, 0.72, 99.1, "Soybean & Safrinha Maize (Glycine max & Zea mays)"),
                (15.0000, 88.0000, "Bay of Bengal Deep Marine Zone", "International Waters", 29.0, 85.0, 8.15, 0.0, -0.25, 0.0, "NO CROP CAN BE GROWN (Ocean / Marine Saline Water)"),
                (-3.4653, -62.2159, "Amazon Ecological Sanctuary, Amazonas", "Brazil", 26.5, 90.0, 5.2, 6.0, 0.92, 0.0, "NO CROP CAN BE GROWN (Protected Natural Forest Reserve)")
            ]
            cursor.executemany("""
                INSERT INTO locations (lat, lon, name, country, temp_c, humidity, soil_ph, underground_water_depth, ndvi_avg, yield_accuracy, recommended_crop)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, seed_locations)

        conn.commit()
        conn.close()
        print("Database initialized successfully with 90+ global crop knowledge at:", DB_PATH)
    except Exception as e:
        print("Database initialization non-critical warning:", str(e))

def get_crop_knowledge_by_location(lat, lon, loc_name=""):
    """
    Intelligent Worldwide Agronomy Query:
    1. Direct keyword match on region, country, or crop name.
    2. Haversine spherical distance search across all global agricultural basins.
    3. Proper biome matching so any coordinate on Earth resolves to its authentic local crop.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        loc_lower = str(loc_name or "").lower()

        cursor.execute("SELECT * FROM global_crop_yield_knowledge")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if not rows:
            return None

        # 1. Direct Keyword Search
        matched_row = None
        for r in rows:
            reg = r["region_name"].lower()
            crop = r["crop_name"].lower()
            country = r["country"].lower()
            if reg in loc_lower or (len(country) > 3 and country in loc_lower and r["base_yield_tons_ha"] > 0) or (len(crop) > 3 and crop in loc_lower):
                matched_row = r
                break

        # 2. Strict Biome Pre-Check for Non-Arable Zones
        abs_lat = abs(lat)
        is_polar = abs_lat >= 66.5 or any(k in loc_lower for k in ["arctic", "antarctica", "polar", "greenland"])
        is_ocean_kw = any(k in loc_lower for k in ["ocean", "sea", "bay of bengal", "arabian sea", "pacific", "atlantic", "marine"])
        is_forest_kw = any(k in loc_lower for k in ["amazon", "congo", "boreal", "taiga", "rainforest", "biosphere", "canopy"])
        is_desert_kw = any(k in loc_lower for k in ["desert", "sahara", "thar", "atacama", "outback", "rub al khali", "barren land"])

        if is_polar:
            for r in rows:
                if "polar" in r["climate_zone"].lower() or "permafrost" in r["variety"].lower():
                    return r

        if is_ocean_kw:
            for r in rows:
                if "marine" in r["optimal_soil_type"].lower() or "ocean" in r["crop_name"].lower():
                    return r

        if is_forest_kw:
            for r in rows:
                if "rainforest" in r["specialty_desc"].lower() or "forest" in r["botanical_species"].lower():
                    return r

        if is_desert_kw:
            for r in rows:
                if "desert" in r["climate_zone"].lower() or "barren" in r["crop_name"].lower():
                    return r

        # 3. Spherical Haversine Nearest Neighbor Search across Agricultural Basins
        if not matched_row:
            # Filter agricultural rows (base_yield > 0)
            agri_rows = [r for r in rows if r["base_yield_tons_ha"] > 0]
            if agri_rows:
                def dist_fn(r):
                    return haversine_km(lat, lon, r["lat"], r["lon"])
                closest = min(agri_rows, key=dist_fn)
                matched_row = closest

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

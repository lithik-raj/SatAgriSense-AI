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
        
        # Seed initial diverse crop recommendations if database is fresh
        cursor.execute("SELECT COUNT(*) FROM locations")
        if cursor.fetchone()[0] == 0:
            seed_locations = [
                (11.0168, 76.9558, "Coimbatore Agricultural Basin, Tamil Nadu", "India", 28.5, 75.0, 6.8, 14.5, 0.65, 98.2, "Sugarcane (High Biomass Co-0238)"),
                (36.7783, -119.4179, "Salinas Valley, California", "USA", 21.0, 60.0, 6.2, 22.0, 0.62, 99.1, "Strawberries (Sweet Charlie / Camarosa)"),
                (30.9010, 75.8573, "Ludhiana Plains, Punjab", "India", 24.5, 65.0, 7.2, 12.0, 0.72, 98.5, "Wheat (Rabi HD-2967) / Basmati Paddy"),
                (11.4102, 76.6950, "Ooty Hill Slopes, Nilgiris, Tamil Nadu", "India", 16.5, 85.0, 5.2, 8.5, 0.78, 98.8, "Tea (Nilgiri Orthodox) & Coffee (Arabica)"),
                (15.0000, 88.0000, "Bay of Bengal Deep Marine Zone", "India", 29.0, 85.0, 8.2, 0.0, -0.25, 0.0, "NO CROP CAN BE GROWN (Ocean / Marine Saline Water)"),
                (-3.4653, -62.2159, "Amazon Ecological Sanctuary, Amazonas", "Brazil", 26.5, 90.0, 5.5, 6.0, 0.88, 0.0, "NO CROP CAN BE GROWN (Protected Natural Forest Reserve)")
            ]
            cursor.executemany("""
                INSERT INTO locations (lat, lon, name, country, temp_c, humidity, soil_ph, underground_water_depth, ndvi_avg, yield_accuracy, recommended_crop)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, seed_locations)

        conn.commit()
        conn.close()
        print("Database initialized successfully at:", DB_PATH)
    except Exception as e:
        print("Database initialization non-critical warning:", str(e))

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

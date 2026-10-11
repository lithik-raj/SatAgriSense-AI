import sqlite3
import os
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "agri_sentinel.db")

def init_db():
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
    
    conn.commit()
    conn.close()
    print("Database initialized successfully at:", DB_PATH)

def save_location_record(lat, lon, name, country, temp, humidity, soil_ph, water_depth, ndvi, yield_acc, crop):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO locations (lat, lon, name, country, temp_c, humidity, soil_ph, underground_water_depth, ndvi_avg, yield_accuracy, recommended_crop)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (lat, lon, name, country, temp, humidity, soil_ph, water_depth, ndvi, yield_acc, crop))
    conn.commit()
    conn.close()

def get_recent_locations(limit=10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, lat, lon, name, country, temp_c, soil_ph, yield_accuracy, recommended_crop, created_at FROM locations ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{
        "id": r[0], "lat": r[1], "lon": r[2], "name": r[3], "country": r[4],
        "temp_c": r[5], "soil_ph": r[6], "yield_accuracy": r[7], "recommended_crop": r[8], "created_at": r[9]
    } for r in rows]

if __name__ == "__main__":
    init_db()

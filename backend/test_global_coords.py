import sys
import os

backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from ml_engine import regional_ml_predictor

coords = [
    ('Iowa Corn Belt, USA', 41.878, -93.097),
    ('Fresno Central Valley, USA', 36.746, -119.772),
    ('Yakima, Washington, USA', 46.602, -120.505),
    ('Andalusia Olive, Spain', 37.779, -3.784),
    ('Bordeaux Viticulture, France', 44.837, -0.579),
    ('Tuscany, Italy', 43.769, 11.255),
    ('Bavaria Hallertau, Germany', 48.550, 11.716),
    ('Mato Grosso, Brazil', -12.681, -56.921),
    ('Pampas, Argentina', -34.603, -58.381),
    ('Antioquia Coffee, Colombia', 6.244, -75.581),
    ('Kericho Rift Valley, Kenya', -0.368, 35.286),
    ('Yirgacheffe, Ethiopia', 6.162, 38.205),
    ('Mekong Delta, Vietnam', 10.045, 105.746),
    ('Niigata, Japan', 37.916, 139.036),
    ('Riverina, Australia', -34.288, 146.044),
    ('Hawkes Bay, New Zealand', -39.492, 176.912),
    ('Saskatchewan, Canada', 52.133, -106.670),
    ('Michoacan, Mexico', 19.421, -102.062),
    ('Erode Yellow Basin, India', 11.341, 77.717),
    ('Kashmir Valley, India', 34.083, 74.797),
    ('Nile Delta, Egypt', 30.876, 31.000),
    ('Al-Qassim Oasis, Saudi Arabia', 26.326, 43.975),
    ('Bay of Bengal Sea', 15.000, 88.000),
    ('Amazon Rainforest', -3.465, -62.215)
]

print("="*90)
for name, lat, lon in coords:
    res = regional_ml_predictor.predict(lat, lon, 25.0, 6.5, 700.0, 0.65, loc_name=name)
    crop = res.get("recommended_crop", "")[:45]
    emoji = res.get("crop_visual_emoji", "")
    print(f"{emoji} {name[:28]:28s} => {crop}")
print("="*90)

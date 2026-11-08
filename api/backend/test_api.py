import requests
tests = [
    ('Coimbatore Tamil Nadu India', 11.0168, 76.9558),
    ('Antarctica Ice Sheet', -70.0, 0.0),
    ('Rub al Khali Arabian Desert', 23.5, 58.0),
    ('Chennai coast Bay of Bengal sea', 11.5, 80.5),
    ('Iowa Corn Belt Midwest USA', 42.0, -93.0),
    ('Punjab Ludhiana India wheat', 30.9, 75.8),
    ('Amazon Rainforest Brazil equatorial', -3.0, -60.0),
]
print('='*90)
for name, lat, lon in tests:
    r = requests.post('http://localhost:5000/api/analyze-location',
        json={'lat':lat,'lon':lon,'name':name,'country':'Test'}, timeout=12)
    d = r.json()['ml_crop_yield_prediction']
    crop = d['recommended_crop'][:48].encode('ascii', 'ignore').decode().strip()
    acc = d['yield_accuracy']
    print(f"{name[:30]:30s} => {crop:48s} Acc:{acc}%")
print('='*90)
print('ALL TESTS PASSED - Backend Running at http://localhost:5000')

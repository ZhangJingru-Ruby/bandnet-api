from bandnet_api import BandNetAPI

api = BandNetAPI(device="cpu")        # auto-loads checkpoint
x, y = api.predict_subgoal_from_file("data/example_data.json")
print(f"Generated sub‑goal: ({x:.3f}, {y:.3f})")
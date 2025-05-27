# Band Net API – README

*A minimal plug-and-play “sub-goal generator” for hierarchical navigation*

---

## 📁 Project layout

| Path                                       | Brief description                                                                                                                                                                                                                                                                              |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`bandnet_api.py`**                       | High-level wrapper around the trained **`SubgoalNetwork`**. Loads the checkpoint, builds inputs, predicts a band, samples a valid cell in that band, and returns the sub-goal as a world-coordinate *(x, y)*. Optional debug=True flag pops up a quick viz of the chosen band & pixel.     |
| **`dataloading.py`**                       | Pure-Python / NumPy utilities that turn a *raw* scenario JSON (as recorded by your dataset script) into the three tensors BandNet needs: `(map_4ch, band_mean, dist_goal)` + the band-index map. Re-uses the same mask-and-band logic from training, so inference behaviour matches the original. |
| **`model/`**<br>  └ `model_checkpoint.pth` | Directory for model artefacts. `model_checkpoint.pth` is the weight file saved after training. Swap in a newer checkpoint at any time – the API loads it automatically.                                                                                                                        |
| **`data/`**<br>  └ `example_data.json`     | One miniature scenario in the *raw* format of the dataset.                                                                                                                                           |
| **`example.py`**                           | Script that calls the API end-to-end: raw JSON → sub-goal (x, y). Run it with `python example.py` to see a coordinate print-out.                                                                                                                       |
| **`requirements.txt`**                     | Exact Python dependencies (torch + CPU wheels, numpy, opencv-python, matplotlib, pytest). Create an env with `pip install -r requirements.txt`.                                                                                                                                                |

---

## 🚀 Quick-start

```bash
# 1. set up env (CPU-only fine)
python -m venv bandnet_env
source bandnet_env/bin/activate    
pip install -r requirements.txt

# 2. run a smoke test
python example.py           
```

---

## 🔧 API in 60 seconds

```python
from bandnet_api import BandNetAPI

api = BandNetAPI(ckpt_path="model/model_checkpoint.pth", device="cuda")

# option A – raw JSON file
x, y = api.predict_subgoal_from_file("data/example_data.json",
                                     debug=True) 

# option B – pre-built scenario dict (you generated)
scenario = {"occupancy_map": ..., "start_point": ..., ...}
x, y = api.predict_subgoal_from_scenario(scenario)
```

*Inputs*
`example_data.json` contains:

```
start_point1       ↳ robot pose at t=0
goal_points_1_1    ↳ one goal pose
path_1_1_1..3      ↳ three global RRT* paths (lists of waypoints)
pointcloud1.grid_map 100×100 occupancy grid (0 = free, 1 = obstacle)
```

The dataloader converts this into:

| Tensor      | Shape         | Meaning                                              |
| ----------- | ------------- | ---------------------------------------------------- |
| `map_4ch`   | (1,4,100,100) | valid mask · path channel · goal one-hot · band-norm |
| `band_mean` | (1,10)        | mean distance (m) of each band to the path           |
| `dist_goal` | (1,1)         | scalar robot→goal distance (m)                       |

Band Net then predicts a band `b ∈ {0..9}` and it samples **uniformly** inside that band, returning world-metre coordinates.

---

## 📊 Debug visualisation

Pass `debug=True` to any `predict_*` call to open a 100×100 heat-map:

* **Colour** = band index (viridis 0-9, white = invalid)
* **Red ✕**  = randomly sampled cell that became the sub-goal

---

## 🛠️ Extending / replacing pieces

| You need to…                     | Edit / add                                                                    |
| -------------------------------- | ----------------------------------------------------------------------------- |
| **Swap to a newer checkpoint**   | Replace `model/model_checkpoint.pth`                                          |
| **Change map resolution / size** | Update constants at the top of `dataloading.py`                               |
| **Custom coordinate transforms** | Override `world_to_map_coords` & `map_to_world_coords` in `dataloading.py` |
| **Batch inference**              | Use `BandNetAPI.predict_subgoal_batch(...)` (returns list of (x, y))          |

---



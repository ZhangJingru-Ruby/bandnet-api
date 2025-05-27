"""
dataloading.py
==============

Utilities for turning *raw JSON scenario files* (see `data/example_data.json`)
into the three tensors expected by `BandNetAPI`:

    map_4ch   : (1, 4, 100, 100)  float32 [0-1]
    band_mean : (1, 10)           float32
    dist_goal : (1, 1)            float32   (robot → goal distance, metres)

Design notes
------------
* • No torch-only dependencies leak to the user – everything here is pure NumPy /
    OpenCV except the final tensor conversion.
* • Band generation & valid mask logic is copied from your training pipeline
    so results stay consistent.
* • Constants (map size, resolution, etc.) are collected at the top for
    easy tweaking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing   import Dict, List, Tuple

import cv2
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# 🧩  Constants – keep in sync with training code
# --------------------------------------------------------------------------- #
MAP_SIZE          = 100       # map is 100 × 100 cells
MAP_RES           = 0.05      # 1 cell = 5 cm
N_BANDS           = 10
SAFE_DIST_M       = 0.30      # metres – obstacle inflation
BOUNDS_WIDTH_M    = 1.00      # inner dead-zone radius around robot
NEAR_PATH_RADIUS  = 25        # pixels – corridor half-width for valid cells
DEVICE_DEFAULT    = torch.device("cpu")


def world_to_map_coords(g_x: float, g_y: float, start_x: float, start_y: float, map_resolution: float = 0.05):
    map_size = 100
    # 计算地图左下角原点（世界坐标）
    origin_x = start_x - (map_size // 2) * map_resolution  # start_x - 2.5 meters
    origin_y = start_y - (map_size // 2) * map_resolution  # start_y - 2.5 meters
    
    # 转换为地图索引（左下角为原点）
    mx = int((g_x - origin_x) / map_resolution)
    my = int((g_y - origin_y) / map_resolution)
    
    return mx, my


def map_to_world_coords(mx: int, my: int, start_x: float, start_y: float, map_resolution: float = 0.05) :
    map_size = 100
    half_size = map_size // 2
    # 计算地图左下角原点（世界坐标）
    origin_x = start_x - half_size * map_resolution  # start_x - 2.5 meters
    origin_y = start_y - half_size * map_resolution  # start_y - 2.5 meters
    
    # 计算实际世界坐标
    g_x = origin_x + mx * map_resolution
    g_y = origin_y + my * map_resolution
    
    return g_x, g_y


# --------------------------------------------------------------------------- #
# 📥  Raw-JSON helpers
# --------------------------------------------------------------------------- #
def load_raw_json(path: str | Path) -> Dict:
    """Load the raw scenario JSON exactly as on disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r") as f:
        return json.load(f)


def scenario_from_json(raw: Dict) -> Dict:
    """
    Re-shape the odd key pattern of the recording script into a cleaner dict.

    Assumes *one* start, *one* goal, and exactly three paths:
        start_point1
        goal_points_1_1
        path_1_1_1 / _2 / _3
        pointcloud1
    """
    occ_map = np.array(raw["pointcloud1"]["grid_map"], dtype=np.uint8)
    start   = raw["start_point1"]
    goal    = raw["goal_points_1_1"]
    paths   = [raw[f"path_1_1_{i}"] for i in (1, 2, 3)]

    return dict(
        occupancy_map = occ_map,
        start_point   = start,
        goal_point    = goal,
        path_dicts    = paths,          # list of 3 path dicts
    )


# --------------------------------------------------------------------------- #
# 🖍️  Path, mask & band helpers
# --------------------------------------------------------------------------- #
def preprocess_paths(paths: List[Dict], target_num: int = 3) -> List[Dict]:
    """
    Ensure *exactly* `target_num` paths by duplicating or truncating.

    This keeps shapes fixed downstream.
    """
    if len(paths) < target_num:
        paths += [paths[-1]] * (target_num - len(paths))
    elif len(paths) > target_num:
        paths = paths[:target_num]
    return paths


def draw_path_lines(paths   : List[Dict],
                    start_pt: Dict) -> np.ndarray:
    """
    Rasterise all paths into a binary mask – path pixels = 1, else 0.
    """
    mask = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)

    for p_dict in paths:
        pts_map: List[Tuple[int, int]] = [
            world_to_map_coords(
                pt["position"][0], pt["position"][1],
                start_pt["x"],      start_pt["y"],
                MAP_RES)
            for pt in p_dict["path"]
        ]
        for i in range(len(pts_map) - 1):
            cv2.line(mask,
                     pts_map[i], pts_map[i + 1],
                     color=1, thickness=1)

    return mask


def compute_path_dist_map_fast(occupancy_map: np.ndarray,
                               paths        : List[Dict],
                               start_pt     : Dict) -> np.ndarray:
    """
    Return a **float32 (100, 100)** array – for each free cell, the Euclidean
    distance (metres) to the nearest rasterised global-path pixel.
    """
    # 1) build *path pixels = 0* mask
    mask = np.ones_like(occupancy_map, dtype=np.uint8)
    path_pix = draw_path_lines(paths, start_pt)
    mask[path_pix == 1] = 0          # 0 where path

    # 2) distance transform in pixels
    dist_pix = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 3) convert to metres
    return dist_pix.astype(np.float32) * MAP_RES


def build_valid_mask_and_bands(occ_map    : np.ndarray,
                               paths      : List[Dict],
                               start_pt   : Dict,
                               safe_dist_m: float       = SAFE_DIST_M,
                               bounds_m   : float       = BOUNDS_WIDTH_M,
                               device     = DEVICE_DEFAULT
                               ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Core logic reused from training:

        • valid_mask      : (100,100) bool
        • band_idx_map    : (100,100) long  (–1 = invalid)
        • band_mean_dist  : (10,)     float np.ndarray
    """
    # --------- 1. path distance map ------------------------------
    dist_map = compute_path_dist_map_fast(occ_map, paths, start_pt)  # metres

    # --------- 2. rule-based masks -------------------------------
    valid_mask = np.ones_like(occ_map, dtype=bool)  # start with all True

    # 2a. boundary dead-zone
    centre_px = MAP_SIZE // 2
    bounds_px = int((MAP_SIZE * MAP_RES / 2 - bounds_m) / MAP_RES)
    lo, hi    = centre_px - bounds_px, centre_px + bounds_px
    valid_mask[lo:hi, lo:hi] = False

    # 2b. obstacle inflation
    if safe_dist_m > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (int(2 * safe_dist_m / MAP_RES + 1),
             int(2 * safe_dist_m / MAP_RES + 1)))
        dilated = cv2.dilate(occ_map.astype(np.uint8), kernel)
        valid_mask[dilated > 0] = False

    # 2c. near-path corridor
    path_corridor = cv2.dilate(
        draw_path_lines(paths, start_pt),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * NEAR_PATH_RADIUS + 1,
                                   2 * NEAR_PATH_RADIUS + 1)))
    valid_mask[path_corridor == 0] = False

    # --------- 3. band index map & means -------------------------
    band_idx_map = np.full_like(occ_map, fill_value=-1, dtype=np.int64)

    # ignore invalid cells when computing max distance
    dist_valid = dist_map[valid_mask]
    if dist_valid.size == 0:
        max_d = 1e-3  # avoid div-by-zero
    else:
        max_d = dist_valid.max()

    band_width = max_d / N_BANDS if N_BANDS > 0 else max_d

    for b in range(N_BANDS):
        lo, hi = b * band_width, (b + 1) * band_width
        band_cells = (dist_map >= lo) & (dist_map < hi) & valid_mask
        band_idx_map[band_cells] = b

    # mean distance per band
    band_mean_dist = np.zeros(N_BANDS, dtype=np.float32)
    for b in range(N_BANDS):
        d = dist_map[band_idx_map == b]
        band_mean_dist[b] = d.mean() if d.size else 0.0

    # convert to torch
    valid_mask_t   = torch.from_numpy(valid_mask)
    band_idx_map_t = torch.from_numpy(band_idx_map)

    return valid_mask_t, band_idx_map_t, band_mean_dist


# --------------------------------------------------------------------------- #
# 🎁  Main preprocessing entry point
# --------------------------------------------------------------------------- #
def preprocess_for_model(scenario: Dict,
                         device  = DEVICE_DEFAULT
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    End-to-end helper used by `BandNetAPI`:

    Parameters
    ----------
    scenario : dict
        Output of `scenario_from_json()`.
    device : torch.device
        Where to place tensors (CPU / CUDA).

    Returns
    -------
    map_4ch   : torch.FloatTensor  (1,4,100,100)
    band_mean : torch.FloatTensor  (1,10)
    dist_goal : torch.FloatTensor  (1,1)
    """
    occ_map   = scenario["occupancy_map"]
    start_pt  = scenario["start_point"]
    goal_pt   = scenario["goal_point"]
    paths     = preprocess_paths(scenario["path_dicts"])

    # 1) valid mask & bands
    valid, band_idx, band_mean = build_valid_mask_and_bands(
        occ_map, paths, start_pt)

    # --- Channel-0 (valid mask as float)
    ch0 = valid.float()

    # --- Channel-1 (rasterised paths)
    ch1 = torch.zeros_like(ch0)
    for p_dict in paths:
        for pt in p_dict["path"]:
            mx, my = world_to_map_coords(
                pt["position"][0], pt["position"][1],
                start_pt["x"],     start_pt["y"],
                MAP_RES)
            if 0 <= mx < MAP_SIZE and 0 <= my < MAP_SIZE:
                ch1[mx, my] = 1.0

    # --- Channel-2 (goal one-hot)
    ch2 = torch.zeros_like(ch0)
    gx, gy = world_to_map_coords(
        goal_pt["x"], goal_pt["y"],
        start_pt["x"], start_pt["y"],
        MAP_RES)
    if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
        ch2[gx, gy] = 1.0

    # --- Channel-3 (band index normalised to 0-1; invalid = 1)
    band_norm = band_idx.clone().float()
    band_norm[band_norm == -1] = N_BANDS
    band_norm = band_norm / (N_BANDS - 1)
    ch3 = torch.clamp(band_norm, 0.0, 1.0)

        # --- Stack & expand dims ------------------------------------
    map_4ch = torch.stack([ch0, ch1, ch2, ch3], dim=0)  # (4,100,100)
    map_4ch = map_4ch.unsqueeze(0).to(device)           # (1,4,100,100)

    band_mean_t = torch.tensor(band_mean, dtype=torch.float32,
                               device=device).unsqueeze(0)   # (1,10)

    dist_goal = np.linalg.norm(
        [(goal_pt["x"] - start_pt["x"]),
         (goal_pt["y"] - start_pt["y"])])
    dist_goal_t = torch.tensor([[dist_goal]], dtype=torch.float32,
                               device=device)                # (1,1)

    # NEW: send band map + start_pos back too
    band_idx_map_t = band_idx.to(device)                     # (100,100) long
    # 1️⃣ make sure it's 2-D before going NumPy
    band_map = band_idx_map_t.squeeze(0).cpu().numpy()   # (100,100)

    return (
        map_4ch,        # (1,4,100,100)  float32
        band_mean_t,    # (1,10)         float32
        dist_goal_t,    # (1,1)          float32
        band_map, # (100,100)      int64    0-9 valid, –1 invalid
        start_pt,       # original dict  {"x":.., "y":..}
    )


# --------------------------------------------------------------------------- #
# 🌟  Quick CLI check (optional)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Rudimentary sanity-run
    json_path = Path(__file__).parent / "data" / "example_data.json"
    raw       = load_raw_json(json_path)
    scenario  = scenario_from_json(raw)
    m4, mean10, dgoal, _, _ = preprocess_for_model(scenario)
    print("map_4ch :", m4.shape, m4.dtype)
    print("band_mean:", mean10.shape)
    print("dist_goal:", dgoal.item())

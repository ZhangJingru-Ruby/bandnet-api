"""
BandNet API
===========
A *single‑file wrapper* that turns **raw JSON scenario data** into a concrete
**sub‑goal world coordinate** by:

1.  Loading your trained `SubgoalNetwork`.
2.  Pre‑processing raw inputs (build masks, band map, etc.).
3.  Predicting the **band index** (0‑9).
4.  Uniformly sampling one valid cell inside that band.
5.  Converting the cell back to *(x, y)* metres in world space.

Your partner needs no knowledge of the model internals—just call
`predict_subgoal_from_file()` and drive! 💫
"""
from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pathlib import Path
from typing import Tuple, List
import random
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloading import (
    load_raw_json,
    scenario_from_json,
    preprocess_for_model,
    map_to_world_coords,  
)
from model.subgoal_network import SubgoalNetwork  

# Constants
N_BANDS = 10
MAP_SIZE = 100

def _sample_cell_from_band(band_map: np.ndarray, band_idx: int) -> Tuple[int, int]:
    """Uniformly choose one (mx,my) pixel where *band_map == band_idx*.
    If the chosen band is empty (can happen near map edges), fall back to the
    nearest non‑empty band by expanding ±1, ±2, … ."""
    if band_idx < 0 or band_idx >= N_BANDS:
        raise ValueError(f"band_idx must be 0 - {N_BANDS - 1} , got {band_idx}")

    candidate = np.argwhere(band_map == band_idx)  # (K,2) array of indices
    if candidate.size == 0:
        # Fallback search
        for delta in range(1, N_BANDS):
            for alt in (band_idx - delta, band_idx + delta):
                if 0 <= alt < N_BANDS:
                    cand = np.argwhere(band_map == alt)
                    if cand.size:
                        candidate = cand
                        break
            if candidate.size:
                warnings.warn(
                    f"Band {band_idx} empty; sampling from band {alt} instead.",
                    RuntimeWarning,
                )
                break
    # Final safety check
    if candidate.size == 0:
        raise RuntimeError("No valid cells found in any band — check input masks.")

    idx = random.randrange(len(candidate))
    # Pick one candidate cell → (mx, my)
    mx, my = candidate[idx]          # <-- NumPy gives a length-2 array
    return int(mx), int(my)

    # ────────────────────────── VISUAL DEBUG ──────────────────────────
def visualize_band_and_subgoal(
    band_map: np.ndarray,
    band_idx: int,
    mx: int,
    my: int,
    title: str = "Band selection debug",
):
    """
    Super-light plot: entire band map coloured by index, with the sampled
    (mx,my) pixel marked by a bright red ✕.

    • No fancy libraries — just matplotlib (keeps deps minimal).
    • Single figure, no sub-plots (policy compliance ✅).
    """

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # 0-9 bands → viridis,  -1 (=invalid) → white
    base_cmap = plt.colormaps.get_cmap("viridis")
    cmap = base_cmap.with_extremes(over="white", under="white")

    cmap.set_bad(color="white")
    show_map = np.ma.masked_where(band_map == -1, band_map)

    im = ax.imshow(
        show_map,
        cmap=cmap,
        origin="lower",           # (0,0) bottom-left
        vmin=0,
        vmax=N_BANDS,
    )
    cbar = fig.colorbar(im, ax=ax, ticks=range(N_BANDS))
    cbar.set_label("Band index")

    # Mark the sampled cell
    ax.scatter(my, mx, marker="x", s=120, linewidths=2, color="red", label="Sub-goal")
    ax.set_title(f"{title}  •  chosen band = {band_idx}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


class BandNetAPI:
    """High‑level interface for **sub‑goal generation**.

    Parameters
    ----------
    ckpt_path : str or Path
        Path to `model_checkpoint.pth` (weights of `SubgoalNetwork`).
    device : {"cpu", "cuda", "cuda:0", …}
        Where to run inference. Defaults to CPU so anyone can test.
    """

    # ──────────────────────────────── INIT ────────────────────────────────────
    def __init__(self, ckpt_path: str | Path = "model/model_checkpoint.pth", device: str = "cpu"):
        self.device = torch.device(device)
        self.model = SubgoalNetwork().to(self.device).eval()

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=self.device)
        # Allow missing keys (e.g., if you saved *strict=False* during training)
        self.model.load_state_dict(state, strict=False)

        # Silence potential batch‑norm eval warnings
        warnings.simplefilter("ignore", category=UserWarning)

    # ─────────────────────── PUBLIC CONVENIENCE METHODS ───────────────────────
    def predict_subgoal_from_file(self, json_path: str | Path) -> Tuple[float, float]:
        """End‑to‑end helper: raw JSON → *(x, y)* world coordinate."""
        raw = load_raw_json(json_path)
        scenario = scenario_from_json(raw)
        return self.predict_subgoal_from_scenario(scenario)

    def predict_subgoal_from_scenario(self, scenario: dict) -> Tuple[float, float]:
        """Scenario dict (already parsed) → *(x, y)* metres."""
        (
            map4,
            band_mean,
            dist_goal,
            band_map,
            start_pt,
        ) = preprocess_for_model(scenario)

        return self.predict_subgoal(map4, band_mean, dist_goal, band_map, start_pt)
        

    # ───────────────────────── LOW‑LEVEL CORE CALL ────────────────────────────
    @torch.inference_mode()
    def predict_subgoal(
        self,
        map4: torch.Tensor,
        band_mean: torch.Tensor,
        dist_goal: torch.Tensor,
        band_map: np.ndarray,
        start_pt: dict,
    ) -> Tuple[float, float]:
        """Given pre-processed tensors **and** the band map, return a sampled
        *(x, y)* world coordinate inside the chosen band.

        Notes
        -----
        • `map4`, `band_mean`, `dist_goal` may have **batch dim = 1** or **B**.  
        • We handle only *first* element when batch > 1 for simplicity.
        """
        # ───── 1. Move tensors to the model’s device ──────
        map4 = map4.to(self.device)
        band_mean = band_mean.to(self.device)
        dist_goal = dist_goal.to(self.device)

        # ───── 2. Forward pass ──────
        out = self.model(map4, band_mean, dist_goal)
        band_idx = int(out["band"].flatten()[0].item())  # grab first sample

        # ───── 3. Sample pixel from that band ──────
        mx, my = _sample_cell_from_band(band_map, band_idx)

        visualize_band_and_subgoal(band_map, band_idx, mx, my, title="BandNet API debug view")

        # ───── 4. Convert map idx → world metres ──────
        wx, wy = map_to_world_coords(mx, my, start_pt["x"], start_pt["y"])
        return float(wx), float(wy)

    # ────────────────────────── BATCH VARIANT ────────────────────────────────
    @torch.inference_mode()
    def predict_subgoal_batch(
        self,
        map4: torch.Tensor,
        band_mean: torch.Tensor,
        dist_goal: torch.Tensor,
        band_map: np.ndarray,
        start_pt: dict,
    ) -> List[Tuple[float, float]]:
        """Vectorised version for *B* scenarios sharing the same `band_map` and
        `start_pt` (rare but handy for large‑batch eval)."""
        out = self.model(map4.to(self.device), band_mean.to(self.device), dist_goal.to(self.device))
        bands: List[int] = out["band"].cpu().tolist()
        coords = []
        for b in bands:
            mx, my = _sample_cell_from_band(band_map, int(b))
            coords.append(map_to_world_coords(mx, my, start_pt["x"], start_pt["y"]))
        return coords


# ──────────────────────────── CLI / QUICK TEST ───────────────────────────────
if __name__ == "__main__":
    api = BandNetAPI(device="cpu")
    x, y = api.predict_subgoal_from_file("data/example_data.json")
    print(f"Generated sub‑goal: ({x:.3f}, {y:.3f})")

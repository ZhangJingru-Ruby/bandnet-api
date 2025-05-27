# bandnet.py
# ------------
# A lightweight CNN-MLP hybrid that outputs a categorical distribution
# over 10 safety-corridor bands, given a 4-channel local map
# + global summary features (band_mean_dist, Ot_to_goal).

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class SubgoalNetwork(nn.Module):
    """
    One-step bandit policy network.

    Inputs
    -------
    map4        : Tensor (B, 4, 100, 100)
                  Channels =
                      0: valid free space mask   (1 free, 0 obstacle)
                      1: global paths raster     (1 on path, 0 elsewhere)
                      2: goal one-hot            (1 at goal cell)
                      3: band-index normalised   ([0..1] inside corridor,  >1 outside)
    band_mean   : Tensor (B, 10)   mean dist_path for each of the 10 bands
    ot2goal     : Tensor (B, 1)    current Euclidean distance to goal [m]

    Output dict
    -----------
    logits      : (B, 10) raw scores per band
    probs       : (B, 10) softmax probabilities
    band        : (B,)    sampled band id  ∈ {0..9}
    log_prob    : (B,)    log-probability of the sampled band
    """

    def __init__(self, map_channels: int = 4, num_bands: int = 10):
        super().__init__()

        # ---------- 1. CNN BACKBONE ------------------------------------
        # Reduces 100×100 field to 6×6 while extracting hierarchical
        # spatial features.  Padding=1 keeps feature maps aligned.
        self.conv = nn.Sequential(
            nn.Conv2d(map_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 100→50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 50→25

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                # 25→12

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)                 # 12→6
        )

        # Spatial output size after four 2×2 pools is 6×6
        flat_dim = 128 * 6 * 6             # = 4608

        # ---------- 2. MLP HEAD ---------------------------------------
        #   [ CNN-features | band_mean_dist (10) | Ot_to_goal (1) ] → logits(10)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + num_bands + 1, 256),
            nn.ReLU(),
            nn.Linear(256, num_bands)
        )

        self.value = nn.Sequential(
            nn.Linear(256, 1)    
        )

    # -------------------------- forward ------------------------------
    def forward(self,
                map4: torch.Tensor,
                band_mean: torch.Tensor,
                ot2goal: torch.Tensor):
        """
        map4      : (B,4,100,100)
        band_mean : (B,10)
        ot2goal   : (B,1)
        """
        device = map4.device
        band_mean = band_mean.to(device)    
        ot2goal   = ot2goal.to(device)     

        # 1) CNN feature extractor
        feat = self.conv(map4)                 # (B,128,6,6)
        feat = feat.view(feat.size(0), -1)     # flatten → (B,4608)

        # 2) Concatenate global summary features
        # inside forward, just after .to(device)
        if band_mean.dim() == 1:          # (B*10,) → (B,10)
            band_mean = band_mean.view(-1, 10)
        if ot2goal.dim() == 1:            # (B,) → (B,1)
            ot2goal = ot2goal.unsqueeze(-1)
        # print(f"feat shape: {feat.shape}, band_mean shape: {band_mean.shape}, ot2goal shape: {ot2goal.shape}")
        x = torch.cat([feat, band_mean, ot2goal], dim=-1)  # (B,4608+10+1)

        # 3) MLP → logits
        logits = self.fc(x)                    # (B,10)

        # 4) Build categorical distribution & sample
        dist      = Categorical(logits=logits)
        band      = dist.sample()              # (B,)
        log_prob  = dist.log_prob(band)        # (B,)

        value = self.value(self.fc[0](x)) 

        return {
            "logits": logits,
            "probs":  dist.probs,
            "band":   band,
            "log_prob": log_prob,
            "entropy": dist.entropy(),
            "value":  value.squeeze(-1)     # (B,)
        }

# --------------------------------------------------------------------
# Quick smoke-test
if __name__ == "__main__":
    B = 2
    dummy_map4     = torch.randn(B, 4, 100, 100)
    dummy_bandmean = torch.rand(B, 10)
    dummy_ot2g     = torch.rand(B, 1)

    net = SubgoalNetwork()
    out = net(dummy_map4, dummy_bandmean, dummy_ot2g)
    print("Sampled bands :", out["band"])
    print("Log-prob shape:", out["log_prob"].shape)

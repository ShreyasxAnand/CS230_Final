"""Model factory. Adapt and Erase networks get added here in Stages 4 and 5."""
from __future__ import annotations

import torch.nn as nn


def build_model(name: str, n_chans: int, n_times: int, sfreq: float, n_outputs: int = 2, **kw) -> nn.Module:
    from braindecode.models import EEGNet, ShallowFBCSPNet

    if name == "eegnet":
        # Original EEGNet uses a temporal kernel of half a second (64 at 128 Hz).
        kw.setdefault("kernel_length", int(sfreq // 2))
        return EEGNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, **kw)
    if name == "shallow":
        return ShallowFBCSPNet(n_chans=n_chans, n_outputs=n_outputs, n_times=n_times, **kw)
    raise ValueError(f"Unknown model: {name}")

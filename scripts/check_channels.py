"""Recompute the OpenBMI ∩ Dreyer2023 channel overlap from MOABB metadata.

Reads only channel names — no Dreyer2023 data is downloaded or looked at,
which keeps it untouched for the final test (guide, Stage 7).
"""
from moabb.datasets import Dreyer2023, Lee2019_MI

from eegproj.channels import CHANNELS


def eeg_sensors(ds):
    sensors = ds.METADATA.acquisition.sensors
    return {s for s in sensors if not s.upper().startswith(("EMG", "EOG"))}


lee = eeg_sensors(Lee2019_MI())
dre = eeg_sensors(Dreyer2023())
overlap = lee & dre
print(f"OpenBMI EEG: {len(lee)}  Dreyer2023 EEG: {len(dre)}  overlap: {len(overlap)}")
print("In Dreyer2023 only:", sorted(dre - lee))
assert overlap == set(CHANNELS), f"eegproj.channels.CHANNELS is out of date: {sorted(overlap)}"
print("eegproj.channels.CHANNELS matches.")

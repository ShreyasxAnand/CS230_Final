"""Channel set shared by OpenBMI (Lee2019_MI) and Dreyer2023.

The guide fixes the channel set as the overlap of the two datasets, because the
simulator's head model and the second-dataset test both depend on it.
Dreyer2023 has 27 EEG channels; OpenBMI has every one except FCz, so the
overlap is 26 channels, all over the motor strip.

`scripts/check_channels.py` recomputes this from MOABB metadata.
"""

CHANNELS = [
    "Fz", "F3", "F4",
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
    "P3", "Pz", "P4",
]

assert len(CHANNELS) == 26

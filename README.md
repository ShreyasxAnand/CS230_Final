# Adapt vs. Erase with a simulator answer key

CS230 final project. Plan: see the project guide doc.

## Setup

```bash
conda env create -f environment.yml   # creates the "eeg" env from requirements.txt
conda activate eeg
pytest -q                             # fast checks, no download needed
```

OpenBMI downloads through MOABB into `~/mne_data` on first use, about 1.1 GB per subject
(about 60 GB for all 54). Full runs are meant for AWS; the Mac is for development and
smoke tests on a few subjects. Preprocessed trials are cached in `cache/` (about 31 MB per
subject) and are not committed.

Each session has 200 labelled trials (100 offline + 100 online run) and a 60 s resting
recording taken before the offline run.

## Stage 1: EEGNet baseline

```bash
python -m scripts.check_channels                     # confirms the 26-channel OpenBMI ∩ Dreyer2023 set
python main.py baseline --folds dev --only dev0 --subjects 1-12 --epochs 30   # smoke test
python main.py baseline --model eegnet --folds dev   # 6 development folds, all 54 people
python main.py baseline --model shallow --folds dev  # second baseline
python main.py baseline --folds loso --repeats 5     # final numbers only
```

Protocol: train on the other people's session 1, early-stop on a few held-out training
people's session 2, test on the unseen people's session 2. Normalisation is fitted on the
training people only.

## Layout

```
eegproj/channels.py   fixed 26-channel set
eegproj/data.py       OpenBMI loading: trials (0.5–3.5 s, 250 Hz) + resting segment per session
eegproj/folds.py      dev folds (6×9), LOSO folds, fold-safe scaler
eegproj/models/       model factory (EEGNet, ShallowConvNet; Adapt/Erase later)
eegproj/train.py      shared training loop; batches are dicts so later models can read "rest"
main.py               CLI
scripts/              one-off checks
tests/                synthetic-data smoke tests (pytest)
```

"""OpenBMI loading and preprocessing.

One call per subject returns, for each session:
    X    (n_trials, n_chans, n_times) float32, microvolts
    y    (n_trials,) int64, 0 = left hand, 1 = right hand
    rest (n_chans, n_rest_times) float32, microvolts — the resting recording
         taken before that session's offline run (about one minute)

The rest segment is loaded now because the Adapt network (Stage 4) needs it,
and Stages 4–6 all reuse this pipeline.

Results are cached as .npz under `cache_dir` so the slow .mat parsing and
resampling run once per subject.

Nothing here is fitted to data: filtering, resampling and cropping are fixed
operations, so this is safe to run outside the folds. Normalisation statistics
are fitted inside each fold (see `eegproj.folds.ChannelScaler`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import mne
import numpy as np

from .channels import CHANNELS

log = logging.getLogger(__name__)

# MOABB's Lee2019_MI stim codes.
_EVENT_TO_LABEL = {2: 0, 1: 1}  # left_hand -> 0, right_hand -> 1


@dataclass(frozen=True)
class PreprocConfig:
    sfreq: float = 250.0
    tmin: float = 0.5          # seconds after cue
    tmax: float = 3.5
    l_freq: float | None = 0.5
    h_freq: float | None = 45.0
    car: bool = False          # common average reference over CHANNELS
    # The online run does carry left/right events (50 + 50), despite MOABB's
    # docstring. Including it gives the guide's 200 trials per session.
    use_online_run: bool = True
    rest_run: str = "0preTrainRest"

    def tag(self) -> str:
        """Short string identifying this config, used in cache file names."""
        d = asdict(self)
        return "_".join(f"{k}-{v}" for k, v in sorted(d.items())).replace(".", "p")


def _pick_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Pick CHANNELS in a fixed order, matching names case-insensitively."""
    lookup = {c.lower(): c for c in raw.ch_names}
    missing = [c for c in CHANNELS if c.lower() not in lookup]
    if missing:
        raise ValueError(f"Channels missing from recording: {missing}")
    raw = raw.copy().pick([lookup[c.lower()] for c in CHANNELS])
    raw.rename_channels({lookup[c.lower()]: c for c in CHANNELS})
    return raw


def _filter_resample(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    raw = raw.copy().load_data()
    if cfg.car:
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if cfg.l_freq is not None or cfg.h_freq is not None:
        raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")
    raw.resample(cfg.sfreq, verbose="ERROR")
    return raw


def _epochs_from_run(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> tuple[np.ndarray, np.ndarray]:
    events = mne.find_events(raw, stim_channel="STI 014", shortest_event=0, verbose="ERROR")
    events = events[np.isin(events[:, 2], list(_EVENT_TO_LABEL))]
    if len(events) == 0:
        return np.empty((0, len(CHANNELS), 0), np.float32), np.empty(0, np.int64)

    eeg = _filter_resample(_pick_channels(raw), cfg)
    # Event samples were found at the original rate; rescale to the new rate.
    scale = cfg.sfreq / raw.info["sfreq"]
    onsets = np.round(events[:, 0] * scale).astype(int) - int(round(raw.first_samp * scale))
    start = int(round(cfg.tmin * cfg.sfreq))
    n_times = int(round((cfg.tmax - cfg.tmin) * cfg.sfreq))

    data = eeg.get_data(units="uV")
    X, keep = [], []
    for i, o in enumerate(onsets):
        a = o + start
        if a >= 0 and a + n_times <= data.shape[1]:
            X.append(data[:, a:a + n_times])
            keep.append(i)
    if len(keep) < len(onsets):
        log.warning("Dropped %d trials that ran past the recording edge", len(onsets) - len(keep))
    y = np.array([_EVENT_TO_LABEL[c] for c in events[keep, 2]], np.int64)
    return np.stack(X).astype(np.float32), y


def _rest_from_run(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> np.ndarray:
    eeg = _filter_resample(_pick_channels(raw), cfg)
    return eeg.get_data(units="uV").astype(np.float32)


def load_subject(subject: int, cfg: PreprocConfig = PreprocConfig(),
                 cache_dir: str | Path = "cache") -> dict[str, dict[str, np.ndarray]]:
    """Return {"1": {"X", "y", "rest"}, "2": {...}} for one OpenBMI subject."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"openbmi_s{subject:02d}_{cfg.tag()}.npz"

    if path.exists():
        z = np.load(path)
        sessions = sorted({k.split("/")[0] for k in z.files})
        return {s: {f: z[f"{s}/{f}"] for f in ("X", "y", "rest")} for s in sessions}

    from moabb.datasets import Lee2019_MI

    ds = Lee2019_MI(train_run=True, test_run=cfg.use_online_run, resting_state=True)
    # get_data() runs MOABB's annotation pipeline, which fails on rest runs
    # (they have no task events). Read the raw runs directly instead; we find
    # events ourselves from the stim channel.
    runs_by_session = ds._get_single_subject_data(subject)

    out: dict[str, dict[str, np.ndarray]] = {}
    # MOABB's session keys vary by version ("0"/"1", "session_1"...); sort them
    # and call them "1" and "2" in recording order.
    for i, sess_name in enumerate(sorted(runs_by_session)):
        runs = runs_by_session[sess_name]
        sess = str(i + 1)
        Xs, ys = [], []
        for run in ("1train", "4test"):
            if run in runs:
                X, y = _epochs_from_run(runs[run], cfg)
                if len(y):
                    Xs.append(X)
                    ys.append(y)
                elif run == "4test":
                    log.warning("Subject %d session %s: online run has no labelled events", subject, sess)
        out[sess] = {
            "X": np.concatenate(Xs),
            "y": np.concatenate(ys),
            "rest": _rest_from_run(runs[cfg.rest_run], cfg),
        }

    np.savez(path, **{f"{s}/{f}": v for s, d in out.items() for f, v in d.items()})
    return out


def load_subjects(subjects, cfg: PreprocConfig = PreprocConfig(), cache_dir="cache"):
    return {s: load_subject(s, cfg, cache_dir) for s in subjects}

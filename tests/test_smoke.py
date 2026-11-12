"""Fast checks that need no dataset download. Run with: pytest -q"""
import numpy as np
import torch

from eegproj.channels import CHANNELS
from eegproj.folds import ChannelScaler, dev_folds, loso_folds
from eegproj.models import build_model
from eegproj.train import TrainConfig, TrialDataset, fit, predict


def test_dev_folds_cover_everyone_once():
    folds = dev_folds()
    assert len(folds) == 6
    tested = sorted(s for f in folds for s in f.test)
    assert tested == list(range(1, 55))
    for f in folds:
        assert not set(f.test) & set(f.train)
        assert not set(f.test) & set(f.val)
        assert not set(f.val) & set(f.train)


def test_loso_folds():
    folds = loso_folds()
    assert len(folds) == 54
    assert all(len(f.test) == 1 and f.test[0] not in f.train + f.val for f in folds)


def test_scaler_uses_train_stats_only():
    rng = np.random.default_rng(0)
    tr = rng.normal(5, 2, (20, 3, 50)).astype(np.float32)
    te = rng.normal(100, 2, (5, 3, 50)).astype(np.float32)
    s = ChannelScaler().fit(tr)
    assert abs(s.transform(tr).mean()) < 1e-4
    assert s.transform(te).mean() > 10  # test data not re-centred


def test_models_train_on_learnable_synthetic_data():
    """A class-dependent 10 Hz burst on C3 vs C4: every model should beat chance."""
    rng = np.random.default_rng(0)
    n, c, t, sfreq = 256, len(CHANNELS), 750, 250.0
    X = rng.normal(0, 1, (n, c, t)).astype(np.float32)
    y = rng.integers(0, 2, n)
    wave = 3 * np.sin(2 * np.pi * 10 * np.arange(t) / sfreq).astype(np.float32)
    X[y == 0, CHANNELS.index("C4")] += wave
    X[y == 1, CHANNELS.index("C3")] += wave
    subj = np.zeros(n, dtype=np.int64)

    tr = TrialDataset(X[:192], y[:192], subj[:192])
    te = TrialDataset(X[192:], y[192:], subj[192:])
    for name in ("eegnet", "shallow"):
        torch.manual_seed(0)
        model = build_model(name, n_chans=c, n_times=t, sfreq=sfreq)
        model, _ = fit(model, tr, te, TrainConfig(epochs=8, patience=8), torch.device("cpu"), log_every=0)
        p, yy, _ = predict(model, torch.utils.data.DataLoader(te, batch_size=64), torch.device("cpu"))
        assert (p == yy).mean() > 0.7, name

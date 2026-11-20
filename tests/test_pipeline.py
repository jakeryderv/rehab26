import numpy as np
import pandas as pd
import torch

from rehab26.data import participant_folds, resample
from rehab26.features import (L_SH, R_SH, biomech_features, canonicalize_rotation, normalize_by_group,
                              relative_to_start, shoulder_yaw)
from rehab26.models import MODELS, build_model
from rehab26.train import Augment


def random_pose(n=6, t=64, seed=0):
    return np.random.default_rng(seed).normal(size=(n, t, 33, 3)).astype(np.float32)


def test_resample_keeps_endpoints_and_is_linear():
    seq = np.arange(10, dtype=float)[:, None] * np.ones((1, 3))
    out = resample(seq, 19)
    assert out.shape == (19, 3)
    np.testing.assert_allclose(out[[0, -1], 0], [0, 9])
    np.testing.assert_allclose(np.diff(out[:, 0]), 0.5)


def test_rotation_alignment_removes_yaw():
    P = random_pose() * 0.01
    P[:, :, L_SH] += [0.2, 0, 0.1]
    P[:, :, R_SH] -= [0.2, 0, 0.1]
    assert np.abs(shoulder_yaw(P)).mean() > 10
    assert np.abs(shoulder_yaw(canonicalize_rotation(P))).mean() < 5


def test_biomech_is_invariant_to_rotation_about_vertical():
    P = random_pose()
    c, s = np.cos(0.7), np.sin(0.7)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    np.testing.assert_allclose(biomech_features(P)[0], biomech_features(P @ R.T)[0], atol=1e-3)


def test_group_normalization_and_start_relative():
    seq = np.random.default_rng(0).normal(3, 2, size=(8, 16, 5)).astype(np.float32)
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    z = normalize_by_group(seq, groups)
    np.testing.assert_allclose(z[groups == 1].mean(axis=(0, 1)), 0, atol=1e-4)
    np.testing.assert_allclose(z[groups == 1].std(axis=(0, 1)), 1, atol=1e-3)
    np.testing.assert_allclose(relative_to_start(seq, 0.1)[:, 0], 0, atol=1e-6)


def test_folds_never_share_a_participant_or_touch_the_test_person():
    meta = pd.DataFrame({"person": list("AABBCCDDEE")})
    folds = participant_folds(meta, test_person="E")
    assert [v for _, _, v in folds] == list("ABCD")
    for train, val, _ in folds:
        assert not set(meta.person[train]) & set(meta.person[val])
        assert "E" not in set(meta.person[train]) | set(meta.person[val])


def test_models_and_augmentation_shapes():
    x, ex = torch.randn(4, 99, 64), torch.eye(3)[[0, 1, 2, 0]]
    aug = Augment(time_warp=0.5, jitter=0.05, rotate=20, scale=0.1, feature_std=torch.ones(99))
    assert aug(x).shape == x.shape
    for kind in MODELS:
        model = build_model(kind, 99)
        model.fit_scaler(x)
        assert model(aug(x), ex).shape == (4,)


def test_rotation_augmentation_preserves_joint_distances():
    x = torch.randn(2, 99, 8)
    out = Augment(rotate=45)(x)
    d = lambda a: (a.view(2, 33, 3, 8)[:, 0] - a.view(2, 33, 3, 8)[:, 1]).norm(dim=1)
    torch.testing.assert_close(d(x), d(out), atol=1e-4, rtol=1e-4)

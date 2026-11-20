"""Load REHAB26-ViewAngles and slice it into fixed-length repetition sequences."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

EX_NAME = {1: "Arm abduction", 6: "Squat", 7: "Shoulder extension"}
EX_INDEX = {1: 0, 6: 1, 7: 2}
CAM_ANGLE = {1: 0, 2: 45, 3: 90}  # approximate camera positions around the participant (degrees)


def load_segmentation(data_dir=DATA_DIR):
    """One row per labeled repetition. y is 1 for incorrect, 0 for correct."""
    seg = pd.read_csv(Path(data_dir) / "raw/Segmentation.csv", sep=";")
    seg["y"] = (seg.correctness_score == 5).astype(int)
    seg["error"] = seg.error_description.fillna("correct")
    seg["n_frames"] = seg.last_frame - seg.first_frame + 1
    seg["rep_id"] = np.arange(len(seg))
    return seg


def resample(seq, n):
    """Linearly interpolate a (frames, ...) array to n evenly spaced time steps."""
    pos = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (pos - lo).reshape(-1, *[1] * (seq.ndim - 1))
    return seq[lo] * (1 - w) + seq[hi] * w


def load_dataset(n_frames=64, data_dir=DATA_DIR):
    """Return (X, meta): X is (N, n_frames, 33, 4) and meta has one row per repetition x camera.

    The last axis of X is x, y, z in metres centred on the hips, then MediaPipe's visibility score.
    `view` in meta is the camera position minus the participant's orientation: 0 is frontal, and negative
    values put the exercising (right) arm on the far side from the camera.
    """
    data_dir = Path(data_dir)
    seg = load_segmentation(data_dir)
    X, rows = [], []
    for (vid, ex), reps in seg.groupby(["video_id", "exercise_id"]):
        for cam in (1, 2, 3):
            arr = np.load(data_dir / f"mediapipe/ex{ex}/{vid}_cam{cam}_3d.npy")
            for r in reps.itertuples():
                clip = arr[r.first_frame:r.last_frame + 1]
                if len(clip) != r.n_frames or np.abs(clip[:, :, :3]).sum(axis=(1, 2)).min() == 0:
                    raise ValueError(f"repetition {r.rep_id} of {vid} cam{cam} is truncated or has undetected frames")
                X.append(resample(clip, n_frames))
                rows.append(dict(rep_id=r.rep_id, person=r.person, exercise_id=ex, exercise=EX_NAME[ex], cam=cam,
                                 orientation=r.person_orientation, view=CAM_ANGLE[cam] - r.person_orientation,
                                 y=r.y, error=r.error, n_frames=r.n_frames))
    return np.stack(X).astype(np.float32), pd.DataFrame(rows)


def participant_folds(meta, test_person):
    """Leave-one-participant-out folds over everyone except the held-out test participant.

    Returns a list of (train_idx, val_idx, val_person).
    """
    dev = [p for p in sorted(meta.person.unique()) if p != test_person]
    folds = []
    for v in dev:
        train = np.where(meta.person.isin([p for p in dev if p != v]))[0]
        val = np.where(meta.person == v)[0]
        folds.append((train, val, v))
    return folds

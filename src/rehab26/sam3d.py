"""Load SAM 3D Body pilot output in the same 33-joint layout as the MediaPipe arrays.

SAM 3D Body returns 70 keypoints (MHR70). Mapping them onto MediaPipe's 33 indices lets the same feature
code run on both pose sources. MediaPipe joints with no direct counterpart are filled from the nearest one.
"""
from pathlib import Path

import numpy as np
import pandas as pd


# MediaPipe index -> MHR70 index
MP_FROM_MHR = {
    0: 0,                                   # nose
    1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,     # eyes (inner / centre / outer all map to the one eye point)
    7: 3, 8: 4,                             # ears
    9: 0, 10: 0,                            # mouth corners -> nose (no counterpart)
    11: 5, 12: 6, 13: 7, 14: 8,             # shoulders, elbows
    15: 62, 16: 41,                         # wrists
    17: 58, 18: 37, 19: 46, 20: 25, 21: 42, 22: 21,   # pinky, index, thumb tips
    23: 9, 24: 10, 25: 11, 26: 12, 27: 13, 28: 14,    # hips, knees, ankles
    29: 17, 30: 20, 31: 15, 32: 18,         # heels, big toes
}
_ORDER = [MP_FROM_MHR[i] for i in range(33)]


def to_mediapipe_layout(kp3d):
    """(..., 70, 3) MHR70 keypoints -> (..., 33, 3), centred on the hip midpoint."""
    p = kp3d[..., _ORDER, :]
    return p - p[..., [23, 24], :].mean(axis=-2, keepdims=True)


def load_sam3d(out_dir, n_frames=64):
    """Return (P, index): P is (N, n_frames, 33, 3) and index has rep_id, cam, n_samples, failed per row.

    Frames where the model returned nothing are linearly interpolated from their neighbours.
    """
    P, rows = [], []
    for f in sorted(Path(out_dir).glob("rep*_cam*.npz")):
        if ".tmp" in f.name:
            continue
        d = np.load(f)
        kp = d["kp3d"].astype(np.float64)
        bad = ~np.isfinite(kp).all(axis=(1, 2))
        if bad.all():
            continue
        if bad.any():
            t = np.arange(len(kp))
            flat = kp.reshape(len(kp), -1)
            for j in range(flat.shape[1]):
                flat[bad, j] = np.interp(t[bad], t[~bad], flat[~bad, j])
            kp = flat.reshape(kp.shape)
        # samples are evenly spaced except possibly the appended last frame; resample on true frame positions
        frames = d["frames"].astype(float)
        pos = np.linspace(frames[0], frames[-1], n_frames)
        flat = kp.reshape(len(kp), -1)
        seq = np.stack([np.interp(pos, frames, flat[:, j]) for j in range(flat.shape[1])], axis=1).reshape(n_frames, 70, 3)
        P.append(to_mediapipe_layout(seq))
        rep, cam = f.stem.split("_")
        rows.append(dict(rep_id=int(rep[3:]), cam=int(cam[3:]), n_samples=len(frames), failed=int(bad.sum()),
                         seconds=float(d["seconds"].mean())))
    return np.stack(P).astype(np.float32), pd.DataFrame(rows)

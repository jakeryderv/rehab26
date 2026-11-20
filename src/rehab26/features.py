"""Feature representations of pose sequences, and per-group normalizations.

Positions are arrays P of shape (N, T, 33, 3). Feature sets are (N, T, F).
"""
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24

FEATURE_LABEL = {
    "raw": "Raw 3D coordinates",
    "scaled": "Scaled coordinates",
    "aligned": "Scaled + rotation-aligned coordinates",
    "biomech": "Biomechanical features",
}
COORDINATE_FEATURES = ("raw", "scaled", "aligned")  # laid out as joint-major x, y, z triples


def torso_length(P):
    mid_sh = (P[:, :, L_SH] + P[:, :, R_SH]) / 2
    mid_hip = (P[:, :, L_HIP] + P[:, :, R_HIP]) / 2
    return np.linalg.norm(mid_sh - mid_hip, axis=-1).mean(axis=1)


def scale_normalize(P):
    """Divide by each repetition's mean torso length so body size does not matter."""
    return P / torso_length(P)[:, None, None, None]


def canonicalize_rotation(P, start_frac=0.1):
    """Rotate each repetition about the vertical (y) axis so its starting pose faces the camera."""
    k = max(1, int(P.shape[1] * start_frac))
    d = ((P[:, :k, L_SH] - P[:, :k, R_SH]) + (P[:, :k, L_HIP] - P[:, :k, R_HIP])).mean(axis=1)
    theta = np.arctan2(d[:, 2], d[:, 0])
    c, s = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    out = P.copy()
    out[..., 0] = c * P[..., 0] + s * P[..., 2]
    out[..., 2] = -s * P[..., 0] + c * P[..., 2]
    return out


def shoulder_yaw(P):
    """Direction of the shoulder line seen from above, in degrees, averaged over each repetition."""
    d = (P[:, :, L_SH] - P[:, :, R_SH]).mean(axis=1)
    return np.degrees(np.arctan2(d[:, 2], d[:, 0]))


def vector_angle(u, v, unsigned_axis=False):
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    if unsigned_axis:
        cos = np.abs(cos)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angle(a, b, c):
    """Angle at b (degrees) between segments b->a and b->c."""
    return vector_angle(a - b, c - b)


def biomech_features(P):
    """Joint angles, segment tilts and stance-width ratios, plus their velocities. Returns (features, names)."""
    J = lambda i: P[:, :, i]
    mid_sh, mid_hip = (J(11) + J(12)) / 2, (J(23) + J(24)) / 2
    trunk, sh_line, hip_line = mid_sh - mid_hip, J(11) - J(12), J(23) - J(24)
    vertical = np.array([0.0, 1.0, 0.0], dtype=P.dtype)
    hip_width = np.linalg.norm(hip_line, axis=-1) + 1e-9
    f = {
        "elbow_L": joint_angle(J(11), J(13), J(15)), "elbow_R": joint_angle(J(12), J(14), J(16)),
        "shoulder_L": joint_angle(J(23), J(11), J(13)), "shoulder_R": joint_angle(J(24), J(12), J(14)),
        "hip_L": joint_angle(J(11), J(23), J(25)), "hip_R": joint_angle(J(12), J(24), J(26)),
        "knee_L": joint_angle(J(23), J(25), J(27)), "knee_R": joint_angle(J(24), J(26), J(28)),
        "ankle_L": joint_angle(J(25), J(27), J(31)), "ankle_R": joint_angle(J(26), J(28), J(32)),
        "trunk_lean": vector_angle(trunk, vertical, unsigned_axis=True),
        "shoulder_tilt": vector_angle(sh_line, vertical),
        "hip_tilt": vector_angle(hip_line, vertical),
        "trunk_twist": vector_angle(sh_line, hip_line),
        "head_tilt": vector_angle(J(7) - J(8), sh_line),
        "neck": vector_angle(J(0) - mid_sh, trunk),
        "arm_plane_L": vector_angle(J(13) - J(11), sh_line), "arm_plane_R": vector_angle(J(14) - J(12), sh_line),
        "foot_incline_L": vector_angle(J(31) - J(29), vertical), "foot_incline_R": vector_angle(J(32) - J(30), vertical),
    }
    feats = [v / 180.0 for v in f.values()]
    names = list(f)
    feats += [np.linalg.norm(J(25) - J(26), axis=-1) / hip_width, np.linalg.norm(J(27) - J(28), axis=-1) / hip_width]
    names += ["knee_width_ratio", "ankle_width_ratio"]
    static = np.stack(feats, axis=-1)
    velocity = np.gradient(static, axis=1)  # per normalised time step
    return np.concatenate([static, velocity], axis=-1).astype(np.float32), names + [n + "_vel" for n in names]


def build_features(P):
    """All four feature sets for positions P, keyed as in FEATURE_LABEL."""
    n, t = P.shape[:2]
    scaled = scale_normalize(P)
    return {
        "raw": P.reshape(n, t, -1),
        "scaled": scaled.reshape(n, t, -1),
        "aligned": canonicalize_rotation(scaled).reshape(n, t, -1),
        "biomech": biomech_features(P)[0],
    }


def summarize(seq):
    """Per-repetition summary statistics over time: mean, min, max, range of every feature."""
    return np.concatenate([seq.mean(1), seq.min(1), seq.max(1), seq.max(1) - seq.min(1)], axis=1)


def normalize_by_group(seq, groups):
    """Z-score every feature within each group, using all of that group's sequences and frames.

    No labels are used. With groups = participant x exercise this removes a person's habitual posture and
    range of motion, at the cost of needing several repetitions from that person before predicting.
    """
    out = np.empty_like(seq)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        m = groups == g
        mu = seq[m].mean(axis=(0, 1))
        sd = seq[m].std(axis=(0, 1)) + 1e-6
        out[m] = (seq[m] - mu) / sd
    return out


def relative_to_start(seq, start_frac=0.1):
    """Express each repetition as change from the mean of its own first frames."""
    k = max(1, int(seq.shape[1] * start_frac))
    return seq - seq[:, :k].mean(axis=1, keepdims=True)

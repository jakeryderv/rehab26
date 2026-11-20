"""Run SAM 3D Body on selected REHAB26 repetitions.

Run this with the SAM environment of the vbmt project, which holds the pinned model source and checkpoint:

    ~/dev/projects/vbmt/.venv-sam/bin/python scripts/sam3d_pilot.py data/sam3d/pilot/jobs.csv data/sam3d/pilot

jobs.csv has one row per repetition x camera: rep_id, video_id, exercise_id, cam, first_frame, last_frame.
One .npz is written per row (frames, kp3d, kp2d, bbox, seconds). Existing files are skipped, so the run can
be stopped and resumed. The person box for each repetition comes from its MediaPipe 2D landmarks.
"""
import argparse
import contextlib
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FPS = 30


def probe_size(video):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(video)], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    return int(s["width"]), int(s["height"])


def read_frames(video, first, count, size):
    """Yield `count` consecutive RGB frames starting at frame index `first` (frame-accurate seek)."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{(first - 0.5) / FPS:.6f}", "-i", str(video), "-frames:v", str(count),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        for _ in range(count):
            buf = proc.stdout.read(w * h * 3)
            if len(buf) != w * h * 3:
                raise ValueError(f"{video}: decode ended early near frame {first}")
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def person_box(landmarks_2d, size, margin=0.15):
    """Box around all MediaPipe landmarks over the whole repetition, with a margin, in pixels."""
    w, h = size
    xy = landmarks_2d[:, :, :2] * [w, h]
    x0, y0, x1, y1 = xy[..., 0].min(), xy[..., 1].min(), xy[..., 0].max(), xy[..., 1].max()
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return np.array([max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my)], np.float32)


def load_estimator(vbmt):
    import torch
    sys.path.insert(0, str(vbmt / "third_party/sam-3d-body"))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    hub_load = torch.hub.load

    def local_backbone(repo, model_name, **kwargs):
        # The backbone normally comes from a remote hub branch; use the pinned local checkout instead.
        kwargs["source"] = "local"
        return hub_load(str(vbmt / "third_party/dinov3"), model_name, **kwargs)

    ckpt = vbmt / "checkpoints/sam-3d-body-dinov3"
    with patch("torch.hub.load", side_effect=local_backbone), contextlib.redirect_stdout(io.StringIO()):
        model, cfg = load_sam_3d_body(str(ckpt / "model.ckpt"), mhr_path=str(ckpt / "assets/mhr_model.pt"))
    return SAM3DBodyEstimator(model, cfg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--stride", type=int, default=3, help="process every n-th frame, plus the last one")
    ap.add_argument("--inference-type", default="body", choices=["body", "full"])
    ap.add_argument("--vbmt", type=Path, default=Path.home() / "dev/projects/vbmt")
    ap.add_argument("--data", type=Path, default=REPO / "data")
    ap.add_argument("--limit", type=int, help="stop after this many repetitions (for benchmarking)")
    args = ap.parse_args()

    import torch
    jobs = list(csv.DictReader(args.jobs.open()))
    todo = [j for j in jobs if not (args.out / f"rep{int(j['rep_id']):04d}_cam{j['cam']}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo) if not args.limit else '?'} done, running {len(todo)}", flush=True)
    if not todo:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    estimator = load_estimator(args.vbmt)
    sizes, started, n_frames, failures = {}, time.perf_counter(), 0, 0
    for k, j in enumerate(todo):
        ex, vid, cam = j["exercise_id"], j["video_id"], j["cam"]
        first, last = int(j["first_frame"]), int(j["last_frame"])
        video = args.data / f"videos/ex{ex}/{vid}_cam{cam}.mp4"
        size = sizes.setdefault(video, probe_size(video))
        mp2d = np.load(args.data / f"mediapipe/ex{ex}/{vid}_cam{cam}_2d.npy", mmap_mode="r")[first:last + 1]
        box = person_box(np.asarray(mp2d), size)
        count = last - first + 1
        wanted = sorted(set(range(0, count, args.stride)) | {count - 1})
        frames, kp3d, kp2d, seconds = [], [], [], []
        for offset, frame in enumerate(read_frames(video, first, count, size)):
            if offset not in wanted:
                continue
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                outputs = estimator.process_one_image(frame, bboxes=box[None], inference_type=args.inference_type)
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - t0)
            frames.append(first + offset)
            if len(outputs) == 1 and np.isfinite(outputs[0]["pred_keypoints_3d"]).all():
                kp3d.append(outputs[0]["pred_keypoints_3d"].astype(np.float32))
                kp2d.append(outputs[0]["pred_keypoints_2d"].astype(np.float32))
            else:
                failures += 1
                kp3d.append(np.full((70, 3), np.nan, np.float32))
                kp2d.append(np.full((70, 2), np.nan, np.float32))
        n_frames += len(frames)
        tmp = args.out / f"rep{int(j['rep_id']):04d}_cam{cam}.tmp.npz"
        np.savez_compressed(tmp, frames=np.array(frames), kp3d=np.stack(kp3d), kp2d=np.stack(kp2d), bbox=box,
                            seconds=np.array(seconds, np.float32))
        tmp.rename(args.out / f"rep{int(j['rep_id']):04d}_cam{cam}.npz")
        if k % 10 == 0 or k == len(todo) - 1:
            elapsed = time.perf_counter() - started
            print(f"[{k + 1}/{len(todo)}] {vid} cam{cam} rep {j['rep_id']}: {len(frames)} frames, "
                  f"{np.mean(seconds):.3f} s/frame inference, {elapsed / n_frames:.3f} s/frame overall, "
                  f"failures {failures}, eta {elapsed / (k + 1) * (len(todo) - k - 1) / 60:.0f} min", flush=True)
    print(f"done: {n_frames} frames in {(time.perf_counter() - started) / 60:.1f} min, {failures} failed, "
          f"peak GPU {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()

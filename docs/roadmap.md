# Roadmap

Written 2026-11-19. Deadlines: final presentation Dec 1 and 3, final report Dec 10.

## Where the project is going

The dataset paper (Jánošová et al., 2026) shows that camera placement and a second camera change how well
exercise errors can be separated, using MediaPipe poses and a model personalized to each participant. This
project asks the follow-up questions:

1. **Better poses instead of better placement.** Does a stronger single-camera pose model (SAM 3D Body)
   shrink the gap between camera viewpoints, and how does it compare with combining two views?
2. **Unseen people.** How well can correctness be judged for someone the model has never seen, and how much
   does knowing a little about them (per-recording normalization) help?
3. **More data, if time allows.** Does it carry over to other people and exercises in a second dataset?

## For the capstone (by Dec 10)

- [x] Finish SAM 3D Body pose extraction for all exercises and participants.
- [x] Run `notebooks/06_sam3d_all_exercises.ipynb`: MediaPipe vs SAM 3D Body on all three exercises, with
      accuracy by exercise, viewing angle and error type, and one camera against two (averaged
      predictions). Write its findings.
- [ ] Choose the final configuration from the development participants only.
- [ ] Evaluate once on the held-out participant.
- [ ] Error analysis of 5 to 10 failure cases.
- [ ] Applied demonstration on new data.
- [ ] Related Work section, at least three papers.
- [ ] Presentation and final report.

## Follow-up work (after the capstone)

- **Reproduce the paper's separability analysis.** Notebook 06 answers the viewpoint and two-camera
  questions with classification accuracy; this step is only needed to put numbers on the paper's own scale
  (their 16.9% and 13.1% are gains in separability, not accuracy). Rebuild the per-sequence template (time-warped mean and
  confidence band over the paper's fifteen features) and the separability score. First match the paper's
  MediaPipe table to confirm the reimplementation, then swap in SAM 3D Body poses (2D and 3D) and compare
  the viewpoint curve with their single-camera and two-camera results.
- **Personalized to stranger.** Put the paper's personalized setting, per-recording normalization and the
  fully unseen setting on one scale.
- **Second dataset.** The Zenodo record says this dataset continues record 13305825, believed to be
  REHAB24-6 from the same group (to be confirmed). If it has motion-capture ground truth, it also allows
  checking SAM 3D Body's joint angles against a reference, which this dataset cannot.

## Decided against

- **Replacing the labels with angle thresholds from the clinical literature.** The labels already come from
  a physiotherapist, and thresholds computed from estimated poses would be circular. The clinical criteria
  are better used as a rule-based baseline to compare the learned models against.
- **A classifier on raw video.** Five participants are too few; a pixel model would most likely memorize
  people and backgrounds. Future work.

## Caveats to carry into the report

- Per-recording normalization needs several unlabeled repetitions from the new person and benefits from the
  dataset's even mix of correct and incorrect repetitions.
- SAM 3D Body runs at about 0.17 s per frame on a laptop GPU; MediaPipe runs live on a phone.
- The person box for SAM 3D Body comes from MediaPipe's landmarks.
- Five participants in total, one of them held out.

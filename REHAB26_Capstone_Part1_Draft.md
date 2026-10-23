# Evaluating Biomechanical and Temporal Movement Representations for Rehabilitation Exercise Assessment

Jake Van Slyke
University of Oklahoma Polytechnic Institute
AAI 4303: Deep Learning I

*Draft — complete the marked data-exploration items before submission.*

## 1. Problem and Research Question

Physical rehabilitation often involves repeating exercises with proper form, but it can be difficult to assess movements consistently outside a clinical setting. Video-based models could help, although camera angles and imperfect pose estimates make this challenging.

I'm also interested in how similar methods could eventually apply to sports biomechanics, such as basketball shooting mechanics or snowboarding movements. Rehabilitation provides a more controlled starting point with labeled examples of correct and incorrect movement.

**Research question:** How do biomechanical features and temporal modeling affect the accuracy and viewpoint robustness of rehabilitation exercise classification? More specifically, can joint angles and movement patterns improve on raw skeletal coordinates?

The goal is to support research into more accessible movement feedback, potentially benefiting clinicians and patients. The **input** is one video-recorded exercise repetition, represented as a sequence of body joints. The **output** is a **correct/incorrect** prediction. This is a research task, not a clinical diagnosis.

## 2. Dataset

I'll use **REHAB26-ViewAngles**, available at https://doi.org/10.5281/zenodo.23018128 and described by Jánošová et al. (2026). It includes **45 long recordings** of three exercises (arm abduction, squats, and shoulder extension) performed by **five people** using **three synchronized cameras** at 30 FPS. Its annotations identify **1,665 repetitions**: **832 correct (49.97%)** and **833 incorrect (50.03%)**. These are repetition labels, not separate participants or independent videos.

The download is approximately **9.1 GB**, mainly **MP4 videos (8.4 GB)** and **MediaPipe pose sequences (686 MB)**. The poses are NumPy arrays shaped **(frames, 33, 4)**, containing joint coordinates and visibility. CSV files contain repetition timing and labels. The target has **two classes**. The cameras are spaced around 0°, 45°, and 90°, and participants rotate through different orientations.

**Representative examples:** [Insert 2–3 actual frames or skeleton visualizations showing correct and incorrect repetitions from different views.]

The dataset allows **non-commercial research by academic or non-profit organizations**, requires citation of the related paper, and requires permission for commercial use. Its main limitations are the **small number of participants**, controlled recording conditions, and the fact that MediaPipe's 3D joints are estimates rather than motion-capture ground truth. Different views of the same repetition are also correlated, so they must not be treated as independent examples across train/test splits.

## 3. Preliminary Data Exploration

The published labels are almost perfectly balanced, so overall class imbalance should not be a major issue. I still need to check distributions by **exercise, participant, angle, and error type**. The dataset documentation also notes that some skeleton sequences end before their videos, which may create alignment issues.

My initial notebook will check repetition counts and lengths, missing or invalid values, low-visibility joints, video/pose alignment, and a few representative frames. Likely preprocessing includes separating repetitions, sampling frames, normalizing coordinates, and calculating joint angles and movement velocities. These are experimental features, not validated clinical measurements.

**[After running the notebook, insert actual findings: counts, missing/alignment issues, repetition-length range, notable observations, and example figures. Do not claim these checks are complete yet.]**

## 4. Proposed Models and Experiments

For the **baseline**, I'll train a small multilayer perceptron (MLP) on summary features from the provided MediaPipe 3D joints. The **advanced model** will use sequences of biomechanical features, such as joint angles, relative positions, and changes over time, with a **1D CNN or LSTM**. This gives me a manageable project using the existing dataset. If time allows, I'll also try **SAM 3D Body** as an alternative pretrained pose-feature extractor.

The three main experiments are:

1. **Features:** Compare raw joint coordinates, normalized coordinates, and biomechanical features, changing one feature group at a time.
2. **Time:** Compare static or summarized poses against full movement sequences using an MLP and a temporal CNN/LSTM.
3. **Viewpoint:** Measure how classification performance changes across camera angles and participant orientations.

I'll evaluate using **F1, balanced accuracy, precision, recall, and confusion matrices**, plus ROC-AUC if appropriate. With only five participants, I plan to use **leave-one-subject-out evaluation** and keep all views of a repetition in the same split. I'll also examine 5–10 errors to see whether mistakes relate to movement form, camera viewpoint, or pose quality.

For the final demo, I'd like to process a new exercise clip and show its predicted correctness alongside a pose visualization. An optional extension is to apply the feature-extraction pipeline to basketball or snowboarding footage to explore what transfers. Without sports-specific labels, that would only be an exploratory visualization, not a validated assessment of athletic performance. Full multi-camera fusion and fine-tuning SAM 3D Body are outside the initial scope.

## References

Jánošová, M., Budikova, P., & Sedmidubsky, J. (2026). *REHAB26-ViewAngles: A multi-view dataset for studying camera viewpoint in rehabilitation exercises* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23018128

Jánošová, M., Lang, A., Budikova, P., & Sedmidubsky, J. (2026). *Impact of patient orientation in single- and multi-view camera environments for AI-based rehabilitation monitoring* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2609.35726

Meta AI. (n.d.). *SAM 3D Body* [Computer software]. GitHub. https://github.com/facebookresearch/sam-3d-body

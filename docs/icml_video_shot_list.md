# ICML Video Shot List

This is the presentation-facing shot list for Debate2Create videos. It mirrors the goal-state shot list but is suitable for project documentation.

## Required Talk Clips

1. Co-design ablation clip
   - Candidate: Walker2d or Hopper first, because gait differences are easier to read quickly than Swimmer.
   - Layout: 2x2 if all four trained controls exist:
     - default morphology plus default reward
     - learned morphology plus default reward
     - default morphology plus learned reward
     - learned morphology plus learned reward
   - Duration: 10-12 seconds.
   - Purpose: main qualitative evidence for the co-design interaction claim.
   - Required source: trained ablation checkpoints or approved cluster runs.

2. Main comparison clip
   - Candidate: Walker2d or Hopper for a clean left-right comparison; Ant if only D2C vs Bayesian is needed.
   - Layout: 1x2 for strongest baseline vs D2C, or 2x2 if four trained policies are available.
   - Duration: 10-15 seconds.
   - Purpose: show method behavior against baselines under the same camera and timing.
   - Required source: trained checkpoints for the compared XML/reward pairs.

3. Short hero or process clip
   - Candidate: D2C Ant or Walker2d, or a morphology progression montage if round-level designs are available.
   - Duration: 3-6 seconds.
   - Purpose: title-slide hook or brief method-process visual. Cut this if it duplicates the comparison or ablation.
   - Required source: trained D2C policy checkpoint for locomotion, or saved morphology snapshots for a process montage.

4. Failure/Q&A backup clip
   - Candidate: random-action rollout, failed policy, or visibly unstable morphology.
   - Duration: 6-8 seconds.
   - Purpose: answer limitations questions and show that poor designs are filtered, not hidden.
   - Required source: failed run or intentionally labeled random/static rollout.

## Labeling Requirements

- Label method, environment, seed or selection rule, and playback speed.
- If the clip is best-of-N, say so on screen or in speaker notes.
- If the clip is static, random-action, or an external checkpoint smoke test, label that explicitly.
- For ablations, label morphology and reward separately.
- Prefer one continuous rollout per clip. If edited for time, say so.
- Put quantitative metrics on the slide or in a compact overlay; do not rely on video alone as evidence.

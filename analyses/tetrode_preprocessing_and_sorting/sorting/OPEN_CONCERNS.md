---
title: Open methodological concerns to revisit (tetrode MP sorting)
updated: 2026-06-20
status: needs_verification
confirmed_by_user: true
---

# Open concerns to revisit — tetrode matching-pursuit sorting

Recorded 2026-06-20 as the project was set aside. Two domain-expert observations from
the user that should steer the next pass. **Neither is resolved** — they are open
concerns, not conclusions, and they partly tension with the current
`MATCHING_PURSUIT_FINDINGS.md` results they critique.

## 1. Suspected over-merging — within-cluster waveform variability too high for single units

The user does not believe the current clusters are clean single units: **there is too
much waveform variability within each cluster**. The cosine-purity acceptance bar (user
phrasing: "±75 ms cosine purity (or wobble, or whatever) needs to be even higher") is
**too loose** — it must be stricter before a cluster is credible as one unit.

Implication: the production pipeline may be **merging too aggressively** and/or the
purity metric is **too permissive** to catch residual multi-unit contamination. This is
in direct tension with the earlier "dominant defect is OVERSPLIT → merge first" finding
(`MATCHING_PURSUIT_FINDINGS.md`, 2026-06-19): the expert judgment now is that the
pendulum may have swung to over-merging. Concretely, the CCG-guarded merge uses
`propose_cos>=0.90` / `fallback_cos=0.95`, and the production deliverable's tight-window
purity is only 0.876 — the user's read is that this is **not high enough** to call
single-unit.

TODO on return:
- **Pin down the metric/window.** "±75 ms" does not match any current cosine window —
  reconcile it with the code: the tight rA window is sub-ms (`ASYM_MS=(-0.3, 0.8)` ms,
  `TIGHT_SHIFT=2` samples), the full rF window is ~3 ms (`ms_before=1.0`,
  `ms_after=2.0`), CCG maxlag is ±25 ms, refractory ±1.5 ms. Determine whether the user
  means a longer comparison window, a stricter threshold on an existing one, or a
  different metric entirely.
- Raise the purity acceptance bar and/or the merge `propose`/`fallback` cosine
  thresholds; re-test whether the merge is combining genuinely distinct cells.
- **Look directly at within-cluster waveform spread** (per-cluster waveform overlays,
  amplitude×shape distributions) to judge single-unit credibility by eye — not just the
  summary purity scalar, which the user distrusts here.

## 2. MAD-based detection may be miscalibrated on very active channels

Spike detection thresholds are `detect_threshold × noise_level`, where the noise level is
a **MAD estimate** (`get_noise_levels` / `detect_window_peaks` at 5.5 MAD;
`_wobble_eval.py`). On a **very active channel**, the trace fluctuates constantly because
of persistent spiking — so its high variability is **not noise, it is activity**. MAD
then **over-estimates the noise floor** on busy channels, raising the effective detection
threshold there and potentially **missing spikes on exactly the most active channels**.

Implication: the high-MAD event detection that feeds coverage (axis A), residual capture,
and the MUA bucket may be **biased against the busiest channels/tetrodes** — the MAD
"noise" estimate is contaminated by signal. (Note: this also bears on observation 1 — if
the detection floor is unreliable on busy channels, downstream purity/merge judgments on
those tetrodes inherit the bias.)

TODO on return:
- Check whether per-channel MAD noise estimates **correlate with firing rate / activity**
  (they should not, if MAD is a clean noise floor).
- Consider a noise estimator more robust to dense spiking — e.g. an iterative or
  spike-excised noise estimate — before trusting MAD-thresholded detection on active
  channels.

---

See `MATCHING_PURSUIT_FINDINGS.md` for the current pipeline and the metrics these
observations critique. The production deliverable as of parking: `assembled_prod`
(`tetrode_analyses.mp_production_sorting` in the artifact registry).

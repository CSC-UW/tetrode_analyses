---
title: Matching-pursuit (circus-omp) for 48h tetrode unit tracking
scope: tetrode_analyses
status: active
source: measurement
created: 2026-06-12
last_updated: 2026-06-12
confidence: medium
confirmed_by_user: not_required
---

# Matching pursuit for tetrode tracking

Investigating whether a template-matching / deconvolution sorter (SpikeInterface
`sortingcomponents.matching`, the component behind Lupin / SpyKING-CIRCUS) beats the
chunk+match approach for tracking units across a 48 h tetrode recording. Motivation: the
chunk+match bottleneck is per-chunk isolation DROPOUT (~7% of clean units MS5 fails to isolate in
a given chunk -> chain breaks; see `TRACKING_FINDINGS.md` / `project_tetrode_unit_tracking`).
Matching pursuit re-detects a unit from its template without re-clustering each chunk, so a unit
need not vanish when clustering misses it.

Geometry-free by construction: motion correction is a SEPARATE, skip-able `sortingcomponents`
component (DREDge), so simply not applying it makes matching geometry-agnostic. Our tetrode
probegroup (tetrodes 300 um apart, ~tens of um within) makes circus-omp's spatial sparsity isolate
each tetrode. Methods available: `nearest`, `nearest-svd`, `tdc-peeler`, `circus-omp`, `wobble`.

Code: `_mp_common.py` (shared helpers), scripts `51_match_pursuit_poc.py` (stage 0 fidelity +
diagnostics), `52_match_pursuit_dedup.py` (stage 0b dedup). Outputs under
`sortings_seed42_pcafix/track_eval/mp_poc/`.

## Stage 0 — fidelity PoC (3 min drift-stable span, [36000,36180) s)

Setup: MS5 single sort of the span = reference (92 units); build a group-sparse Templates bank from
it; re-detect the whole span with circus-omp; compare to the reference (`compare_two_sorters`, same
frame base). circus-omp runs in ~20 s for 3 min on CPU.

**Verdict: matching pursuit is viable geometry-free on tetrodes.** Label-agnostic detection recall on
well-isolated reference spikes = **0.958** (circus-omp lands on 96% of them). One-to-one unit
agreement only ~0.36 -- but that is NOT a detection failure; it is label-splitting across MS5's
oversplit near-identical templates (~5.75 units/tetrode), plus residual over-detection, scored against
an oversplit reference.

### Load-bearing fixes found in stage 0 (all operational, verified)
- **Units must match.** circus-omp reads the recording in RAW units; the materialized binary carries
  `gain_to_uV=0.195` (raw std ~80 vs uV std ~15.5). Building templates in uV made them ~5x too small
  -> fitted amplitude centered ~3.25 (should be ~1.0), OMP could not subtract the spike and
  re-detected it: 4.4x over-detection, 3x temporal duplication, agreement ~0.13. Fix: build the
  analyzer `return_in_uV=False`, `get_dense_templates_array(return_in_uV=False)`, `Templates
  is_in_uV=False`. After fix: amplitude median ~0.89, over-detection 1.0-1.4x, recall 0.96.
- **Templates with a sparsity_mask need the SPARSE array** (n_units, n_samples, max_active=4), packed
  per unit onto its tetrode channels -- not the dense 64-channel array.
- circus-omp default `amplitudes=[0.6, inf]` over-accepts on 4-ch templates; `amp>=0.8` trims
  over-detection to ~1.4x with no recall loss. `omp_min_sps` had ~no effect here.

### Amplitude-band sweep (raw units, 3 min), well-isolated agreement
| setting | n_spk/ref | well-med agreement |
| --- | --- | --- |
| default [0.6,inf] | 2.2x | 0.26 |
| amp>=0.8 | 1.4x | 0.36 |
| amp 0.9-1.5 | 1.0x | 0.34 |

## Stage 0b — template dedup (3 min span, amp>=0.8)

Merge within-tetrode near-identical units (template cosine >= thr) before matching, then compare vs
the deduped reference. Well-isolated one-to-one agreement:

| setting | units | well-med agreement | recall |
| --- | --- | --- | --- |
| nodedup | 92 | 0.352 | 0.899 |
| dedup cos0.95 | 68 | 0.422 | 0.899 |
| dedup cos0.9 | 43 | 0.420 | 0.883 |
| dedup cos0.85 | 30 | 0.381 | 0.871 |

Dedup helps MODESTLY (0.35 -> 0.42, best cos 0.9-0.95) but PLATEAUS far below the 0.90 detection-recall
ceiling; over-merging (cos 0.85, 30 units) hurts. So MS5 oversplit was only part of the gap. With
recall ~0.90 but agreement ~0.42 and over-detection ~1.4x, the residual is **over-detection +
cross-unit label confusion**: on 4-channel templates, distinct-but-similar units are confusable, so
matching-pursuit *assignment* is intrinsically noisier than on dense probes.

**Reframing:** assignment fidelity vs MS5 is the wrong yardstick for tracking. Matching pursuit's
value is DETECTION continuity (high recall, no chunk boundaries). The decisive test is whether a
confident unit's template yields a clean, continuous train across drift (stage 1/2), not whether it
reproduces MS5's exact labels.

## Stage 1/2 — carry-forward matching (mechanics validated)

`_mp_common.windowed_carry_forward`: detect a fixed unit set window-by-window with circus-omp; each
window re-derive PRESENT units' templates (track drift) and KEEP prior templates for ABSENT units
(so a unit that drops out one window is still sought next -- the dropout-recovery mechanism). Two
modes: `fixed` (initial templates throughout) vs `reestimate`.

- **Smoke (30 min drift-stable, 3x600 s, script 53):** fixed AND reestimate both give 100% continuity
  (all 130 confident units present every window) and CLEAN trains (ISI<1 ms median 0.0006-0.0010).
  Confirms the assembled trains are real refractory-respecting single units, not noise.
- **1 h validation (4x900 s, seed bank from window-0 MS5 sort, script 54):** fixed continuity 1.00,
  flat trend [1,1,1,1], ISI<1 ms 0.0013. The seed-from-start + carry-forward path is clean.
- **10 h run ([56000,92000) s, 20x1800 s, seed from first 30 min, script 54):**
  | mode | continuity median / frac_full(>=0.95) | ISI<1 ms median | per-window trend |
  | --- | --- | --- | --- |
  | fixed | 1.00 / 0.97 | 0.0018 | 1.0 -> ~0.98, flat |
  | reestimate | 1.00 / 0.96 | 0.0006 | 1.0 -> ~0.97, flat |

  RESULT: matching pursuit gives **~97% continuous tracks over 10 h** for confident units (vs
  chunk+match fragmenting at ~2 h median) -- a real continuity win -- and the assembled trains are
  clean single units (re-estimation 3x cleaner: ISI<1 ms 0.06% vs 0.18%). BUT continuity is tied
  fixed-vs-reestimate and both trends are ~flat -> this pre-discontinuity span is LOW-DRIFT (fixed
  templates stay valid 10 h), so re-estimation's drift-*tracking* benefit is untested here (only its
  cleanliness benefit shows). Need a higher-drift / discontinuity-crossing span to stress it.

- **Discontinuity-crossing run ([92000,110000) s, 10x1800 s, weld ~100858 s ≈ window 4):**
  | mode | frac_full(>=0.95) | ISI<1 ms | per-window trend |
  | --- | --- | --- | --- |
  | fixed | 0.97 | 0.0017 | [1,1,1,1,1,1,1,0.99,0.98,0.97] |
  | reestimate | 0.98 | 0.0006 | [1,1,1,1,1,1,1,1,1,0.98] |

  RESULT: the ~100858 s weld that broke the one-shot MS5 sort is a NON-ISSUE for matching pursuit --
  continuity stays ~1.0 straight through the weld (windows 4-6) under both modes. The one-shot sort
  failed there because its GLOBAL whitening/classifier was disrupted by the in-band noise step;
  per-unit template matching is immune (it bridges via each unit's own template). Reestimate edges
  fixed at the tail (1.0 vs 0.98) and is 3x cleaner -- the first sign of re-estimation's value as
  drift accumulates.

- **Full 48 h deliverable run ([2000,172000) s, 95x1800 s, reestimate-only, seed from first 30 min,
  109 confident units):** the headline result.
  - **103/109 units tracked across >=90% of the 47 h; 94/109 present in ALL 95 windows;** median unit
    present every window. Continuity median 1.00, frac_full(>=0.95)=0.94, ISI<1 ms median 0.0006.
  - **NOT false continuity (checked):** units present late are robustly active -- median ~4500
    spikes/30 min (~2.5 Hz), 103/109 with late counts >=100 spk (far above the 20-spk "present" floor),
    late/early rate ratio 0.86 (gentle biological decline). Present-fraction trend decays GRADUALLY
    1.0 -> 0.91 (>=20 spk) / 0.80 (>=100 spk), not suspiciously flat -> real loss of some units, not a
    template hallucinating on noise.
  - **Clean single units throughout:** 100% of the 109 units have <1% ISI<1 ms violations over the
    full 47 h (median 0.06%, p90 0.30%).
  - Deliverable saved: `mp_long_s2000_d170000/assembled_reestimate` (NumpySorting) + `long_drift.npz`.
  - **Identity-stability check (script 55, 5 template samples at 0.2/11.9/23.6/35.3/47.0 h):** of the
    100/109 units present at all 5 points, MIN consecutive-template cosine (across ~11.7 h gaps)
    median 0.853, 85% >=0.7, 64% >=0.8. Since waveforms legitimately drift over 11.7 h, this says MOST
    tracks hold a stable identity (smooth drift, no swap). **15/100 have a low-cosine (<0.7) segment ->
    suspect for large-drift OR a swap (coarse 11.7 h sampling can't distinguish; needs per-window
    follow-up).** So qualify the headline: ~95% continuity, ~85% verified identity-stable on this
    coarse check, ~15% needing finer verification.

- **Fixed-vs-reestimate over the full 48 h (same span/seed, bonus comparison):**
  | mode | frac_full(>=0.95) | tracked >=90% windows | trend 0->47 h | ISI<1 ms |
  | --- | --- | --- | --- | --- |
  | fixed | 0.93 | 0.94 | 1.0 -> 0.91 | 0.0032 |
  | reestimate | 0.94 | 0.94 | 1.0 -> 0.91 | 0.0006 |

  CONTINUITY is essentially IDENTICAL fixed-vs-reestimate over 48 h (both ~94% tracked, both decay to
  0.91) -> the units' waveforms drift mildly enough that even the t=0 template still detects them 47 h
  later. Re-estimation's only edge is CLEANLINESS (5x fewer refractory violations: 0.06% vs 0.32%) and
  faster convergence. REFINES the earlier hypothesis: re-estimation is NOT needed for continuity (drift
  is mild here), only for train PURITY. (Fixed ran slower because t=0 templates fit drifted units worse
  -> more OMP residual iterations + more spurious spikes, consistent with its higher ISI.)

## Conclusion (measurement-based; scientific framing pending user review)

Carry-forward matching pursuit (circus-omp, geometry-free, motion-correction off, per-window template
re-estimation) **tracks ~95% of confident well-isolated units continuously across the full 48 h** as
clean single units -- the whole-recording identity that chunk+match could not deliver (it fragmented
at ~2 h median; well-isolated units ~15 h). It bridges the ~100858 s weld that broke the one-shot sort.
This realizes the two-pass design: confident/curated units -> seed template bank -> deconvolve forward.

CAVEATS (honest):
1. **Assignment confusion on 4-channel templates** (agreement vs MS5 ~0.42): a track is clean +
   continuous, but a mid-recording IDENTITY SWAP to a template-similar neighbor would also look clean
   + continuous and is not excluded by refractory/rate checks. Lower risk for distinct-template units;
   needs spot verification (e.g. template-similarity trajectory, or manual curation of a sample).
2. **Seed-at-start only:** the bank is fixed from window 0, so units APPEARING after the first 30 min
   are not tracked. A complete sorter would re-seed periodically (detect+cluster residual, add new
   templates).
3. Tested on one span/seed; not yet validated against ground truth or manual curation; "confident" =
   snr>=5 & >=100 spk in window 0 (marginal units not tracked).

## Interim conclusion (pending full-48h result + user review)

Matching pursuit (circus-omp) with carry-forward IS a viable geometry-free tracker for these tetrodes:
high detection recall (0.96), ~97% continuity over 10 h with clean single-unit trains, and it bridges
the discontinuity that defeated the one-shot sort. Its weakness is unit-ASSIGNMENT fidelity on
4-channel templates (similar units confusable; agreement vs MS5 ~0.42), so it is best used to TRACK a
confident, well-isolated unit's template forward in time (continuity), not to re-derive an exact
clustering. This is exactly the two-pass design: MS5 (or curated) confident units -> seed template
bank -> carry-forward deconvolution over 48 h.

## Inspecting the results (analyzer + figures)

- **Curation analyzer** (`56_build_carry_forward_analyzer.py`): `mp_long_s2000_d170000/analyzer_tracks.zarr`
  -- geometry-free, group-sparse SortingAnalyzer on the 109 tracked units (waveforms, templates, PCA,
  correlograms, ISI, template_similarity, quality_metrics; SNR median 7.78). Persisted sortable unit
  columns: `track_hours` (median 47.5 h), `n_windows`, `identity_min_cos` (<0.7 = suspect, 15 units).
  Open it with `launch_curation.py --analyzer-path .../analyzer_tracks.zarr --style grahams_curation`.
- **Figures** (`57_plot_carry_forward.py`): `mp_long_s2000_d170000/figures/`
  -- `continuity_trend.png` (fixed vs reestimate present-fraction over 48 h, weld marked),
  `track_span_hist.png` (median span 48 h), `rate_heatmap.png` (per-unit firing rate over 48 h),
  `template_evolution.png` (peak-channel template at 8 time points, stable vs suspect -- stable units
  tightly overlaid over 48 h; suspect units mostly low-SNR/noisy, a few genuinely changed).

## Seed-bank dedup on the 48 h carry-forward (0.9 vs 0.95, reestimate)

Re-seed the 48 h carry-forward from the confident window-0 bank after collapsing within-tetrode
near-identical seed units (shift-tolerant cosine >= thr, `_mp_common.dedup_sorting`: +/-10-sample
shift-max, 0.3 ms coincidence merge), to strip MS5 oversplit before tracking. Over-merge is detected
by refractory contamination (fusing two distinct neurons injects <1 ms ISI violations); continuity and
identity-min-cos guard against losing real tracks. Full-48 h reestimate pass, same span/window/amplitude
band as the headline run:

| run | seed units | ISI<1 ms median | p90 | units >1% | units >2% | identity median min-cos | suspect (<0.7) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| non-dedup | 109 | 0.0006 | 0.0030 | 0 (0%) | 0 | 0.853 | 15% |
| dedup 0.95 | 75 | 0.0011 | 0.0044 | 0 (0%) | 0 | 0.875 | 13% |
| dedup 0.90 | 36 | 0.0034 | 0.0222 | 7 (19%) | 5 (14%) | 0.904 | 15% |

Continuity is excellent for all three (median per-unit present-fraction 0.94-1.00, frac_full >= 0.94).
The over-merge alarm is decisive: **dedup 0.95 collapses 109 -> 75 (removes ~31% oversplit redundancy)
with ZERO refractory contamination -- identical to non-dedup on the >1% alarm -- while dedup 0.90
(109 -> 36) trips it hard (19% of merged units contaminated, ISI<1 ms median 6x higher).** So the safe
within-tetrode dedup boundary sits between 0.90 and 0.95: at 0.95 the merged pairs are genuine oversplit
twins; at 0.90 the shift-tolerant cosine fuses distinct-but-similar neurons. (NB the +/-10-sample
shift-tolerant cosine merges more aggressively than the analyzer's `template_similarity`, which is
fixed-align `max_lag_ms=0`; this is why a "0.9" here is stricter-looking than a 0.9 there.)
RECOMMENDATION (measurement-backed, pending user adoption): use 0.95 as the seed-bank dedup threshold,
or the contamination-guarded per-pair merge (propose by similarity, accept only if the merged unit's
refractory stays clean) for a principled stop. Artifacts built for all three variants:
`analyzer_tracks{,_dedup09,_dedup095}.zarr` + `identity_check{,_dedup09,_dedup095}.npz` +
`figures{,_dedup09,_dedup095}/`.

## Open / next
- Long-drift result (above) -> if reestimate >> fixed continuity, matching pursuit tracks drift
  geometry-free; then build the two-pass deliverable (confident bank -> carry-forward over 48 h).
- Alternatives if assignment confusion limits: `wobble` matcher; tighter amplitude band; per-tetrode
  template-set dedup before matching.
- Cross-cutting: MS5 oversplit reduction (shared bottleneck for chunk+match AND matching pursuit).

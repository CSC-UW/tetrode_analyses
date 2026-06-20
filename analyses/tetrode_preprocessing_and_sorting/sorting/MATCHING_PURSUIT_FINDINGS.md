---
title: Matching-pursuit (circus-omp) for 48h tetrode unit tracking
scope: tetrode_analyses
status: active
source: measurement
created: 2026-06-12
last_updated: 2026-06-20
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

## 2026-06-19 — Assignment-purity re-score (corrects the "Pareto-dominates" reading)

A reorientation review found this document's headline metrics -- median `rp_contamination`, >=N MAD
coverage, low-amplitude retention -- are all PRECISION/RECALL proxies measured at the TETRODE or
whole-sorting level; NONE is per-unit ASSIGNMENT PURITY (are a unit's spikes assigned to the CORRECT
same-tetrode unit, vs merely landing on the right tetrode?). New machinery (`_assignment_eval.py`,
`_scoreboard.py`; scripts 86-89) scores the deliverables on three SEPARATE axes -- (A) event coverage,
(B) per-unit assignment purity, (C) identity stability. Measured (3 windows, 5/26/40 h; full re-score in
`axis_bc_rescore.json` / `scoreboard_unified.json`):

| variant | units | cov>=10 | purity (full rF) | purity (tight rA) | within-tetrode CCG-duplicate pairs |
| --- | --- | --- | --- | --- | --- |
| reestimate | 109 | 78.8% | 0.976 | 0.788 | 110 |
| reseed_c12 | 98 | 80.3% | 0.974 | 0.817 | 72 |
| dedup095 | 75 | 78.8% | 0.977 | 0.832 | 34 |

Two corrections follow:

1. **The "Pareto-dominates on all three axes" claim (gate bake-off, below) is on the
   rp/coverage/low-amp frontier ONLY** -- it does NOT establish assignment correctness. Full-window
   cosine purity is ~0.97 (about what median-rp + pooled coverage rewarded), but the discriminative
   tight-trough purity is only 0.79-0.83, and the base sorting carries 110 within-tetrode CCG-duplicate
   pairs (one cell split across un-merged tracks). Merging (dedup095) improves tight purity AND parsimony
   at zero coverage cost.
2. **The reassignment correction (script 85, below) is the HEADLINE, not a footnote**: the full-window
   cosine MIS-ASSIGNS a substantial fraction of spikes to a same-tetrode NEIGHBOUR (54% pooled, 72-80% on
   crowded tetrodes), which `rp_contamination` cannot see (an independent neighbour does not violate the
   host's refractory). The dominant defect is OVERSPLIT, an axis-C MERGE problem -- competitive
   reassignment on the oversplit sorting is mostly churn (script 88: only 7.5-15.7% of moves are to a
   genuinely DISTINCT neighbour; the rest shuffle between oversplit twins). Operating rule: **MERGE
   duplicate tracks first, THEN assign.**

Residual-capture E2 (script 87) is now MEASURED, not hypothetical: ~33% of >=10 MAD unclaimed events are
cleanly recoverable (cosine>=0.8 + refractory; recovered-vs-host CCG = 49 own-dropout / 42 abstain / 0
independent-contaminant), and ~67% are no-template collisions/MUA -> the Part C MUA bucket, NOT discarded.
See `_assignment_eval.py` + `docs/plans/` reorientation for the three-axis framework.

## 2026-06-20 — Production pipeline (the three-axis deliverable)

`97_production_pipeline.py` assembles the final deliverable from `assembled_reseed_c12` in four
checkpointed stages -- CCG-guarded merge -> residual SUA capture -> per-tetrode MUA bucket -> A/B/C
scoring -- over all 95 windows (47.2 h), in 3.21 h. **No blanket competitive reassignment** (script 88
showed it is mostly oversplit churn on the un-merged sorting; the merge already resolves the duplicates it
would shuffle).

| variant | units | covA>=10 | covA>=12 | purity full | purity tight | CCG-dup pairs |
| --- | --- | --- | --- | --- | --- | --- |
| base (`reseed_c12`) | 98 | 80.3% | 87.2% | 0.972 | 0.823 | 72 |
| `prod_sua` (merge + residual) | 71 | 86.1% | 91.7% | 0.960 | **0.876** | **23** |
| `prod_sua+mua` (`assembled_prod`) | 87 | **96.9%** | **98.3%** | -- | -- | -- |

**All three axes improve at once.** (B) tight-window purity 0.823->0.876 -- the discriminative metric
`rp`/pooled-coverage were blind to; (C) CCG-duplicate pairs 72->23 from the merge; (A) >=10-MAD coverage
80.3->86.1% from residual capture, reaching 96.9% once the MUA bucket claims the neural-but-unsortable
remainder. Full-window purity dips slightly (0.972->0.960) because residual capture *adds* genuinely harder
spikes (each unit's noisier dropout) to the units -- expected, and the tight axis (the one that
discriminates correct assignment) is what improves.

Residual capture recovered **29.4%** of the 6.79 M unclaimed >=10-MAD events (1.99 M spikes; only 1,558
refractory-rejected), with a clean recovered-vs-host CCG verdict (**56 own-dropout / 15 abstain / 0
independent-contaminant**) -- i.e. recovered spikes are the units' own dropout, not co-located
cross-contamination. The MUA pass classified 29 M unclaimed >=7-MAD events as 66% MUA / 21% noise (dropped)
/ 13% SUA-recoverable, pooled into 16 per-tetrode `is_mua=True` pseudo-units that preserve spatial
localization without polluting the SUA set.

**Deliverable:** `assembled_prod` (87 units = 71 SUA `is_mua=False` + 16 MUA `is_mua=True`); the SUA-only
purity/identity deliverable is `assembled_prod_sua`, and `assembled_prod_merge` is the pre-residual merged
base. Curation analyzers: `analyzer_prod.zarr` / `analyzer_prod_sua.zarr` / `analyzer_prod_merge.zarr`
(script 98 -- group-sparse, geometry-free, `is_mua` carried through). Note: axis C is not reported for the
`+mua` variant -- a pooled MUA unit is CCG-"duplicate" against every SUA unit on its tetrode by
construction, so the metric is meaningless there; the deliverable's identity number is `prod_sua`'s 23.
Registered as `tetrode_analyses.mp_production_sorting` in `docs/artifacts/registry.yaml`.

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

## Identity swaps and periodic re-seeding (scripts 55-58)

The seed-at-start design has two coupled weaknesses, both addressed by periodic re-seeding.

**Identity swap (rare, ~3-6%).** A track seeded on a low-rate neuron can be CAPTURED by a louder
same-tetrode neighbour as that neighbour's firing ramps up: per-window re-estimation gradually walks
the template from the seeded cell onto the loud one. Worked example (dedup-0.95 track u20, tetrode 5):
its early shape (peak ch3) and late shape (peak ch0/1) are ORTHOGONAL (shift-cos -0.004); it follows
the early neuron for ~10 h then walks onto the late one over 18-24 h, and the early neuron survives as
a SEPARATE track (cos 0.93 to it) -- proving a swap, not drift. The swap is SEQUENTIAL (the two cells
never co-fire) so it produces NO refractory violations and passes every standard QC metric; only the
`identity_min_cos` template-stability check flags it. PREVALENCE (early[1,13]h vs late[35,46]h
self-cosine over 72 evaluable tracks): median 0.90, only 2/72 confirmed swaps (self-cos<0.7 + a
surviving same-tetrode twin>=0.8), +2 ambiguous; ~94% stable.

**Step-cap does NOT fix it (measured negative result).** A per-window re-estimation step-cap
(`reestimate_min_cos`, reject a window's update if 4-ch shift-cos to the running template < thr) is a
no-op: at 0.8 it rejected 0/95 windows. The swap is a GRADUAL multi-window walk -- u20's per-window
consecutive cosine (min 0.962, median 0.996) is indistinguishable from stable tracks (min 0.971-0.987,
median 0.997). A cap tight enough to catch the walk (~0.99) fires constantly on benign re-estimation
noise. Per-window step size cannot separate slow-swap from slow-drift; only template COMPETITION can.

**Periodic re-seeding (`windowed_carry_forward_reseed`, script 58).** Every `reseed_every_windows`
windows (6 h here), re-sort that window with MS5 and ADD any confident cluster (snr>=5, >=100 spikes)
whose 4-channel template doesn't match an existing same-tetrode bank unit (shift-cos < 0.8) as a NEW
tracked unit. RESULT (full 48 h, dedup-0.95 seed): **73 seed -> 95 tracked units (+22, ~30%)**; 21/22
new units clean (ISI<1 ms < 1%), most major (1e4-2e6 spikes); continuity median 1.00, ISI<1 ms median
0.0009 (vs 0.0011 seed-only), 2/95 units >1% refractory. Post-reseed swap check: median self-cos 0.907,
NO catastrophic orthogonal capture (worst 0.58); 5 mild residual suspects (0.58-0.69 =
template-similar-neighbour assignment confusion, which re-seeding does not target). One curation flag:
new u89 (tetrode 10, ~30 Hz, ISI 2.1%) is a likely merge.

New units appear predominantly in the **18-30 h window** (+2 caught at the 24 h reseed, +10 at 30 h),
which BRACKETS the sleep-deprivation period (~24 h start, ~29 h end), a high-movement epoch (user,
2026-06-16). Whether movement makes new units detectable is plausible but UNCONFIRMED -- it is NOT the
27.7 h weld, and the user is not confident attributing the births even to the SD boundaries. The
operational point stands regardless: ~23% of this recording's units appear after the seed window and
are lost without re-seeding, so periodic re-seeding is the path to a COMPLETE sorter here.
(`build_templates_object` now SEEDS random_spikes, so the dedup'd seed is reproducible -- it had
jittered 73-77 run-to-run.)

**Re-seeding does NOT prevent the swap (same-seed A/B, script 59).** Holding ONE deterministic seed and
running no-reseed (arm A) vs reseed (arm B) over 60 windows (the only difference re-seeding on/off):
seed 109->76; arm A median self-cos 0.956 (2 units <0.7, 1 <0.5), arm B median 0.959 (3 <0.7, 1 <0.5,
+4 reseeded units). The catastrophic orthogonal capture (u21, self-cos 0.033) persists in BOTH arms
(B=0.049, not rescued). So the earlier reseed run's absence of a catastrophic swap was SEED LUCK, not
re-seeding -- the confound the A/B was built to expose. Likely cause: by the reseed window the capturing
track has already drifted partway onto the neighbour, so the neighbour's cluster matches it (cos>=0.8)
and is NOT added as a competing track. CORRECTED CONCLUSION: re-seeding's validated value is the
NEW-UNIT YIELD only (+22 units/48 h); it is swap-NEUTRAL. No method tried (step-cap, re-seeding)
prevents the identity swap -- it remains a curation item flagged by `identity_min_cos`.

Deliverable: `assembled_reseed_rs` + `reseed_rs.npz`; curation analyzer `analyzer_tracks_rs.zarr`.
Diagnostics: `/tmp/u20_{diag,split,confirm,stepsize}.py`,
`/tmp/{swap_prevalence,swap_compare,reseed_validate}.py`; A/B = script 59 + `ab_reseed_swap.npz`.

## Re-seed cadence sweep + add-cos gate probe + temporal adjudication (scripts 61-63)

Re-seeding's only validated value is new-unit yield (it is swap-neutral, above), so the open question
was the cadence: does a finer interval pick up more late neurons, or just more duplicates? Script 61
sweeps `reseed_every_windows` ∈ {12, 6, 3} (= 6 h / 3 h / 1.5 h at 1800 s) sharing ONE seed bank across
arms (same-seed, so cadence is the only variable), and for every re-seeded unit records `birth_cos` (its
max 4-ch template cosine to the LIVE same-tetrode bank AT BIRTH -- the exact quantity the add-cos=0.8
gate thresholds on) plus a post-hoc full-span twin check (`final_cos` ≥ 0.9 to a pre-existing
same-tetrode unit). The cadence-12 arm reproduces the known +22 (validation).

| every | hours | born | confident twin (final≥.95) | ambiguous (.90-.95) | distinct-ish (<.90) | birth_cos med [min,max] |
| ----: | ----: | ---: | -------------------------: | ------------------: | ------------------: | ----------------------: |
| 12 | 6.0 | 22 | 3 | 7 | 12 | 0.76 [0.45, 0.80] |
| 6 | 3.0 | 27 | 4 | 12 | 11 | 0.77 [0.44, 0.80] |
| 3 | 1.5 | 40 | 6 | 19 | 15 | 0.78 [0.45, 0.80] |

**Whether finer cadence adds neurons or twins is UNRESOLVED by cosine (user calibration, 2026-06-17).**
An earlier draft of this section claimed "finer cadence adds duplicates, not neurons" using a `final_cos`
≥ 0.9 twin bar; that was an artifact of the threshold. Reading `cosine_realpairs.png`, the user calibrates
the same-neuron line at ≥ 0.95 (cos 0.8 = plausibly different cells, 0.9 = doubtful), which the
independent dedup evidence corroborates (a 0.9 within-tetrode merge trips refractory contamination in
~19% of merges = conflates distinct cells; 0.95 was the safe sweet spot, above). At that confident bar
only ~15% of re-seeds are duplicates (3/4/6 of 22/27/40), NOT the ~half a 0.9 bar implies; the
distribution splits **15% confident-twin / 43% AMBIGUOUS (0.90-0.95) / 43% distinct-ish (<0.90)**. The
genuine-yield-vs-cadence answer lives in that ambiguous band and FLIPS with the threshold:
present+clean+non-twin counts are 7/5/8 (flat) at the 0.9 bar but 11/13/20 (growing) at 0.95. Cosine
cannot classify that band -- and the born-vs-pre-existing comparison drags true twins' cosine DOWN
(late-only born template vs drift-averaged full-span neighbour), so `final_cos` ≥ 0.95 is itself a
floor, not a point estimate, on the twin count. WHAT IS ROBUST: a finer cadence adds more units of
uncertain status (the ambiguous + confident-twin total grows 10 → 16 → 25), i.e. more curation burden
for unknown real gain; and the decisive test is temporal (combined-train refractory / CCG), not template
cosine at any cutoff.

**RESOLVED -- temporal adjudication (script 63, `reseed_twin_adjudication.npz`, `ccg_adjudication_examples.png`).**
For each re-seeded unit, take its best same-tetrode COSINE match among pre-existing units and score the
cross-correlogram central(±1.5 ms)/flank(5-25 ms) ratio over the windows where BOTH are active: two MP
tracks of the SAME neuron never co-fire within the refractory period (deconvolution assigns each spike once)
-> a DIP at zero lag (ratio < 0.30); two distinct cells fire independently -> a flat/filled CCG (ratio > 0.70).

| every | h | born | duplicate (dip) | distinct (filled) | ambiguous | segregated |
| ----: | --: | ---: | --------------: | ----------------: | --------: | ---------: |
| 12 | 6.0 | 22 | 10 | 3 | 9 | 0 |
| 6 | 3.0 | 27 | 12 | 4 | 11 | 0 |
| 3 | 1.5 | 40 | 15 | 5 | 18 | 2 |

So ~45-50% of re-seeds are CONFIRMED duplicates (deep refractory dip), only ~13% CONFIRMED distinct neurons,
~40% ambiguous (partial dip). Two consequences. (1) CADENCE is settled on temporal grounds: the
confirmed-distinct yield barely moves (3 -> 4 -> 5) while duplicates+ambiguous balloon (19 -> 23 -> 33) -- a
finer cadence buys curation burden, not neurons. DON'T go finer. (2) The temporal arbiter and template cosine
DISAGREE per unit: cos-0.80 pairs are confirmed duplicates (u88->u77, u96->u77) and several cos>=0.95 pairs
are NOT (verdict distinct/ambiguous); `ccg_adjudication_examples.png` shows cos 0.89/0.92 = deep dip (same
neuron) vs cos 0.82/0.84 = filled (distinct). So the aggregate ~half-twin fraction the 0.9-cosine bar implied
was about RIGHT in magnitude, but cosine cannot RANK-ORDER the verdict -- the refractory test is the arbiter,
exactly as the user's "cosine is a weak same-unit test" predicted. CAVEAT: OMP greedy subtraction can suppress
co-assignment of SIMILAR templates, so a dip between high-cosine pairs may be part deconvolution-artifact, not
pure biology -- but those pairs should merge either way (same neuron or unresolvable duplicate); the
low-cosine (0.80) dips are genuine refractoriness. **Re-seeding's real new-neuron yield is therefore only
~3-5 over 47 h, NOT the +22 headline:** of the 22 reseeds on the 6 h deliverable ~10 are duplicates to MERGE
(transitive, collapsing onto hubs -- u79,u88,u89,u97 -> u77 -> u58; u76,u78 -> u54), 3 distinct to KEEP, 9 to
inspect.

**Gate probe (`gate_probe_birthcos.png`): the gate is upside-down, not a clean lever.** `birth_cos` is
the real live-bank value (not a proxy). The admitted population piles into [0.72, 0.80) right against the
gate (84/89 pooled). Only 5/89 are distinct-from-bank at birth (`birth_cos` < 0.6, min 0.44) -- and ALL
5 are full-span TWINS (`final_cos` 0.93-0.99): such a unit looks momentarily distinct only because the
re-estimated bank template has drifted off the neuron it split from, while over the full span its average
template matches that neighbour (the right-panel scatter shows the distinct-at-birth points sitting at
the TOP of the twin axis). So tightening the gate toward 0.6 would admit ONLY those distinct-at-birth
candidates -- i.e. exactly the drift-split twins -- and reject the borderline bulk. No add-cos threshold
cleanly isolates real new neurons: the distinct-at-birth candidates are duplicates and the rest are
borderline. The gate is not the lever; the bank already covers the separable units, and what re-seeding
finds is overwhelmingly near-duplicates of tracked cells.

**Why cosine is a weak distinctness test on a tetrode (`cosine_realpairs.png`, `cosine_transform_demo.png`).**
On 4 co-located wires, distinct neurons share gross biphasic morphology, and the 4-ch cosine is
amplitude-BLIND (scale-invariant). Real within-tetrode pairs from this recording sit at 0.6-0.8 while
looking visibly different; a single real template rescaled globally holds cosine = 1.00, and it takes a
large single-channel re-weighting to drag cosine down to 0.6. So `birth_cos` ~0.77 does not by itself
prove a born unit is a duplicate -- but it does mean re-seeding never finds a clearly-distinct waveform,
and `final_cos` ≥ 0.9 (a much stronger bar, given the morphology floor) confirms ~half are duplicates.

**Conclusion (operational, RESOLVED by script 63).** Do NOT adopt a finer cadence: the temporal test
shows the confirmed-distinct yield is flat (3/4/5) while duplicates+ambiguous balloon (19/23/33). A longer
seed window is separately not promising (cuts against the short-window stationarity that makes
carry-forward work; re-seeding already backfills late units). Re-seeding's genuine new-neuron yield is only
~3-5 over 47 h -- far below the +22 headline -- because ~45-50% of reseeds are confirmed duplicates and ~40%
ambiguous. ACTION on the 6 h `_rs` deliverable: run the contamination-guarded merge (cosine to PROPOSE, CCG
refractory dip to ACCEPT) to fold the ~10 duplicate `is_reseeded` units into their parents (transitive:
u79,u88,u89,u97 -> u77 -> u58; u76,u78 -> u54; plus u82->u14, u80->u3, u91->u44), keep the 3
confirmed-distinct, curate the 9 ambiguous. The PROPOSE-by-cosine / ACCEPT-by-CCG merge is the right tool
for ACROSS-TIME merges (reseed-duplicate folding): the cosine-vs-CCG per-unit disagreement is a DRIFT
phenomenon -- over 47 h a true duplicate's full-span cosine is dragged down (cos-0.80 duplicates) while
distinct cells stay high, so cosine mis-ranks. The WITHIN-WINDOW seed dedup is a DIFFERENT regime: the
seed window is ~stationary, so true oversplit twins there are HIGH-cosine and cosine is a fine
discriminator -- its `dedup_sorting` default is now 0.95 (the validated bar; every script already passed
0.95, so deliverables are unchanged -- this only removes the 0.9-default footgun). Outputs:
`reseed_cadence_sweep.npz`, `reseed_twin_adjudication.npz`, `assembled_reseed_c{12,6,3}`, and the figures
(`gate_probe_birthcos`, `cosine_realpairs`, `cosine_transform_demo`, `ccg_adjudication_examples`), all under
`mp_long_s2000_d170000/`.

## Spike coverage / detection completeness (scripts 64-67)

Question (user, 2026-06-17): what fraction of the recording's spikes do our units capture? Method: detect
threshold-crossing EVENTS (locally-exclusive negative peaks, per tetrode/probe) = the spike-event reference;
a sorted spike "claims" an event if within +/-0.5 ms (and, on NP, within ~75 um depth). The decisive cut is
AMPLITUDE: unclaimed near-threshold (~5-7 MAD) events are multi-unit hash/noise no isolation sorter claims
(expected); unclaimed LARGE events are the real false negatives.

TETRODE (scripts 64/66; 47 h, 16 tetrodes, 141.8 M events >=5.5 MAD; IDENTICAL events for both sortings):

| amplitude (MAD) | MP-reseed (95 u) | chunk+match (2204 u) |
| --------------: | ---------------: | -------------------: |
| 5.5-7  | 18.6% | 20.4% |
| 9-12   | 64.0% | 93.3% |
| 12-16  | 82.9% | 99.5% |
| 16-24  | 92.7% | 99.9% |
| >24    | 95.5% | 99.9% |
| overall| 42.2% | 55.0% |

The 42%/55% "overall" is dominated by the near-threshold MUA floor (44% of all events are 5.5-7 MAD, where
even the 2204-unit chunk+match leaves ~80% unclaimed = genuine hash/noise) -- so high-amplitude coverage is
the meaningful number. MP leaves a real gap at large amplitude (17% unclaimed at 12-16 MAD); chunk+match (no
t=0 bank; re-clusters every chunk) captures ~all (99.5%+).

CAUSE of the MP large-spike gap (script 65; MS5-resort at 5/26/40 h, classify each clean MS5 unit): NOT
missing units -- 0 of 68-88 well-isolated MS5 units per window are absent from the bank. The unclaimed-large
events decompose into ~50-75% within-unit DROPOUT of BANKED units (the event sits inside a clean MS5 unit the
bank already has -- circus-omp just didn't place that spike: collisions, amplitude excursions outside the
[0.8,inf] accept band) and ~25-48% NON-isolable overlap/MUA (large threshold crossings a fresh per-window
MS5 sort doesn't resolve into any clean unit). So the false negatives are detection DROPOUT + intrinsic
collisions, NOT a seed-at-start bank-completeness problem -- this CORRECTS the earlier "missing-from-bank"
inference in the gate-probe section above.

NEUROPIXELS reference (script 67; Kilosort 2.5 sortings, novel_objects_deprivation/full; the pipeline's
saved `peaks.npy` as the event reference, detect_threshold 5 MAD; loaded via SortingAnalyzer (Santiago) or
WNE `get_kilosort_extractor` + `unit_locations` (Doppio/Charles)):

| recording | units | events | overall | 9-12 | 12-16 | 16-24 | >24 |
| --------- | ----: | -----: | ------: | ---: | ----: | ----: | --: |
| Santiago imec0 | 484 | 245 M | 63.9% | 84.9% | 91.0% | 92.7% | 97.7% |
| Doppio imec1   | 855 | 683 M | 42.9% | 76.8% | 96.6% | 98.7% | 99.6% |
| Charles imec0  | 553 | 206 M | 73.2% | 96.0% | 96.3% | 97.0% | 98.4% |

The "overall %" is NOT comparable across recordings -- it tracks the detector's low-amplitude tail (Doppio
detected 683 M events, mostly hash -> 42.9% overall yet 96-99.6% of LARGE events claimed). The comparable
metric is high-amplitude coverage, uniformly HIGH (>=12 MAD: 91-99.6%).

SYNTHESIS: the "striking false negatives" are a GENERAL property of spike sorting, not a tetrode-pipeline
flaw. Every production NP KS sort here leaves ~1-9% of large (>=12 MAD) spikes and 30-90% of near-threshold
events unclaimed. The tetrode MP (83-95% large) sits just below the NP KS sortings (91-99.6%); the tetrode
chunk+match (99.5%+) matches the best of them. The unclaimed remainder is dominated by sub-threshold MUA
(intrinsic) plus a minority of large overlap/dropout events -- NOT a population of cleanly-isolable units any
sorter is missing. Outputs: `spike_coverage{,_chunkmatch}.npz`, `spike_coverage.png` (tetrode); NP numbers
from script 67 (reads `peaks.npy` on npx_nfs; no artifact written).

## Wobble vs circus-omp head-to-head (scripts 69-72; measurement-based)

Question (user, 2026-06-17): is SI's `wobble` matcher (per-spike amplitude scaling + sub-sample jitter)
better than `circus-omp` on these tetrodes -- specifically for >=10 MAD coverage AT MATCHED PRECISION?
Design: SINGLE-WINDOW comparison on the three task-1 windows (5/26/40 h, 1800 s) -- one shared MS5 bank per
window feeds BOTH matchers (isolates the matcher; no carry-forward/reseed confound); each output is deduped
(0.95) and scored on four axes. Output: `wobble_vs_circus/` (json + figures); shared scorer `_wobble_eval.py`.

`wobble` is ALREADY in SpikeInterface (`method="wobble"`) and is geometry-free (visibility = the group
sparsity mask; pairwise overlaps only between channel-sharing units -> no cross-tetrode leakage), so this was
an integration + calibration task, not a port. Integration = one helper `_mp_common.wobble_method_kwargs`
(WobbleParameters nest under `parameters`; `engine`/`torch_device`/`shared_memory` are siblings).

**Smoke (script 69):** wobble runs error-free on our sparse 4-ch RAW bank, is deterministic (`engine="numpy"`,
two runs bit-identical), and its detection scale is the per-template ||t||^2 (median ~3e6 raw^2 at h=26).
Two tetrode-specific settings are load-bearing: (1) `approx_rank=4` (a 4-active-channel template has spatial
rank <=4; the SI default 5 keeps a ~0-singular-value garbage component); (2) `threshold` must be CALIBRATED --
it gates on the normalized objective `2*conv-||t||^2` (units of amplitude^2), so it is gain^2-scale-dependent
and the SI default 50 (Neuropixels uV) is wrong by ~5 orders of magnitude for raw units. circus-omp's
`amplitudes=[0.8,inf]` is, by contrast, scale-invariant.

**Calibration (script 70, matched-precision discipline):** sweep wobble `threshold` (as a factor x ||t||^2
median); pick the SMALLEST threshold whose median `rp_contamination` (over >=50-spike units, rate-gated, NOT
contamination-gated -> not circular with the tiers) is <= circus-omp's. circus-omp at h=26 produces MORE
spikes (1.35 M) and is MORE contaminated (rp 0.156) than wobble at 1x median, so the matched crossing sits
BELOW 1x median. Chosen operating point: **factor 0.55 x ||t||^2 median** (applied per-window; threshold
scales with each window's template energy).

**Head-to-head (script 71), wobble@0.55x vs circus[0.8,inf]:**

| window | matcher | spikes | units | >=12 MAD cov | rp_contam | spurious | MS5 match (mod tier) |
| -----: | ------- | -----: | ----: | -----------: | --------: | -------: | -------------------: |
| 5 h  | circus | 1.01 M |  99 | 94.8% | 0.312 | 11.2% | **0.254** |
| 5 h  | wobble | 1.09 M |  84 | **98.1%** | 0.310 | 13.2% | 0.184 |
| 26 h | circus | 1.35 M | 104 | 97.9% | 0.156 | 16.8% | **0.368** |
| 26 h | wobble | 1.59 M |  84 | **99.4%** | **0.085** | 24.7% | 0.226 |
| 40 h | circus | 1.17 M | 122 | **92.0%** | 0.357 | 14.9% | **0.328** |
| 40 h | wobble | 1.12 M | 110 | 82.7% | 0.305 | 15.3% | 0.248 |

**Verdict: wobble is NOT a clear win; keep circus-omp (but the case is weaker than first stated).** Against
the decision rule (higher >=12 MAD coverage at matched-or-better rp AND >= circus MS5 agreement AND not-higher
spurious): wobble has lower MS5 `match_frac` in all 3 windows (0.18/0.23/0.25 vs 0.25/0.37/0.33), under-covers
at h=40 (82.7% vs 92.0%), and higher spurious in 2/3. BUT each of these is softer than the headline:
- **MS5 agreement -- WEAK evidence (corrected 2026-06-17).** `match_frac` counts how many tier-good MS5 units a
  matcher reproduces, but MS5 is an oversplit/dropout-prone reference, so it scores FIDELITY-TO-MS5, not
  correctness. Wobble matches FEWER MS5 units but at HIGHER mean Jaccard (0.744/0.778/0.708 vs circus
  0.680/0.695/0.682) and yields fewer total units (84/84/110 vs circus 99/104/122, MS5 122/114/134). Fewer
  matches at higher per-match quality is EQUALLY consistent with wobble CONSOLIDATING MS5 oversplits (a GOOD
  thing -- oversplit is the project bottleneck) as with losing units; match_frac alone cannot distinguish. So
  this axis does not cleanly disfavor wobble. (Absolute match_frac is low for both, ~0.25-0.37 = the PoC's
  ~0.42 oversplit ceiling.)
- **h=40 under-coverage** is largely an INTRINSIC h=40 ceiling (script 74: even the loosest gate caps h=40 >=12
  cov at ~88%; wobble's admitted set never fits ~12% of h=40's large events), not pure threshold miscalibration.
- **higher spurious** is ~82% sub-threshold template-shaped fits (script 72), not noise.
Wobble's one consistent edge is lower `rp_contamination` (large only at h=26: 0.085 vs 0.156); rp is high for
BOTH at h=5/h=40 (0.31-0.36, intrinsically contaminated windows). NET: "wobble does not clearly beat circus,"
NOT "wobble is clearly worse" -- the strongest single reason to keep circus is the absence of a clear win plus
its scale-invariant gate (simpler to operate), not the MS5-agreement deficit.

**What "spurious" actually is (script 72, `spurious_examples.png`).** Spurious = matcher spikes with no
locally-exclusive >=5.5 MAD reference peak within +/-0.5 ms on the tetrode (over-detection proxy; the
reference detector is sorter-agnostic, NOT MS5's or the matchers' front-end). For BOTH matchers ~**82%** of
spurious spikes are **sub-threshold (<5.5 MAD, median ~5.0)** -- small, template-shaped deflections just under
the reference cut, NOT noise (the example snippets track the assigned template); only ~18-19% are
supra-threshold (collision-suppressed / near-miss). So "spurious" is largely a 5.5-MAD-thresholding artifact;
wobble's higher spurious (26.8% vs 18.8% at h=26) = MORE of the same sub-threshold fits (and at marginally
lower MAD), consistent with its amplitude-scaling reaching further below threshold -- not a noisier failure mode.

**Cross-cutting:** within a single stationary window with a window-matched bank, circus-omp already claims
**92-98%** of >=12 MAD events -- i.e. there is little within-window high-amplitude headroom for ANY matcher.
This reconfirms the scripts 64-66 finding: the full-recording >=12 MAD coverage gap is a TEMPLATE-DRIFT /
TRACKING phenomenon (fixed bank vs drift over 47 h), NOT a single-window matcher deficiency. So matcher choice
is not the lever for that gap; template tracking (carry-forward/reseed) is.

Artifacts: `wobble_vs_circus/{smoke_w26h,threshold_calib_w26h,headtohead}.json`,
`{threshold_calib_w26h,coverage_bands,precision_agreement,spurious_examples,supported_examples}.png`.

## Threshold selection on intrinsic merits — BombCell, the refractory knee, per-unit gate (scripts 73-75)

Follow-up (user, 2026-06-17): how to pick the OPTIMAL matcher threshold on each matcher's own merits, not by
matching the other? And what does BombCell consider max-tolerable contamination for a "good" unit?

**BombCell (SI `curation/bombcell_curation.py`).** Its MUA gate uses the SAME Llobet `rp_contamination` we
compute, with default **`rp_contamination < 0.1`** -> a unit with rp >= 0.1 is downgraded from "good" to MUA.
So 0.1 is BombCell's max-tolerable refractory contamination for a good single unit (per-unit, not a median).
Of BombCell's full metric set, only two are BOTH threshold-responsive (driven by the spike set, not the fixed
template shape) AND geometry-free: `rp_contamination < 0.1` (false-positive) and `amplitude_cutoff < 0.2`
(false-negative / missed-spike fraction). Its waveform-shape metrics (peak_to_trough_duration 0.1-1.15 ms,
num peaks, baseline_flatness, peak/trough widths and ratios) are geometry-free but template-static -> useful
for the MUA/noise-filtering track (Part C), not threshold selection. Two are UNUSABLE on fictional-geometry
tetrodes: `exp_decay` (spatial amplitude decay across channels) and `drift_ptp` (needs spatial localization).
`snr`/`presence_ratio` were non-discriminating / dropped in our isolation-tier work.

**Wobble intrinsic knee (script 73, h=26): ~0.65-0.70x ||t||^2 median** (vs the circus-MATCHED 0.55x =
looser/dirtier than wobble's own optimum). Sweep + per-threshold median rp / per-unit rp<0.1 count / isolation
tiers / >=12 cov / spurious / MARGINAL-spike quality (of spikes newly admitted when loosening, what % land on
a real >=5.5 MAD peak). Converging criteria: clean knee (median rp<0.03) = 0.70x; BombCell-median (rp<0.1,
loosest) = 0.60x; per-unit rp<0.1 count peaks 0.65-0.70x (48-49 units); marginal quality degrades monotonically
66%->50% as you loosen 0.90x->0.50x (below ~0.70x you buy spikes that are increasingly half-noise). At 0.70x:
median rp 0.029, >=12 cov 96.8% (~circus 98%), spurious 18.4% (~circus 17%). CAVEAT: `amplitude_cutoff`
SATURATED at 0.000 across the whole sweep -> NOT discriminating on these high-amplitude well-isolated units;
the live recall signals are >=12 coverage + marginal-quality, not amp_cutoff. (median rp is unit-noisy: read
the BAND 0.65-0.75x, not a point.) Artifacts: `intrinsic_knee_w26h.{json,png}`.

**Per-unit normalized gate (script 74, 5/26/40h): the COSINE/SHAPE gate dominates the AMPLITUDE gate.**
Post-filtered a generously-admitted wobble run by each spike's per-unit amplitude `a=conv/||t||^2` (= circus's
criterion) vs cosine `r=cos(snippet, template)`; swept fixed fractions. A fixed **`r>=0.6` drives median
rp_contamination to 0.000 in ALL three windows** while keeping 86-99.5% of >=12 MAD coverage; the amplitude
gate cannot reach rp~0 without crushing coverage (a>=0.9 still rp 0.018-0.055 at 78-89% cov). This QUANTIFIES
"cosine proposes, refractory disposes": a real low-amplitude spike has high cosine but low amplitude, so
amplitude gating wrongly drops it (the dropout mechanism) while shape gating keeps it and still excludes the
randomly-timed collision/noise fits that drive refractory violations. GENERALIZATION: a fixed cosine fraction
gives IDENTICAL (zero) contamination across windows (fully scale-invariant on precision, unlike the absolute
threshold's rp 0.085-0.36); the amplitude fraction only partially equalizes (rp 0.06-0.12 at a>=0.7). The
residual >=12 coverage spread (~12 pts at any fixed gate) is NOT miscalibration -- it is an intrinsic h=40
CEILING: even the loosest gate caps h=40 >=12 coverage at ~88% (wobble's admitted set never fits ~12% of h=40's
large events = drift / unit population the bank misses), while h=5/h=26 reach 97-100%. Directly supports the
residual-capture design (shape acceptance + refractory disposal) and shows circus's amplitude gate is
fundamentally limited vs a shape gate. Artifacts: `normgate.json`, `normgate_generalization.png`.

**Circus-omp amplitude knee (script 75, h=26): circus has NO BombCell-clean operating point.** Swept
`amplitudes=[lo,inf]`, lo in {0.5..1.0}, + the per-template auto bound (`amplitudes=None`); same intrinsic
scoring. `rank=5` auto-clamps to min(rank,n_channels)=4 on tetrodes (no fix needed, unlike wobble's approx_rank);
`omp_min_sps=0.1` permissive so `amplitudes` is the gate.

| amplitudes[0] | spikes | median rp | >=12 cov | spurious | marginal (% of newly-admitted on a real >=5.5 MAD peak) |
| ------------- | -----: | --------: | -------: | -------: | -----: |
| [0.5,inf] | 2.28 M | 0.232 | 99.2% | 41.8% | 10% (loosen 0.6->0.5) |
| [0.6,inf] | 1.94 M | 0.238 | 99.2% | 34.2% | 19% |
| [0.7,inf] | 1.64 M | 0.193 | 99.1% | 26.0% | 37% |
| [0.8,inf] (prod) | 1.35 M | 0.156 | 98.1% | 18.4% | 61% |
| [0.9,inf] | 1.05 M | 0.118 | 91.8% | 12.6% | 79% |
| [1.0,inf] | 0.71 M | 0.110 | 71.5% | 9.0% | -- |
| auto(None) | 2.16 M | 0.232 | 99.2% | 39.4% | -- |

KEY (and it partially walks back the head-to-head rejection): **median rp_contamination NEVER drops below
BombCell's 0.1 at any amplitude setting** -- floor ~0.11 even at `[1.0,inf]`, where coverage has collapsed to
71.5%. The amplitude gate cannot remove the residual contamination from same-tetrode shape/assignment confusion
+ collisions. Contrast wobble: rp 0.029 at its 0.70x knee, rp ~0 with the cosine gate. So on the intrinsic
refractory/coverage FRONTIER, **wobble dominates circus at h=26** (at ~97% coverage wobble is ~5x cleaner: rp
0.029 vs circus's 0.156; circus can't reach rp<0.1 at all). The earlier "wobble not a win" came from the
matched-precision framing forcing wobble to circus's loose 0.156 point + the weak MS5-agreement axis + the h=40
intrinsic ceiling -- NOT from a precision/coverage deficit. SECOND finding: circus's low-amplitude admissions
are MOSTLY JUNK -- loosening below [0.8,inf] adds spikes only 10-37% of which land on a real peak (vs wobble's
50-66%), because OMP fits low-projection residual/noise once the amplitude floor drops. This is WHY [0.8,inf]
is the standard (below it = noise) and WHY circus creates the dropout -- and it makes circus a POOR engine for
recovering the low-amplitude dropout (residual-capture Part A). `amplitudes=None` is permissive/dirty here
(rp 0.232) -- not recommended. NET verdict update: circus stays the validated, simple, scale-invariant PRIMARY,
but for CLEAN SUA / residual-capture, wobble's amplitude-scaling (real low-amp admissions) + a cosine/refractory
disposal is the better engine. Artifacts: `circus_knee_w26h.{json,png}`.

## Wobble as primary — gating bake-off (scripts 79-80; measurement-based)

Following the intrinsic studies, the question became: could wobble REPLACE circus-omp as the primary MP
matcher? That needs (a) a recalibration-free gate (circus's `amplitudes` ratio is scale-invariant; wobble's
objective threshold is `||t||^2`-scale-dependent), and (b) evidence the gate generalizes. The engineering is
now in place: `run_matching(..., shape_gate_r=)` applies a scale-invariant cosine acceptance gate
(`per_spike_cosine`: r = cos(snippet, assigned template)), and `method=`/`shape_gate_r=`/`wobble_factor=` are
threaded through `windowed_carry_forward[_reseed]` (circus stays the default; wobble is selectable). Unit test:
`tests/test_wobble_primary_wiring.py` (incl. the scale-invariance assertion -- a half-amplitude copy scores
r=1). A 1-window production smoke through script 58 (`--method wobble --shape-gate-r 0.6`) gave 76 units,
ISI<1ms median 0.000, 0 units >1% refractory.

**Bake-off (script 79):** per the "compare, don't assert" discipline, the two recalibration-free candidates
were tested HEAD-TO-HEAD across 6 windows (3/11/19/27/35/43 h) with circus `[0.8,inf]` as reference, scored on
>=10 MAD coverage (the task-1 spec) / median rp_contamination / low-amplitude (5.5-10 MAD) retention:
  - **(i) adaptive-||t||^2** -- wobble threshold = factor x tsq_median(bank), sweep factor.
  - **(ii) cosine shape gate** -- one permissive admit (0.45x), sweep r post-hoc (scale-invariant, free).

Generalization = spread of (rp, >=10 cov) across the 6 windows at a single fixed gate value:

| gate (fixed across windows) | median rp range (spread) | >=10 MAD cov range | low-amp retention |
| --- | --- | --- | --- |
| circus `[0.8,inf]` | 0.206 - 0.417 | 90.8 - 96.1% | 44 - 61% |
| adaptive-||t||^2 0.55x | 0.134 - 0.332 (0.198) | 92.3 - 97.7% | 40 - 65% |
| adaptive-||t||^2 0.80x | 0.056 - 0.211 (0.155) | 77.6 - 88.6% | 24 - 47% |
| **cosine r>=0.60** | **0.000 - 0.000 (0.000)** | **92.5 - 97.4%** | **45 - 69%** |
| cosine r>=0.70 | 0.000 - 0.000 (0.000) | 85.1 - 92.8% | 35 - 56% |

**Result (decisive):** ONE fixed cosine value (r>=0.60) gives median rp_contamination = 0.000 in EVERY window
(spread 0.000 -- recalibration-free, the scale-invariance claim confirmed) while holding >=10 MAD coverage
92.5-97.4% and low-amp retention 45-69%. It **Pareto-dominates both alternatives on all three axes at once**
[CORRECTED 2026-06-19, see top banner: these "three axes" are rp / coverage / low-amp retention -- all
precision-recall PROXIES, none per-unit ASSIGNMENT PURITY; tight-trough purity of the deliverable is only
0.79-0.83 and the gate is delete-only, so this does NOT establish assignment correctness]:
vs circus (rp 0.21-0.42 at comparable coverage) and vs the adaptive-||t||^2 threshold, whose rp swings
window-to-window at any fixed factor (spread 0.16-0.23) and only nears clean at 0.80x -- where rp is still
0.056-0.211 AND coverage drops to 78-89% AND low-amp retention collapses to 24-47% (the documented amplitude-gate
floor: it cannot get clean without dumping real low-amplitude spikes; the cosine gate gets clean WHILE keeping
them). Spurious fraction is comparable across gates (cosine 8-26% vs circus 11-20%), so the cosine gate does not
buy precision by over-rejecting. Artifacts: `gate_bakeoff.{json,png}`.

**Finer-r sweep to pin the knee (script 79 `--tag _finer_r --abs-factors 0.45`, r in 0.50-0.65 by ~0.025):**
the coarse grid's clean point (rp=0 first at 0.60) was bracketed -- the true knee is lower. Per-window
median-rp range across the 6 windows: r=0.50 0.006-0.025 | 0.53 0.000-0.014 | 0.55 0.000-0.003 |
**0.57 0.000-0.000** | 0.60 0.000-0.000 | 0.65 0.000-0.000; >=10 cov falls monotonically 0.57->0.65
(92.8-98.0% -> 89.2-95.9%). **Pinned r* = 0.57** = the SMALLEST r with median rp=0.000 in EVERY window; it
strictly dominates 0.60 (same zero contamination, ~1 pt more >=10 cov + ~1-2 pt more low-amp retention). 0.55
is near-clean (rp<=0.003) for a hair more coverage if a trace of contamination is acceptable, but 0.57 is the
principled fully-clean pick. **Measured knee r* = 0.57; operating value set to r>=0.60 per user** (conservative
margin at ~1 coverage pt, "close enough") -- Phase B uses `--shape-gate-r 0.60`. Artifacts: `gate_bakeoff_finer_r.{json,png}`.

**Gate gallery (script 80, qualitative companion -- what each gate admits, on fixed templates):** for the top-3
circus-count units in each of 2 windows (u83/u23/u26 @ 26 h, u119/u30/u21 @ 40 h -> 6 figures
`gate_gallery_u*_w*h.png`), ONE figure per unit with 6 rows: the four GATE-admit rows (circus [0.8,inf] /
adaptive 0.55x / knee 0.68x / cosine r>=0.57), each panel a 4-ch snippet over the template overlay (MAD
labelled), plus the two DIFFERENTIAL rows from the permissive 0.45x run split by a=conv/||t||^2 vs r=cos
(a/r labelled): cosine-only (a<0.8, r>=0.57 -- shape gate KEEPS, amplitude gate DROPS) and amplitude-only
(a>=0.8, r<0.57 -- the reverse). The pictures make the selectivity unmistakable: the cosine-only row is a LARGE
population of small-but-template-tracking deflections (the low-amplitude recoveries), while the amplitude-only
row is TINY and the snippets visibly DEPART from the template (large wrong-shape mis-fits). Concretely for u83
(tet 12) @ 26 h: circus admits 93,383 spikes to it, cosine-only ~145k vs amplitude-only ~510 -- so the shape
gate recovers orders of magnitude more real template-shaped spikes than the amplitude gate uniquely keeps, and
rejects the few large mis-fits the amplitude gate wrongly admits. (An earlier single-unit a/r decomposition,
u71 @ 26 h, gave the same story quantitatively: cosine-only 44,751 vs amplitude-only 9,726; circus 24,086 vs
the cosine gate's 86,554.) CAVEAT worth carrying into Phase C/D: the cosine-only population is large, so the
full-deliverable characterization must confirm these low-amplitude admissions are genuine recovery (refractory
already clean, rp=0) and not inflating spurious units.

**Admit x cosine-r 2D grid (script 81; 4 windows, admit in {0.25..0.65} x r free post-hoc) -- validating the
permissive admit.** The bake-off's 0.45x admit was an unvalidated heuristic; this maps the (admit, r) plane to
test whether the two knobs interact (admit ~ an amplitude gate, since the objective `||t||^2(2a-1)` makes
threshold-factor f ~ a>=(1+f)/2; r = scale-invariant shape gate). Aggregated over 4 windows, median >=10 MAD
coverage [admit x r=0.60] and rp:

| admit | >=10 cov @ r0.60 | rp @ r>=0.55 |
| --- | --- | --- |
| 0.25x | 97.7% | 0.000 |
| 0.35x | 97.6% | 0.000 |
| 0.45x | 96.7% | 0.000 |
| 0.55x | 94.1% | 0.000 |
| 0.65x | 90.2% | 0.000 |

FOUR findings: (1) **r GOVERNS precision** -- median rp=0.000 at r>=0.55 for EVERY admit (admit is irrelevant
to contamination once r>=0.55; only at the loose r=0.50 edge does a higher admit clean a residual 0.014->0.000).
(2) **the admit CAPS coverage and the knobs INTERACT** -- at fixed r=0.60, coverage falls 97.7->90.2 as admit
climbs 0.25x->0.65x, because a high admit (amplitude gate) pre-kills low-amplitude spikes before r ever sees
them; so they are NOT a free trade (your "lower-admit+stricter-r == higher-admit+lower-r" holds only at the rp
edge, not for coverage). (3) **coverage SATURATES at admit ~0.35x** (0.25x == 0.35x within noise); **0.45x was
mildly binding** (~1 cov pt below the plateau). (4) below 0.35x the raw kept-spike COUNT keeps climbing (panel c)
while coverage stays flat -- the extra admitted spikes are redundant/MUA near already-claimed peaks, NOT new
large-event coverage, so saturation must be judged on COVERAGE not count (going lower only adds non-coverage
spikes + compute + MUA-inflation risk). **Operating point for Phase B: `--wobble-factor 0.35` (the measured
saturation, replacing the 0.45x heuristic, +~1 cov pt) `--shape-gate-r 0.60`.** Artifacts: `admit_r_grid.{json,png}`.
Visual companion (script 82, `admit_gallery_u*_w26h.png`): per admit level, the ADMITTED vs REJECTED spikes for a
fixed unit. Candidate pool = one permissive 0.15x run; each spike is graded at level f by ITS OWN a against
wobble's exact bar a_f(u) = (1 + f*M/||t_u||^2)/2 (label=decision, so admitted panels provably have a>=bar). MAD
is measured at the spike (+/-0.5 ms), not window-max. Concretely the energy bias is visible across two units:
u71 (tsq_u/M=2.04, big) has a low bar (~0.64 @ 0.55x) so its spikes sail through; u6 (tsq_u/M=0.72, small) has a
bar climbing 0.67->0.95 across 0.25x->0.65x, so it loses most spikes to the energy threshold even at the loosest
level. As f rises the rejected block fills with template-shaped low-a spikes (high r) -- the amplitude-gate
dropout the scale-invariant cosine gate keeps.

Scope: this is Phase A (the gating choice) of a study-first, go/no-go plan. The decision to actually FLIP the
default to wobble is deferred to Phase B/C/D -- a full-48 h `windowed_carry_forward_reseed` run with the cosine
gate, characterized vs the circus `assembled_reseed` deliverable (yield, isolation tiers, identity continuity,
determinism, wall-clock, curated-good agreement). circus remains the validated default until that completes.

## Phase B ladder — controls, gate refinement, B_w full-48 h (scripts 81, 83, 58; measurement-based)

To attribute any wobble-vs-circus difference to the *matcher* vs the new *gate*, Phase B is a 2x2 ladder
(matcher in {circus, wobble} x gate in {off, cosine}); B0 = circus/off = the existing `assembled_reseed`.
This section records the control-factor lock, a gate refinement that was tested and REJECTED, and the first
ladder rung (B_w = wobble/off).

**B_w factor lock (script 81 no-gate row, 4 windows).** B_w (wobble alone, no shape gate) must sit at
circus's precision so the comparison isolates the matcher. The admit-x-r grid now also scores each wobble
run with the gate OFF. Median across 4 windows, vs circus median rp 0.249 / cov10 93.9%:

| wobble admit (no gate) | wobble-alone median rp | \|rp - circus\| | cov10 |
| --- | --- | --- | --- |
| 0.45x | 0.310 | 0.061 | 98.5% |
| **0.55x (locked)** | **0.224** | **0.025** | 95.4% |
| 0.65x | 0.153 | 0.096 | 90.9% |

**Locked B_w factor = 0.55x** (closest no-gate rp to circus); 0.6x interpolates to ~0.19 rp (~0.06 cleaner
than circus). Per-window the match wanders 0.45-0.55x (circus's own rp ranges 0.21-0.37). Confirms the
gate-equivalence value f=0.6 (a>=(1+f)/2 == circus a>=0.8) runs slightly conservative, because wobble's
objective is energy-weighted, not a uniform a>=0.8 gate. (The grid also re-confirmed r>=0.55 -> median rp
0.000 in EVERY window x admit -- the cosine gate's precision is admit-independent.)

**Tight-window / shift-tolerant cosine -- TESTED, REJECTED (script 83).** The gallery discussion proposed
restricting r to the discriminative peak (drop the flat pre-peak baseline + low-SNR AHP tail) and tolerating
+/- small integer shifts (undo integer-rounding misalignment). Both MONOTONICALLY raise r, so each
(window x shift) variant was evaluated at its OWN re-pinned r* holding median rp <= 0.01 (MATCHED precision),
judged on low-amp (5.5-10 MAD) coverage -- recovery at fixed precision, not a higher r. Same 4 windows, one
permissive 0.25x pool, 20 variants (5 windows x {0,+-1,+-2,+-3} shift). Median across windows, low-amp
coverage at matched precision (baseline = full-window fixed cosine = production r):

| cosine window | shift | r* @rp<=0.01 | cov_low % | vs baseline |
| --- | --- | --- | --- | --- |
| wide (-0.5..+1.0 ms) | 0 | 0.65 | 84.2 | +0.9 |
| **full (-1.0..+2.0 ms, production)** | **0** | **0.525** | **83.3** | **baseline** |
| full | +-2 | 0.575 | 80.9 | -2.4 |
| sym (+-0.75 ms) | 0 | 0.725 | 78.0 | -5.4 |
| asym (-0.3..+0.8 ms) | 0 / +-2 | 0.78-0.80 | 75.7 | -5.7 / -7.6 |
| tight (-0.27..+0.4 ms) | any | -- | nan | gate never reaches rp<=0.01 |

**Verdict: the full-window fixed cosine is best; do NOT implement the refinement.** Tightening the window
HURTS low-amp retention at matched precision (asym -5.7, sym75 -5.4; the 20-sample "tight" window can't reach
rp<=0.01 at any r* up to 0.80) -- for low-SNR spikes the repolarization/tail carries discriminative shape, so
cutting it forces a higher r* that drops more real spikes. Shift tolerance is neutral-to-harmful (full|+-2
-2.4): it only raises r* without recovering coverage, so the integer-misalignment concern does not translate
into recoverable low-amp spikes. The one variant that nominally beats baseline (wide|0, +0.9 pt) is within
window noise AND needs a higher r* -- not worth changing. So r* stays at the ~0.55 knee (operating 0.60) and
`per_spike_cosine` is unchanged. Artifacts: `tight_cosine_test.{json,png}`.

GALLERY CONFIRMATION (script 84, window 27 h, units 66/79, one figure per MAD band, asym window shaded):
cov_low measures coverage of detected 5.5-10 MAD *events*, not proven-real spikes, so the verdict needed the
waveforms. For each unit, candidate spikes split into DROPPED-BY-TIGHTENING (full keeps rF>=0.525, tight+shift
drops rA2<0.80 -- each at its OWN matched-precision r*) vs KEPT-BY-BOTH. Two model-free results make "tightening
drops REAL spikes" hold: (a) the dropped fraction is **amplitude-INDEPENDENT** (~13-19% across every band; u79's
LARGEST drop, 19%, is at 10-14 MAD) -- if tightening rejected noise the drop would collapse at high MAD, not
peak there; (b) the dropped 10-14 MAD panels are unmistakably large template-tracking spikes (a 10-14 MAD
locally-exclusive deflection on the template can't be noise), visually indistinguishable from the kept-by-both
spikes -- the tight window rejects them only on benign trough fine-shape/alignment that its short, high (0.80)
r* over-penalizes, while the full window's larger sample count clears them at r* 0.525. The low (5.5-7 MAD) band
is inherently fuzzy (both dropped and kept are low-SNR wiggles), so the clinching evidence is the high-MAD
dropped spikes + the flat drop-vs-amplitude curve. Artifacts: `tight_cosine_gallery_u{66,79}_w27h_mad*.png`.

REASSIGNMENT TEST -- the gallery's "real" was right but INCOMPLETE; the dropped spikes mostly DON'T best-belong
to the assigned unit (script 85; user: "real, but maybe not THIS template's"). For each dropped spike, cosine to
EVERY same-tetrode template under full AND tight+shift windows; is the assigned unit u still the best trough
match, or a NEIGHBOR? Result (27 h, r*_full 0.525 / r*_asym 0.80): pooled dropped n=524k -> u-is-best-trough 46%,
**NEIGHBOR wins 54%** (of those, neighbor clears r*_asym 46%); on CROWDED tetrodes far higher (u66 tet10 7 units:
72% neighbor-wins; u79 tet12 8 units: 80%). Control kept-by-both: 77% u-best. So the full-window cosine
MIS-ASSIGNS a large fraction of the dropped spikes to u, and the trough catches it -- "tightening drops u's real
spikes" was WRONG for the majority. CONSEQUENCES (corrected): (1) the gate is a FILTER not a reassigner -- in
production `run_matching` deletes sub-r* spikes and the neighbor never re-claims them, so tetrode coverage still
falls (cov_low correct for the gate-as-implemented); "belongs to a neighbor" doesn't rescue coverage, it means the
full window CONTAMINATES u's train with neighbor spikes. (2) cov_low measures TETRODE COVERAGE, BLIND to per-unit
ASSIGNMENT PURITY -- the axis where the tight window helps and which the test never scored (and rp can't see:
independent co-located neighbors don't violate u's refractory). (3) the purity benefit is itself CONFOUNDED by
twin-vs-distinct (cosine can't tell if the winning neighbor is u's oversplit TWIN [mixing harmless] or a DISTINCT
cell [contamination]; only refractory-CCG settles it -- same limit as the reseed work). NET: keep the full-window
cosine as the precision/coverage gate (a gate deletes, doesn't reassign; can't cleanly buy purity), BUT co-located
trough-distinct mis-assignment is real + substantial on crowded tetrodes and belongs to the template-separation/
dedup track, NOT the acceptance gate -- the trough cosine is a SIGNAL for separation, not a gate threshold.
Artifacts: `tight_reassign_u{66,79}_w27h.png` (snippet vs u-template vs winner-neighbor-template).

**B_w full-48 h run (script 58 `--method wobble --wobble-factor 0.6 --tag _bw_f06`; kept at 0.6x per user
despite the 0.55x lock -- a slightly conservative control).** Reseed pipeline (1800 s windows, reseed every
12, dedup 0.95), matcher = wobble alone, no gate. Seed 109 confident -> 76 after dedup; 76 seed -> **162
total units (86 added by re-seeding)** at windows [12,24,36,48,60,72,84]; continuity median per-unit
present-frac 0.61; ISI<1ms median 0.0006; **9/162 units >1% refractory** (clean); born-unit at-birth
cos-to-bank median 0.74 (71/86 borderline >=0.6). Deliverable: `assembled_reseed_bw_f06/` + `reseed_bw_f06.npz`.
The 86 reseed births with high cos-to-bank (no shape gate to suppress near-duplicates) is the no-gate arm's
expected signature. Head-to-head vs circus B0 (`assembled_reseed`) per isolation tier is Phase C/D (not yet
run); the gated arms (B_wg = wobble+cosine, B_cg = circus+cosine) are also not yet run.

## Open / next
- **Residual-capture (SUA dropout recovery + MUA extraction): see `RESIDUAL_CAPTURE_PLAN.md`** for the full
  plan — recover banked-unit dropout (cosine proposes, refractory disposes), extract per-tetrode MUA
  (cosine-to-any-template sieve + BombCell shape QC, NOT UnitRefine/state), and the gating experiments
  (E1 neural/noise calibration, E2 SUA residual-capture prototype). E3 wobble-vs-circus is DONE (scripts
  69-72): wobble was not a win at circus's LOOSE matched point; but on its own intrinsic frontier + with the
  scale-invariant cosine gate it DOMINATES (scripts 73-80, sections above), so it is now the candidate primary
  matcher and the better residual-capture engine, pending the full-48 h go/no-go (Phase B/C/D).
- Long-drift result (above) -> if reestimate >> fixed continuity, matching pursuit tracks drift
  geometry-free; then build the two-pass deliverable (confident bank -> carry-forward over 48 h).
- Alternatives if assignment confusion limits: `wobble` matcher + cosine gate (scripts 69-80 -- leading
  candidate, Pareto-dominant gating frontier; go/no-go pending); tighter amplitude band; per-tetrode
  template-set dedup before matching.
- Cross-cutting: MS5 oversplit reduction (shared bottleneck for chunk+match AND matching pursuit).

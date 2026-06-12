---
title: Cross-chunk unit tracking for long tetrode recordings
scope: tetrode_analyses
status: active
source: code_inspection
created: 2026-06-11
last_updated: 2026-06-11
confidence: high
confirmed_by_user: not_required
---

# Cross-chunk unit tracking for long tetrode recordings

Companion to `SORTING_COMPARISON_FINDINGS.md`. That document established the
single-pass MountainSort5 behavior on the 48 h tetrode recording
(`2026-05-27_09-07-52`). This one covers the **drift-tracking** problem: how to
sort a recording whose units' waveforms evolve over time, on a probe whose
geometry is fictional.

## Problem

- 48 h, 16 tetrodes (64 ch). Probe geometry is **fictional** (4 wires very close,
  configuration unknown; the `ProbeGroup` from `io.build_tetrode_probegroup` is
  synthetic). See `project_tetrode_zarr_scheme`.
- MS5 scheme 3 (12 h blocks, 1 h training): a 12 h block pattern is visible in
  firing-rate profiles, and per-block clusters of one physical neuron do not merge
  by traditional criteria.
- MS5 scheme 2 (full 48 h, single template): firing-rate profiles are
  discontinuous at ~100000 s (~27.7 h). The user's working hypothesis is that the
  tetrodes physically moved during the sleep-deprivation period; no clean merge
  exists across the discontinuity.
- The need: a sorter that lets a unit's waveform **evolve gradually over time**,
  without a geometric drift model.

## Method evaluation (which existing tools apply to tetrodes)

References live in `sorting/refs/`. Findings are from code inspection of the
vendored repos at the workspace root and the cited papers.

### Tetrode localization is possible but needs known geometry + a forward model

Mechler et al. 2011 (`3d_unit_loc_with_tetrodes.pdf`): a tetrode's *nonplanar*
contact configuration gives it full spherical spatial sensitivity, and neurons can
be localized in 3D — **but only** via a dipole source model fitted against a volume
conductor + lead fields computed for the *measured* tetrode geometry (≈96% of EAP
power captured; recording radius R₅₀≈97 µm). Approximate geometry already raises
the fit error (fMSE 3%→6%); a fictional geometry has no forward model at all. The
cheap point-source localization used by drift-correction pipelines is **not** how
tetrodes localize. Consequence: **localization-driven drift correction is not
available for this data.**

### DARTsort — REJECTED for tetrodes (drift model is linear-array-specific)

`DARTsort.pdf` + code at `dartsort/`. DARTsort's drift model is, end to end, built
for a linear probe:

- **Registration corrects only depth.** `Algorithm 1` and §4.2 (Eq. 1):
  `x_n^reg = (x_n, y_n, z_n − P(t_n, z_n))` — only the depth coordinate `z` is
  registered; `P` is a `T × D` displacement over depth bins. Motion is estimated by
  DREDge (`dredge.pdf`), which operates on a 1D depth axis (rigid, or non-rigid
  windowed along depth). Code: `dartsort/src/dartsort/util/registration_util.py`
  (`dredge_register(depths_um=z, ...)`), `util/motion.py`, `util/drift_util.py`.
- **Drift correction assumes a repeating geometric pitch.** §4.3: drift is
  decomposed into integer-pitch shifts ("the distance at which the contact geometry
  repeats along the probe's long axis, e.g. 40 µm for Neuropixels 1.0") plus a
  sub-pitch amplitude rescale under a point-source model. A 4-wire tetrode has no
  depth axis and no pitch; `util/waveform_util.py:get_pitch` is ill-defined for
  near-coincident contacts, and `util/motion.py` asserts a valid pitch when
  `drifting`.
- **Even with motion off it leans on localization features.** `config.py`:
  `do_motion_estimation=False` is supported, but the global clustering (§5.1.2)
  clusters on `(x_n^reg, z_n^reg, ptp)` localization features
  (`localize/localize_torch.py`, point-source/dipole fit over a channel
  neighborhood). With 4 co-located channels these collapse.

Decision (confirmed with user): **documented rejection**, no empirical run. The
above is sufficient evidence.

### UnitMatch — geometry-free attributes usable; spatial ones not

`UnitMatch.pdf` + `UnitMatch/`. Of its ~7 similarity attributes
(`ExtractSimilarityMetrics.m`), roughly half are pure waveform shape (waveform
correlation `WVCorr`, `WaveformMSE`, `AmplitudeSim`) and half require known channel
positions (`spatialdecay`/`spatialdecayfit` ∝ distance-to-channel, `CentroidDist`/
`CentroidVar`/trajectory angle & distance — all from amplitude-weighted positions
using `channelpos`). On a fictional geometry the spatial terms collapse or
mislead. The waveform-shape terms survive. → use the *idea* (geometry-free waveform
similarity), not the stock pipeline.

### Yuan EMD — REJECTED as stock (architecturally linear-probe)

`EMD_unit_matching.pdf` + `Yuan-Neuron_Tracking/`. The EMD distance
(`weighted_gdf_nt.m`) mixes waveform features (duration, FWHM, peak-trough ratio,
recovery slope) with hard spatial terms: fitted `fitX/fitZ` from `chan_pos`
(`fit_loc.m`), z-spread, and a z-distance match threshold (`NT_main.m`,
`EMD_unit_match.m`) — the pipeline assumes drift along a 1D probe axis. Waveform
features are reusable; the framework is not.

## Chosen approach — overlapping chunks + geometry-free agreement tracking

Sort in short **overlapping** chunks (MS5 scheme 2), match units across
**consecutive** chunks by spike-train **Jaccard agreement in the overlap region**
(the same physical spikes appear in both sorts there — geometry-free,
ground-truth-like), corroborated by a geometry-free 4-channel template cosine, then
**chain consecutive matches transitively** into global unit identities. Every link
is a high-confidence short-range match, so a unit's template may drift arbitrarily
over 48 h; a chain splits at ~100000 s iff no overlap match bridges it (honest if
the electrode moved). Decisions confirmed with user: chunk sorter = MS5 scheme 2
only; matching = overlap agreement (primary) + cosine (corroboration); UnitMatch/EMD
deferred.

### Implementation

- Library: `tetrode_analyses/src/tetrode_analyses/tracking.py`
  (`plan_chunks`, `materialize_chunk`, `sort_chunk`, `build_chunk_analyzer`,
  `match_overlap`, `template_cosines`, `chain_matches`,
  `assemble_global_sorting`, `track_span`).
- Shared eval helper: `sorting/_track_eval.py` (well-isolated tiers from
  `20_curated_agreement.py`; reconstruction-vs-single-sort scoring).
- Scripts: `30_derisk_chunk_uss.py`, `31_track_reconstruct_vs_singlesort.py`,
  `32_track_param_sweep.py`, `33_track_across_discontinuity.py`.

### Key reused facts / gotchas

- Genuine-crop materialize avoids the `frame_slice` time-vector memory pitfall
  (`project_si_frame_slice_timevector_memory`): slice the *zarr*, `reset_times()`,
  save a small binary. Verified at start_frame > 0 (see de-risk below).
- `NumpySorting.from_samples_and_labels` / `from_unit_dict` take FRAMES;
  `from_times_and_labels` takes SECONDS. `BaseSorting.frame_slice(a, b)` keeps
  `[a, b)` and re-zeros — used to crop chunk sorts to the overlap.
- Group sparsity (`ChannelSparsity.from_property(..., by_property="group")`) gives
  every unit its own tetrode's 4 channels in a fixed order, so two units' (T, 4)
  templates are directly comparable without geometry.
- Short chunks (≤3600 s) stay far below the MS5 int32 duration ceiling
  (`project_ms5_int32_duration_limit`).

## Experimental results

### De-risk: per-chunk memory at start_frame > 0 — PASS

`30_derisk_chunk_uss.py`, chunk `[86400, 88200] s` (start_frame 2.59e9): materialize
0.3 min, sort 1.1 min (16 tetrodes, 110 units), **peak USS 7.7 GB** (binary 13.8 GB)
vs ~415 GB full-parent. Confirms the genuine-crop path keeps per-worker memory
chunk-sized away from frame 0, and that a chunk sort is cheap (~1.4 min).

### Reconstruction vs single sort — tracking is lossless; chains break (31)

`31_track_reconstruct_vs_singlesort.py`, 2 h drift-stable epoch `[36000, 43200) s`,
chunk_s=1800, overlap=0.5, jaccard_min=0.5, cosine_min=0.9. Reference = single MS5
scheme-2 sort of the whole epoch (119 units). Reconstruction = 162 global units
from 7 chunks. Moderate-tier well-isolated reference units (n=78):

| metric | single chunk vs ref (in-window, NO tracking) | chunk+track recon vs ref (whole epoch) |
| --- | --- | --- |
| matched-unit mean agreement | 0.782 (intrinsic ceiling) | **0.801** |
| match fraction | 0.888 | **0.641** |

Two separate effects:

1. **Matched-unit agreement is at the ceiling** (0.80 ≥ 0.78): the chaining does NOT
   corrupt the units it tracks. The 0.78 ceiling is MS5's intrinsic chunk-vs-whole
   decomposition variability — the same block-size sensitivity documented in
   `SORTING_COMPARISON_FINDINGS.md` (halving the block already drops well-isolated
   agreement to ~0.78). The absolute 0.9 gate is unreachable because the single sort
   is not ground truth; the right reference is this intrinsic ceiling, which tracking
   meets.
2. **Match fraction drops 0.89 → 0.64**: this gap is **chains breaking**. Only 41% of
   global units span all 7 chunks (`frac_global_spanning_all_chunks=0.407`); a good
   unit present in every chunk gets reconstructed as several partial globals, each
   below 0.5 Jaccard with the full-epoch reference unit, so it counts as unmatched.
   Cause: at jaccard_min=0.5 some consecutive-chunk overlaps fall below threshold
   (MS5 chunk variability + few spikes in the 900 s overlap). Fixable by lowering
   jaccard_min and/or enlarging the overlap → the sweep (32).

Net: the approach tracks units without degrading them; the open tuning problem is
preventing chain breaks. (No `n_singletons`, so no orphan single-chunk units.)

### Operating-point sweep (32)

`32_track_param_sweep.py`, same 2 h epoch, chunk_s ∈ {900,1800,3600} × overlap ∈
{0.25,0.5}, plus a free threshold sub-sweep (jaccard_min ∈ {0.3,0.4,0.5} × cosine_min
∈ {off,0.85,0.9,0.95}) over cached edges. Moderate-tier match fraction / agreement:

| chunk_s | overlap | n_chunks | n_global | match_frac | agreement |
| --- | --- | --- | --- | --- | --- |
| 900 | 0.5 | 15 | 211 | 0.667 | 0.767 |
| 1800 | 0.5 | 7 | 162 | 0.641 | 0.801 |
| **3600** | **0.5** | **3** | **108** | **0.846** | 0.787 |
| 3600 | 0.25 | 3 | 113 | 0.821 | 0.782 |

(best per cell at jaccard_min=0.3; reference had 119 units.)

- **Larger chunks win on match fraction**: fewer boundaries (fewer chain breaks) AND
  less MS5 per-chunk over-splitting (900 s yields ~2× the global units of 3600 s).
  3600 s gives 108 globals (vs 119 reference) and frac 0.846, near the 0.888
  single-chunk ceiling. Agreement is flat at ~0.78–0.80 everywhere (the sorter ceiling).
- **The cosine gate is redundant**: match_frac and global-unit count are *identical*
  across cosine_min ∈ {off, 0.85, 0.9, 0.95} — every reciprocal-Jaccard match already
  has cosine ≥0.95. Overlap spike-agreement alone is the operative signal. The cosine
  is a free cross-check, but should be DISABLED for drift robustness: at an
  electrode-movement event the waveform jumps while the spikes stay continuous, so a
  strict cosine gate could wrongly break a same-neuron bridge.
- Tradeoff for production: chunk size trades drift-stationarity (favors small) against
  sort quality/trackability (favors large). 1 h chunks are far shorter than the 12 h
  blocks that caused the original artifact, while keeping over-splitting and chain
  breaks low. **Operating point: chunk_s≈3600 s, overlap=0.5, jaccard_min≈0.3, cosine off.**

### The ~100000 s discontinuity is a STOP/RESTART, not drift (33)

The `openephys_provenance` annotation on the recording shows it is **two Open Ephys
experiments welded into one continuous sample axis** by `io.get_recording`
(`ConcatenateSegmentRecording`):

- experiment1: samples `[0, 3025748403)`, ends at 100858.3 s (≈28.0 h), `settings.xml`.
- experiment2: samples `[3025748403, 5215033052)`, `settings_2.xml`.
- The weld is at **sample 3025748403**; the time vector jumps **57.66 s** there
  (`t[b-1]=100858.280 → t[b]=100915.938`) — the recording was stopped and restarted,
  and the acquisition config changed (different settings file).

So the discontinuity the single-template scheme-2 sort shows at ~100000 s is this
stop/restart boundary, NOT gradual electrode drift. (User-confirmed cause.)

`33_track_across_discontinuity.py` ran chunk+track over `[90000, 110000) s` at the
operating point — i.e. with chunks laid on the WELDED axis (the mis-framed setup):
bridge_rate_at_discontinuity = 1.0, median at other boundaries = 1.0 (saturated). The
metric does NOT isolate the weld — and that is the finding: because the two
experiments are welded contiguously at the sample level, overlap-agreement sees
"identical samples in both chunks" across the weld and bridges right through it. The
stop/restart is **invisible to sample-level overlap tracking**; it can only be handled
from the provenance, by splitting the recording at sample 3025748403 and never letting
a chunk straddle it.

**User clarification + measurement (why the one-shot sort breaks but tracking should not):**
The two experiments are the same units on the same channels with the same waveforms;
welding the 58 s gap away is legitimate (treat it as ~58 s of discarded data). So
bridging across the weld is *correct*, not a bug — no segment-aware special-casing is
needed. The question is then why the one-shot scheme-2 sort showed per-unit
discontinuities. Measured at the weld (`30_`-adjacent probe):

- In-band (300–6000 Hz) MAD-noise steps **up ~8.5%** across the weld (per-channel
  after/before ratio 1.04–1.14, coherent across all 64 ch); raw DC shifts ~6 counts
  (removed by bandpass). Origin ambiguous from this measurement — could be a
  power-cycle effect or a brain-state change (the weld is in the sleep-deprivation
  period, more activity → higher in-band hash). [needs_verification]
- The one-shot sort's **aggregate** spike rate is continuous across the weld
  (~22–25k/30 s both sides) → the per-unit discontinuities are NOT a detection failure.

Mechanism: MS5 applies a **global** whitening matrix and (scheme 2) a **global**
classifier. The ~8.5% in-band step shifts the whitened snippet distributions enough that
spikes near cluster decision boundaries get reassigned between neighboring units at the
weld — individual units jump while the total stays flat. Chunk+track is robust to this
because each chunk is whitened and classified **locally** (statistics matched within the
chunk), and chains bridge the weld via shared spikes. So the welded chunk+track approach
handles the stop/restart correctly; no segment-aware chunking required. (The earlier
"must be segment-aware" note is superseded by this clarification.)

### Full 48 h production run — consecutive chaining FRAGMENTS over many boundaries (34)

`34_track_full_48h.py` at the operating point (chunk_s=3600, overlap=0.5, jaccard_min=0.3,
cosine recorded), whole recording, 96 chunks, 283 min (resumable; per-chunk sorting +
group-templates checkpointed). Result: **1352 global units**, median **4 chunks/unit**
(~3 h lifespan), only **1.6%** span ≥48 chunks, **0.1%** (1 unit) span all 96. 1122
unmatched unit-nodes were dropped (chain_matches only nodes edge endpoints — should
retain unmatched units as singletons; minor fix).

Root cause — **compounding chain breaks**, quantified:
- Per-(boundary,group) bridge rate = **0.817 mean / 0.833 median**. At p=0.82 per boundary,
  the geometric mean chain length is 1/(1−p) ≈ 5.5 chunks — matching the observed median 4.
  Spanning all 96 chunks needs ≈ 0.82⁹⁵ ≈ 10⁻⁸, so essentially none survive end-to-end.
- The 18% per-boundary break rate is the same MS5 chunk-vs-whole variability (~0.78–0.82)
  seen throughout: within-chunk over-splitting + the one-to-one (Hungarian + reciprocity)
  match drop ~1 in 5 units' clean continuation at each boundary. The 2 h validation (6
  boundaries) didn't expose this; 95 boundaries do.

This means consecutive-overlap chaining alone does **not** deliver 48 h tracks; it delivers
~3 h tracks. Bigger chunks help only weakly (fewer boundaries) before reintroducing the
drift problem. The fix is **chain healing**: re-link chain ends to chain starts across
boundaries where overlap-Jaccard failed, using the geometry-free template cosine — the
cross-gap matching the cosine was always meant for (it is redundant for consecutive
matching, but load-bearing here). Healability measured on the checkpoints: of 1248 breaks,
**39% have an unambiguous continuation** at end+1 (cosine ≥0.95, clear winner; median
continuation cosine 0.973), 6% ambiguous, 27% no candidate at end+1 (gap >1 chunk or a
genuine ending). Caveat: template-only links carry a false-merge risk on tetrodes (4
channels → limited spatial discrimination), so healing needs a conservative threshold and
likely human review of merges. Implementation is cheap to iterate — Phase A (the 4.7 h
sort) is checkpointed, so healing variants re-run only Phase B/C (~2 min).

### Chain healing helps marginally — consecutive chaining has a HARD compounding ceiling (35)

`35_track_heal.py` (`tracking.heal_chains`, unambiguous-only: mutual-best end→start at
chunk+1, cosine ≥0.95, margin ≥0.03; unmatched units retained as singletons). Applied
**519 merges** (median cosine 0.994 — confident), 2474→1955 global units. But long-track
coverage barely moved: units spanning ≥half = 0.8%→1.5%. Decisive follow-up — bridge rate
stratified by template amplitude (isolation proxy):

| amplitude | per-boundary bridge rate |
| --- | --- |
| Q1 (low, junk-like) | 0.74 |
| Q3 | 0.82 |
| top 10% (cleanest) | **0.913** |

**Even the cleanest units bridge at only 0.91**, and 0.91⁹⁵ ≈ 1.8e-4 → essentially no unit
spans the recording; clean-unit geometric-mean chain ≈12 chunks (~7 h). The ~9% per-boundary
failure is MS5's irreducible chunk-vs-whole variability (over-split + one-to-one Hungarian
match drop), and it **compounds fatally** over 95 boundaries. No per-boundary tweak (healing,
larger overlap, looser jaccard) escapes this — you'd need >0.99/boundary.

**Architectural conclusion:** consecutive-chain tracking (chain + heal) is the wrong
architecture for 48 h. The correct one is **global clustering of unit-nodes** per tetrode:
treat every (chunk, unit) as a node with a template + temporal extent, take the
overlap-Jaccard matches as high-confidence must-link anchors, and cluster the rest by
template similarity **across arbitrary gaps** — so a single missed boundary never severs a
global unit (membership is similarity-to-cluster, not an unbroken chain). This is what
DARTsort's time-window ensembling (§5.1.3) and UnitMatch/EMD cross-session matching do. The
existing per-chunk checkpoints (sorts + group templates) are exactly the input such a global
matcher needs, so it's a Phase-B/C swap (minutes to iterate), not a re-sort.

### Global clustering result + the fundamental ceiling (36)

`36_track_global_cluster.py` (`heal_chains` generalized with `max_gap`: gap-tolerant global
per-tetrode agglomeration; overlap chains = must-link anchors, template links across up to
`max_gap` missed chunks, mutual-best + unambiguous + cosine ≥0.95). Swept max_gap ∈ {1,2,3}.
Result: clean-unit (template pp ≥170 µV) coverage spanning ≥half went 2.6% → only ~4%, and
**merges DECREASED with wider gaps** (399→336→270) — wider windows surface more
similar-template candidates, so the unambiguity test fails (4-channel tetrode templates are
not discriminative enough). The merges that pass are **safe** (long-clean-unit ISI-violation
median 0.0006, p90 0.0017 — no false merges), just far too few.

Decisive diagnostic (re-ran overlap comparisons, full agreement, clean units, sampled
boundaries): of clean-unit boundary breaks, only **1% are recoverable** by relaxed matching
(a high-Jaccard partner strict one-to-one rejected) while **7% are genuinely missed** — the
unit has NO Jaccard ≥0.3 partner in the next chunk because MS5's local sort there simply
didn't isolate it. So the achievable overlap bridge rate is ~93% even with perfect matching,
and 0.93⁹⁵ ≈ 0.001.

**Conclusion (negative result, well-quantified):** robust 48 h *single-unit* tracking on
this tetrode data via chunk-and-match is not achievable, and it is not an algorithm we can
tune away. The per-boundary breaks are dominated by per-chunk *isolation* failures (~7% even
for clean units — no signal to match), not matching failures (~1%), and template healing
can't bridge them because 4-channel templates are too ambiguous. Three matchers
(consecutive chain, endpoint heal, gap-tolerant global cluster) all hit the same 0.93⁹⁵
wall. The fundamental tension: drift needs short chunks (many boundaries) while tracking
needs few; and cluster-per-chunk isn't consistent enough chunk-to-chunk to survive 95 of them.

### KS4-per-chunk does NOT lift the ceiling (38) — 2026-06-12

`38_track_ks4_chunks.py` sorts the same overlapping chunks with Kilosort4 (deconvolution;
`templates_from_data=False` to avoid KS4's no-cap KMeans hang on dense data and to keep seed
templates identical across chunks; `do_correction=False`/`nblocks=0`; 64-ch-together +
peak-channel tetrode assignment) over the 6 h drift-stable epoch [36000, 57600) s, then measures
the same clean-unit (template pp ≥170 µV) per-boundary bridge rate as MS5 on the IDENTICAL chunk
grid (MS5 from the existing checkpoints, chunks 20–30; no re-sort).

Result (10 boundaries): **KS4 0.773 (197/255) vs MS5 0.874 (263/301)** — KS4 is WORSE and finds
~15 % fewer clean units/chunk. So the per-boundary isolation ceiling is **not** specific to MS5's
clustering; KS4's deconvolution does not beat it. The "a template-matching sorter would not lose
a unit in a window" hypothesis is, for KS4-per-chunk on this tetrode data, **refuted**. Caveat:
the fast 64-ch-together mode + peak-channel tetrode assignment could marginally disadvantage KS4
via cross-chunk tetrode reassignment vs MS5's strict per-tetrode sort; a per-tetrode KS4 rerun
could tighten the gap but is unlikely to reverse a 0.10 deficit plus fewer clean units. No full
48 h KS4-per-chunk run was launched (`39_track_ks4_full_48h.py` staged, unused).

### QC analyzer + the well-isolated reframing (37, 40) — 2026-06-12 (user-confirmed gate)

`37_build_analyzer_tracked.py` builds a geometry-free SortingAnalyzer (`analyzer_clustered.zarr`)
on `global_sorting_clustered.npz` (2204 units): NO `unit_locations`/`spike_locations` and NO
`drift` metric (fictional geometry); PCA isolation metrics (mahalanobis / d_prime /
nearest_neighbor / silhouette) on PCA of the sparse 4-ch waveforms.

**Isolation tiers (final, user-confirmed):** a unit is well isolated at tier T iff
`firing_rate >= fr_floor` AND (`rp_contamination <= rp_hi` OR `sliding_rp_violation <= srp_hi`).
Thresholds (SPOT in `ecephys.wne.siutils`; gate in `_track_eval.isolation_tier_mask`):

| tier | fr ≥ | rp_contamination ≤ | sliding_rp ≤ | **count** |
| --- | --- | --- | --- | --- |
| permissive | 0.1 | 0.5 | 0.3 | **262** |
| moderate | 0.2 | 0.3 | 0.2 | **173** |
| conservative | 0.3 | 0.1 | 0.1 | **103** |

`40_characterize_tracked_qc.py` crosses isolation with chunk-SPAN (from provenance):
- All 2204 units: median span 2 chunks (most are short fragments / MUA).
- **Conservative (103): median span 29 chunks (~15 h), max 96 (full 48 h); 95/103 span ≥8 chunks
  (~7 h), 25/103 span ≥48 chunks (~24 h), across 15 of 16 tetrodes**, firing rates 0.31–30 Hz.
- Link audit (on the earlier 44-unit set): of the 41 conservative units spanning ≥8 chunks,
  **38 are PURELY CONSECUTIVE** overlap chains (zero gaps); only 3 use a healed gap (max gap 2).

**How the gate was derived (the metric investigation, 42–44):**
- The §34–36 negative result conditioned on AMPLITUDE-clean units — an over-inclusive proxy
  dominated by marginal/MUA fragments that break (→ 0.87 bridge). The genuinely WELL-ISOLATED
  units (by refractory contamination) are a different, better-behaved subpopulation that DOES
  track over a median ~15 h, up to the full 48 h.
- **Firing-rate floors lowered to 0.1/0.2/0.3 Hz** — L2/3 cells fire ~0.1–0.3 Hz; the old 0.5 Hz
  floor excluded real units (median FR across all 2204 units is 0.025 Hz).
- **`isi_violations_ratio` dropped from the gate.** It disagrees with `rp_contamination` sharply
  and asymmetrically (≈398 units pass rp but fail isi at 0.1; ≈1 the reverse). The decomposition
  (42) shows this is driven by the **refractory window** (isi 1.5 ms vs rp 1.0 ms), not the
  formula: matching windows collapses the 398-unit disagreement to 0. `isi_violations_ratio`
  also has a ~1/rate² inflation that over-rejects low-rate units. The population ISI histogram
  (43) shows the 1.0–1.5 ms band is a **smooth refractory-recovery flank, not a discrete
  artifact** (no fixed-lag double-detection; no 2.5 ms / 400 Hz ISI fingerprint; no sub-0.5 ms
  ISIs), so the wider window catches real low-level contamination — a biological judgment, so we
  don't hard-gate on it.
- **`sliding_rp_violation` is OR'd in as a salvage clause, never a sole gate.** It is **NaN for
  89 % of units** (44): refractory coincidences scale as rate², ≈0 at 0.025–0.3 Hz, so it has
  power only for high-rate units. NaN = abstain and MUST fail its clause (else the 89 % would
  pass spuriously). Where it *does* have power it slides the refractory period, so it salvages
  genuine short-refractory fast-spikers the fixed 1 ms window miscounts: it adds **28** units to
  the conservative tier, all high-rate (median 1.7 Hz, ~298 k spikes) and verified clean at a
  0.5 ms window (`rp_contamination@0.5ms` clean for 100 % of them, rising monotonically with
  window — the signature of a real ~0.5–0.75 ms refractory period).

**Conclusion (confirmed):** robust 48 h tracking of **well-isolated single units** on this
tetrode data IS achievable via chunk-and-match — ~103 conservative units over a median ~15 h, up
to the full 48 h; what is not achievable is tracking the marginal majority (noise/MUA), which is
fine. Residual risk: a healed-gap bridge could stitch two template-similar but distinct neurons
(refractory metrics can't see a non-time-overlapping merge); applies to ≤3 of the long
conservative units (the rest are dense consecutive chains).

## Bottom line

> **Revised 2026-06-12 — see the two sections immediately above.**
> The negative result below is real but specific to AMPLITUDE-clean units. Conditioned on actual
> isolation quality (rp_contamination OR sliding_rp + a firing-rate floor), **103 conservative
> well-isolated units track over a median ~15 h, up to the full 48 h** (the long ones are dense
> consecutive chains). KS4-per-chunk does not beat MS5 (0.773 vs 0.874), so the amplitude-clean
> ceiling is intrinsic to chunk-and-match on this data, not an MS5 artifact.

- What the approach DOES deliver: drift-tolerant tracking over **~7 h spans for clean units**
  (geometric mean ~12 chunks), with **no 12 h-block artifact** and correct handling of the
  stop/restart weld — a real improvement over the prior 12 h-block and 48 h-single-template
  sorts, just not whole-recording identity.
- What would be required for true 48 h tracking: a **drift-aware template-matching /
  deconvolution sorter** that carries per-unit templates *through* time and updates them
  gradually, so a unit never disappears merely because clustering failed to re-find it in one
  window (the ~7% miss). That is the missing infrastructure (cf. SpikeInterface issue #4427);
  no off-the-shelf tetrode tool does it without a geometric drift model. Kilosort 4 per chunk
  (deconvolution, drift off) is the nearest experiment worth trying, since template matching
  would not "lose" a unit in a window the way clustering does.

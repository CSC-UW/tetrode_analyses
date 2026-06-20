---
title: Residual-capture plan — recover SUA dropout + extract MUA from the matching-pursuit residual
scope: tetrode_analyses
status: active
source: inference
created: 2026-06-17
last_updated: 2026-06-19
confidence: medium
confirmed_by_user: not_required
---

# Residual-capture plan

Proposed implementation for recovering spikes the matching-pursuit (MP) tetrode sorter currently misses,
in two complementary tracks: (A) recovering dropped spikes of **well-isolated banked units** (SUA), and
(B) collecting **real-but-unsortable neural events into a per-tetrode MUA** product, separated from noise.
Synthesized from the spike-coverage investigation (see `MATCHING_PURSUIT_FINDINGS.md` §"Spike coverage /
detection completeness", scripts 64-68) and a design discussion with the user (2026-06-17). NOT yet built —
this is the roadmap, gated by the calibration experiments in §8.

## 1. Goal

Capture more of the **>=10 MAD detected events** the current MP deliverable leaves unclaimed — events the
user confirmed are real spikes by eye (>=7 MAD "all spikes", >=10 MAD "definitely want them") — **without
sacrificing assignment precision**, and additionally to **retain real low-amplitude / unsortable activity
as MUA** rather than discarding it (it is neural information useful for population rate and OFF-period work).

Two hard constraints from the user:
- **No blanket gate-loosening.** Capturing more must not come by relaxing acceptance everywhere; the metric
  is coverage of >=10 MAD events **at fixed-or-better assignment precision** (refractory contamination +
  cross-unit assignment), not coverage alone.
- **No circular use of brain state.** ON/OFF and sleep state are *outputs* of the sorting pipeline, so they
  must not be foundational *inputs*. State is admissible only as **downstream validation**.

## 2. What we measured (self-contained background)

Coverage = fraction of detected threshold-crossing EVENTS (locally-exclusive neg peaks) within +/-0.5 ms
of a sorted spike, stratified by amplitude (MAD). Tetrode, 47 h, 141.8 M events >=5.5 MAD:

| amplitude (MAD) | MP-reseed (95 u) | chunk+match (2204 u) |
| --------------: | ---------------: | -------------------: |
| 5.5-7  | 18.6% | 20.4% |
| 9-12   | 64.0% | 93.3% |
| 12-16  | 82.9% | 99.5% |
| >=16   | 93-96% | 99.9% |

- The low overall % (MP 42%, chunk+match 55%) is the **near-threshold MUA floor** — even 2204 units leave
  ~80% of 5.5-7 MAD unclaimed; that band is genuine hash. High-amplitude coverage is the meaningful number.
- **Cause of the MP >=12 MAD gap** (script 65, MS5-resort at 5/26/40 h): NOT missing clean units (0 of
  68-88 well-isolated MS5 units/window absent from the bank). The unclaimed-large events are
  **~50-75% within-unit DROPOUT of banked units** (the spike sits inside a clean banked unit; circus-omp
  did not place it) **+ ~25-48% non-isolable overlap/MUA** (collisions a fresh MS5 sort can't resolve).
- NP reference (script 67, KS2.5): false-negatives are a **general** property — every production NP sort
  leaves ~1-9% of large + 30-90% of near-threshold events unclaimed. MP (83-95% large) sits just below NP
  KS (91-99.6%); chunk+match (99.5%+) matches the best.

Key mechanism for Part A. circus-omp gates on the fitted amplitude `a` with `amplitudes=[0.8, inf]`, where
`a = ⟨spike,template⟩/⟨template,template⟩ = cos(spike,template) · |spike|/|template|`. So a **correct-shape
but smaller-amplitude** spike (e.g. a big unit's 10.5 MAD spike against its 15 MAD average template:
`a≈0.7`, cos≈0.95) is **rejected by the amplitude gate but would pass a shape/cosine test**. That is exactly
the recoverable dropout — and a dropped spike, being unsubtracted, sits in the residual at full amplitude.

> **MEASURED 2026-06-19 (E2 DONE, script `87_sua_residual_capture_e2.py`).** The mechanism is SUPPORTED but
> MODEST, not the dominant story. Of 358 k >=10 MAD unclaimed events (reseed bank, 5 windows): **~33%
> recovered** (cosine>=0.8 + refractory pass), **~67% no-template** (cos<0.8 to the best same-tetrode
> template = collision / MUA, which flow to Part C, NOT discarded). Recovered-vs-host CCG: 49 own-dropout
> (refractory dip = clean recovery) / 42 abstain / **0 independent-contaminant** -> no detectable cross-unit
> leakage. Caveat: rp_contamination is SATURATED at 1.0 for many affected units (they are already oversplit
> twins), so the CCG-duplicate verdict, NOT rp-delta, is the load-bearing purity check. Recovery lands mostly
> in already-oversplit units, so Part A is entangled with the MERGE-FIRST step (see MATCHING_PURSUIT_FINDINGS
> 2026-06-19 banner): merge duplicate tracks first, then recover.

## 3. Design principles

1. **Targeted, not blanket.** Operate only on clear unclaimed events (>= an amplitude floor), with explicit
   per-event validation — never relax global acceptance.
2. **Cosine proposes, refractory disposes (SUA).** Cosine is non-discriminative between shape-similar
   tetrode units, so it can only *propose* a candidate unit; the *identity* decision is the refractory
   check (does inserting the spike keep that unit's train clean?).
3. **Shape separates neural from noise (MUA).** With state off-limits upstream and pooled MUA lacking a
   usable refractory structure, **waveform shape (physiology) is the only foundational neural-vs-noise
   signal.** Use rule-based physiological metrics (BombCell) + a matched-filter (cosine-to-any-template),
   not an isolation-quality classifier.
4. **State validates, never gates.** ON/OFF and sleep-state checks are applied only to the finished product.

## 4. Part A — SUA residual-capture (recover banked-unit dropout)

**Order:** SUA templates -> SUA residual-capture -> (then) MUA pass, so the MUA bucket never eats spikes
that were recoverable SUA.

**Mechanism (per unclaimed event >= amplitude floor):**
1. Best-matching bank template by max-shift (±10 samp) 4-ch cosine.
2. **Propose** assignment iff cosine >= `tau_cos` (a known neural shape match).
3. **Dispose** by refractory: accept iff inserting the spike does NOT create a <~1.5 ms ISI in that unit's
   existing train (the unit's own dropped spike falls where it was silent; a different unit's spike forced
   onto it usually lands inside its refractory window -> rejected).

**Why it recovers what the omp gate rejected:** a correct-shape lower-amplitude spike fails the amplitude
gate (`a<0.8`) but passes cosine + refractory (§2 mechanism). **Why it is precision-safer than loosening
the gate:** restricted to clear large peaks + best-match + refractory-validated, vs a blanket relaxation of
every fit everywhere (which also admits the confusion-prone 5-9 MAD band).

**Honest limits:**
- cosine+refractory is imperfect on tetrodes (shape-similar units + chance refractory-pass) -> residual
  cross-unit leakage; measure it (E2).
- collisions have a distorted residual -> low cosine -> correctly left for the MUA/overlap bucket.
- drift-mismatch (carried template lagged the spike) -> low cosine -> NOT recovered here; that is a
  re-estimation problem, not residual-capture.

**Implementation notes:** the unclaimed peaks are already saved (`spike_coverage.npz`: `peak_sample`,
`peak_group`, `amp_mad`, `claimed_0`); a prototype can run directly on those (E2) before wiring it into the
windowed pass. The proper version reconstructs the residual (recording − fitted templates) per window, or
re-detects on the residual, then applies steps 1-3 against the live (re-estimated) bank.

## 5. Part B — alternative at the source: the `wobble` matcher

`circus-omp`'s dropout traces to fixed templates + a single amplitude-fit gate. `wobble`
(spikeinterface.sortingcomponents.matching) models **per-spike amplitude scaling** and **sub-sample
temporal jitter** with upsampled templates, so it should (a) accept a unit's amplitude-varying spikes
natively — no second pass, no global gate change — and (b) resolve overlaps/collisions better, hitting both
failure modes at the source. **Untested on 4-channel fictional-geometry tetrodes** (the PoC only validated
circus-omp). Treat as an experiment (E3), not a known win; it could replace or complement Part A.

> **UPDATE 2026-06-17 — E3 + intrinsic studies DONE (scripts 69-75; see MATCHING_PURSUIT_FINDINGS.md).**
> Wobble runs geometry-free on tetrodes (deterministic; `approx_rank=4`). As a drop-in PRIMARY at matched-loose
> precision it was NOT a clear win (head-to-head, scripts 69-72). BUT on each matcher's OWN intrinsic optimum
> (scripts 73-75) the picture reverses for the residual-capture use case:
> - **circus's amplitude gate has a hard contamination floor it cannot beat**: median rp_contamination NEVER
>   drops below BombCell's 0.1 at any `amplitudes[0]` (floor ~0.11 at `[1.0,inf]`, where >=12 coverage collapses
>   to 71.5%); the residual contamination is same-tetrode shape/collision confusion, which amplitude gating can't
>   remove. **wobble reaches rp 0.029 at its knee, ~0 with a cosine gate.** So wobble's precision/coverage frontier
>   DOMINATES circus's (at ~97% coverage: rp 0.029 vs 0.156).
> - **circus is a poor engine for recovering the low-amplitude dropout** (the Part-A target): loosening below
>   `[0.8,inf]` admits spikes only 10-37% of which are real (on a >=5.5 MAD peak), because OMP fits residual/noise
>   once the floor drops. wobble's amplitude-scaling admits real low-amp spikes (50-66% real) -- it recovers the
>   dropout AT THE SOURCE. That, plus the cosine-gate result (74: r>=0.6 -> rp~0 at 86-99% cov, all windows),
>   says SHAPE acceptance (not amplitude) is the right gate -- and neither matcher uses it natively.
> **Decision: circus stays the validated, simple, scale-invariant PRIMARY; but for CLEAN SUA / Part A recovery,
> wobble (or the cosine+refractory gate below) is the better engine.** Reconfirmed: within a stationary window
> circus already claims 92-98% of >=12 MAD events, so the full-recording gap is template DRIFT, not the matcher.

> **WOBBLE vs CIRCUS for residual-capture (data-backed, 2026-06-17).** REFRAME: Part A rests on
> `a = cos(spike,templ)*|spike|/|templ|`, so a real in-unit spike with good shape but low magnitude has high cos
> but `a<0.8` -> circus drops it (the dropout). The clean recovery gates on SHAPE (cosine), which NEITHER matcher
> does natively. So the cosine-proposes/refractory-disposes machinery is the right tool regardless of primary
> matcher; the matcher choice affects (i) how much dropout exists, (ii) at-the-source vs residual-pass recovery.
> - **Part A (SUA dropout).** WOBBLE pros: models per-spike amplitude scaling -> recovers low-amp in-unit spikes
>   at the source (50-66% real admissions vs circus's 10-37%); sub-sample jitter resolves collisions that Part A's
>   `locally_exclusive` peak detector SUPPRESSES (~25-48% of the gap is overlap). WOBBLE cons: still amplitude-,
>   not shape-based -> recovered spikes are a real/noise mix needing the SAME refractory disposal; harder to
>   operate (absolute threshold, no clean single-value generalization). CIRCUS pros: predictable dropout target
>   (`a<0.8` in-unit spikes), validated/simple, native per-template auto bound. CIRCUS cons: its amplitude gate
>   is the CAUSE of the dropout and cannot recover it cleanly; worse collisions.
> - **Part C (MUA).** Largely matcher-agnostic (cosine-to-any-template sieve on the residual). Wobble leaves a
>   SMALLER residual (less for Part C, but may absorb genuine MUA into SUA = MUA-in-SUA risk); circus leaves a
>   LARGER residual (more for Part C, but more SUA-dropout to pull out first = SUA-in-MUA risk).
> - **Recommendation:** keep circus PRIMARY + implement Part A as a COSINE-gated residual pass (the clean, safe
>   recovery; works whichever matcher is primary). Wobble's one residual-relevant edge is COLLISION resolution
>   (jitter) -> if the non-isolable-overlap portion matters, run a collision-aware residual step (wobble on the
>   residual, or matching-pursuit-with-subtraction in Part A) rather than the peak-detector, which misses collisions.
>   Wobble's amplitude-scaling does NOT justify replacing circus as primary, but IS the better low-amp recovery
>   engine if Part A is folded at-the-source.

## 6. Part C — MUA recovery (real-but-unsortable neural)

**Goal:** one **MUA "unit" per tetrode** = real neural events not assignable to SUA, with noise removed.
Per-tetrode (not one global) preserves spatial/anatomical localization.

**Cluster-light pipeline:**
1. **Input** = events still unclaimed after SUA + SUA-residual-capture, above an amplitude floor (~7 MAD;
   see the low-amplitude tension in §10).
2. **Per-event neural sieve (cheap, no clustering): cosine >= `theta` to ANY SUA template.** This inverts
   the "cosine can't discriminate units on a tetrode" finding into an asset: real spikes share the
   stereotyped tetrode morphology so they match *some* template; broadband noise/artifact matches none. A
   matched-filter "spikeness" test. Cuts the low-amplitude mass without clustering it.
3. **Light cluster the survivors** (a much-reduced set) — only now, after the sieve.
4. **BombCell shape QC** on those clusters (`spikeinterface.curation.bombcell_label_units`,
   `bombcell_get_default_thresholds`): use the **geometry-free temporal metrics** (`peak_to_trough_duration`
   ~0.1-1.15 ms, `num_negative_peaks`/`num_positive_peaks`, `waveform_baseline_flatness`<0.5,
   `trough_width`/`peak_before_width`, peak/trough ratios). **REJECT only the `noise` label** (shape-based);
   **KEEP good+mua+non-soma** as candidate-neural. Do NOT use BombCell's good-vs-mua split — that is an
   isolation/contamination axis that would wrongly penalize real MUA for being contaminated (same trap as
   UnitRefine, see §7). **Drop BombCell's spatial/decay metrics** — OOD on 4 co-located fictional-geometry
   channels (the UnitRefine geometry-free lesson).
5. **SUA-rescue:** any survivor cluster that passes the isolation gate -> promote to SUA (few expected per
   §2, but free).
6. **Artifact-burst rejection:** clusters of repeated identical-shape artifact events (pass a per-event
   shape test individually but are obviously one artifact) -> reject.
7. **Pool kept events per tetrode -> MUA** (union; de-dup coincident), persisted with an `is_mua` property.

## 7. Noise filtering (neural vs noise) — the crux, and what NOT to use

- **Foundational signal = waveform shape.** State is off-limits upstream; pooled MUA has no clean refractory
  structure (many units -> within-window coincidences -> no autocorrelogram dip), so refractory cannot
  validate MUA neural-ness the way it does SUA. Shape carries it.
- **Two complementary shape signals:** cosine-to-any-SUA-template ("matches a known neural shape") +
  BombCell physiological metrics ("has spike-shape statistics"). An event/cluster passing **both** is
  confidently neural; BombCell catches the cosine sieve's chance-match false positives (a noise snippet
  that clears `theta` against one template usually fails the duration/n-peaks/baseline checks).
- **Do NOT use the UnitRefine `unitrefine_advisory` classifier as the discard gate.** It was retrained to
  agree with the **isolation gate**, i.e. it ranks isolation quality; MUA is low-isolation-but-neural, so it
  sits in the classifier's "reject" class alongside true noise -> it would discard the MUA we want to keep.
  It is also uncalibrated ("rank, don't threshold"). A quick falsification test: feed it clusters with clear
  state-modulation (definitely neural) but poor isolation; if it calls them noise, it is confirmed unfit.
- **Do NOT (easily) train a learned neural/noise event classifier** — the blocker is clean *noise* labels:
  positives are easy (SUA spike snippets) but negatives need guaranteed noise *peaks* ("peaks no sorter
  could cluster" is circular; random-time snippets aren't peaks, so a spike-vs-random classifier calls every
  threshold crossing neural). Prefer rule-based physiology (BombCell) + matched-filter (cosine). If a model
  is trained later, use artifact-epoch / synthetic-noise negatives and verify it doesn't label all peaks
  neural.

## 8. Experiments / tests (ordered by cost; the first two gate everything)

- **E1 — neural/noise calibration (cheapest; gates Part C).** On these tetrodes, compute (a) max-cosine-to-
  SUA-bank and (b) BombCell shape metrics for two reference sets: **known well-isolated SUA spikes** (should
  read neural) and **guaranteed-noise snippets** (random non-peak times; should read noise). Deliver: the
  separation each signal — and the two together — achieves, the false-positive/false-negative rates, the
  chosen `theta`, and whether BombCell's NP-tuned defaults transfer to tetrodes (recalibrate narrowly on
  known-good SUA if not, per the threshold-derivation policy). If neither signal separates cleanly, the
  shape-only MUA approach is not viable as specified — reconsider before building.
- **E2 — SUA residual-capture prototype — DONE (script `87_sua_residual_capture_e2.py`, 2026-06-19).** On
  the saved unclaimed >=10 MAD peaks (`spike_coverage.npz`), best same-tetrode template cosine PROPOSES +
  refractory DISPOSES, recovered-vs-host CCG validates. Result: ~33% recovered, ~67% no-template (-> Part
  C), 0 CCG-detected cross-unit contaminants (rp saturated for oversplit hosts -> CCG is the cost metric).
  Verdict: Part A is worth wiring in, but AFTER the merge-first step (recovery lands in oversplit units).
  Reuses `_assignment_eval.best_template_for_events` / `ccg_verdict_pair`. Outputs `residual_capture_e2.*`.
- **E3 — `wobble` vs `circus-omp` head-to-head — DONE (scripts 69-72, 2026-06-17).** On the task-1 windows
  (5/26/40 h), scored both on >=12 MAD coverage at matched precision (rp_contamination), MS5 agreement, and
  over-detection (spurious fraction). Result: wobble is NOT a win (loses MS5 agreement all 3 windows,
  under-covers at h=40, higher spurious; cleaner rp but with fewer units). Part B shelved; stay on circus-omp.
  See MATCHING_PURSUIT_FINDINGS.md.
- **E4 — MUA end-to-end (medium).** Build the per-tetrode MUA over a multi-hour span via §6; report yield,
  retained-event shape stats, and how much of the >=10 MAD residual it accounts for.
- **E5 — MUA validation, strictly downstream / non-circular (after E4).** Does the MUA rate modulate
  Wake>NREM, silence during OFF periods, and align with mua-bugnon OFF detection? This is where state
  legitimately re-enters — checking the answer, not making it.

**Metric discipline throughout:** coverage of >=10 MAD events **at fixed-or-better precision**; calibrate
every threshold on known-good SUA (should pass) + guaranteed-noise (should fail) before trusting it on the
residual.

## 9. Deliverables / outputs

- Scripts (ACTUAL numbering, updated 2026-06-19): E3 was realized as `69_wobble_smoke.py`,
  `70_wobble_threshold_calib.py`, `71_wobble_vs_circus.py`, `72_wobble_spurious_examples.py`; the
  threshold-selection follow-up as `73_wobble_threshold_intrinsic.py`, `74_wobble_normgate_prototype.py`,
  `75_circus_amplitude_knee.py` (+ shared `_wobble_eval.py`). The assignment-purity reorientation added
  `_assignment_eval.py` / `_scoreboard.py` and scripts `86_rescore_axis_b_c.py`, `87_sua_residual_capture_e2.py`
  (= E2), `88_competitive_reassign.py`, `89_scoreboard.py`. STILL TO BUILD: E1 neural/noise calibration and
  E4 MUA pass as the next free numbers (90/91), plus the merge-first candidate build. Key finding for Part A:
  a per-unit COSINE/shape gate (r>=0.6) drives rp_contamination to ~0 at 86-99% coverage where the amplitude
  gate cannot -- "cosine proposes, refractory disposes" -- BUT rp~0 is precision-proxy, not assignment purity
  (see MATCHING_PURSUIT_FINDINGS 2026-06-19): the gate cannot fix cross-unit/oversplit mis-assignment.
- Recovered SUA spikes merged into the assembled MP sorting; per-tetrode MUA added with an `is_mua` unit
  property. Outputs under `track_eval/mp_long_s2000_d170000/`.
- Findings -> `MATCHING_PURSUIT_FINDINGS.md`; update this plan's front-matter `status` as parts land.

## 10. Open questions / risks

- **Amplitude floor for MUA.** MUA lives at low amplitude, exactly where noise dominates and every shape
  discriminator is weakest, and clustering the full low-amplitude mass is expensive. A ~7 MAD floor (user's
  by-eye boundary) cuts cost and noise but discards the faintest distant MUA. Decide deliberately; the faint
  tail is unrecoverable by shape alone.
- **Cosine library for the sieve.** Larger library (2204 chunk+match) -> more chance a noise snippet matches
  *some* template -> worse specificity. Prefer the clean ~95-unit SUA bank. Calibrate `theta` on noise (E1).
- **BombCell default transfer** to tetrodes (NP-tuned) — validate on known-good SUA (E1).
- **Is Part A worth it** vs simply adopting chunk+match (already 99.5%+ large) + the contamination-guarded
  merge? E2 decides. If the dropout recoverable fraction is small or leaky, chunk+match may be the better
  route to coverage.
- **Collisions are irreducible** — a single >=10 MAD crossing that is two overlapping spikes should not be
  assigned to one unit; expect (and want) <100%.

## 11. Relationship to the rest of the pipeline

- **chunk+match** already captures 93-99.5% of >=10 MAD (it re-clusters every chunk, no carry-forward
  dropout) at the cost of 2204 oversplit units. If coverage is the priority over SUA cleanliness, "adopt
  chunk+match + the contamination-guarded merge" is an alternative to Parts A/B.
- The **contamination-guarded merge** (cosine proposes within-tetrode pairs, CCG refractory-dip accepts;
  see the cadence/gate findings) shares the exact "cosine proposes, refractory disposes" machinery as Part A
  — build them on a common helper.
- All cosine here uses the geometry-free max-shift 4-ch `cosine_from_templates` (tracking.py); thresholds
  audited in `MATCHING_PURSUIT_FINDINGS.md` (dedup/heal merge at 0.95; admit gates 0.8/off).

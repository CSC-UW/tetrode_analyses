---
title: MountainSort5 parameter defaults — vanilla ms5 vs NP example vs SpikeInterface wrapper
scope: tetrode_analyses
status: active
source: code_inspection
created: 2026-06-02
last_updated: 2026-06-02
confidence: high
confirmed_by_user: not_required
---

# MountainSort5 parameter comparison

Compares the **Scheme 2** sorting parameters (and preprocessing) across three sources:

1. **Vanilla ms5** — `Scheme2SortingParameters` dataclass defaults
   (`mountainsort5/schemes/Scheme2SortingParameters.py`, also Scheme1/3).
2. **ms5 NP example** — `examples/neuropixel_quickstart/spikeglx.py` (the
   `sorting_params` dict + preprocessing pipeline). Uses scheme 2.
3. **SpikeInterface wrapper** — `sorters/external/mountainsort5.py` `_default_params`
   and how it constructs `Scheme2SortingParameters`.

Legend: **=** value matches vanilla ms5 default; **≠** differs. "(required)" =
the dataclass field has no default (caller must supply). ms5 itself does **no**
preprocessing — it expects already-filtered + whitened input — so preprocessing
rows are N/A for vanilla.

## Detection
| Parameter | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| `detect_threshold` (phase 2) | 5.5 | **5.25** ≠ | 5.5 = | NP lower |
| `phase1_detect_threshold` | 5.5 | 5.5 = | 5.5 = (wrapper sets = `detect_threshold`) | same |
| `detect_sign` | −1 | **0** ≠ | −1 = | NP detects both signs |
| `detect_time_radius_msec` (phase 2) | 0.5 | 0.5 = | 0.5 = | same |
| `phase1_detect_time_radius_msec` | 1.5 | **0.5** ≠ | **0.5** ≠ | both override vanilla to 0.5 |
| `detect_channel_radius` (phase 2) | (required) | 60 | 50 | both supply; differ from each other |
| `phase1_detect_channel_radius` (scheme 2) | (required) | 60 | 200 | both supply; differ |
| `detect_channel_radius` (scheme 1) | None | (n/a — example uses scheme 2) | 150 | wrapper sets 150 |

## Snippets
| Parameter | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| `snippet_T1` | 20 | **15** ≠ | 20 = | samples before peak |
| `snippet_T2` | 20 | **40** ≠ | 20 = | samples after peak |
| `snippet_mask_radius` | None | **60** ≠ | **250** ≠ | drives classifier neighborhood size (→ n_features) |

`snippet_mask_radius` is decisive for `n_features = (T1+T2)·M`: at 250 µm the NP
neighborhood is ~25 channels (n_features ≈ 1000–1100); at 60 µm it is ~5 channels
(n_features ≈ 275). For tetrodes M=4 either way (n_features = 160).

## PCA / classifier
| Parameter | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| `phase1_npca_per_channel` | 3 | 3 = | 3 = | same |
| `phase1_npca_per_subdivision` | 10 | 10 = | 10 = | same |
| `classifier_npca` | None → `max(12, M·3)` | **10** ≠ | None → `max(12, M·3)` = | wrapper can't set it (hardcoded None) |

## Training
| Parameter | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| `max_num_snippets_per_training_batch` | 200 | **1000** ≠ | 200 = | caps L per batch → drives PCA n_samples |
| `training_duration_sec` | None (full recording) | **350** ≠ | **300** ≠ | all three differ |
| `training_recording_sampling_mode` | `'initial'` | `'uniform'` ≠ | `'uniform'` ≠ | both override vanilla |
| `classification_chunk_sec` | None → `ceil(100e6/M)` samples | **100** ≠ | None = | wrapper leaves at vanilla default |

## Scheme 3
| Parameter | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| `block_duration_sec` | (required) | 300 (example's scheme-3 branch; example actually runs scheme 2) | 1800 | wrapper default 30 min |

## Preprocessing (ms5 does none; expects filtered + whitened input)
| Step | Vanilla ms5 | NP example | SI wrapper | Notes |
|---|---|---|---|---|
| Bandpass | N/A | 500–12000 Hz | 300–6000 Hz | different bands |
| Whiten | N/A (expected) | yes (`si.whiten`, unseeded) | yes (`whiten`, `whitening_seed`, default None) | wrapper adds optional seed |
| `phase_shift` | N/A | **yes** | no | NP-specific ADC deskew |
| `detect_bad_channels` + remove | N/A | **yes** | no | example removes bad channels |
| Common reference (CAR/CMR) | N/A | no | no | none in either |

## Headline differences
- **SI wrapper ≈ vanilla ms5 generic defaults**, with a few overrides: it sets
  `snippet_mask_radius=250` (vanilla None), `training_duration_sec=300` (vanilla
  None = full recording), `training_recording_sampling_mode='uniform'` (vanilla
  `'initial'`), `phase1_detect_time_radius_msec=0.5` (vanilla 1.5), and supplies
  the channel radii. It does **not** expose `classifier_npca` (stuck at None →
  `max(12, M·3)`), and it does no `phase_shift` / bad-channel removal.
- **The SI wrapper defaults are not Neuropixels-tuned.** Its `snippet_mask_radius=250`
  yields a ~25-channel neighborhood on NP (n_features ≈ 1000), vs the NP example's
  60 µm → ~5 channels (n_features ≈ 275). The wrapper's `detect_channel_radius`
  values (50/200/150) are also far larger than the example's 60.
- **The NP example diverges from both**, with NP-appropriate values:
  `snippet_mask_radius=60`, channel radii 60, `snippet_T1=15`/`T2=40`,
  `detect_threshold=5.25`, `detect_sign=0`, `classifier_npca=10`,
  `max_num_snippets_per_training_batch=1000`, `training_duration_sec=350`
  (`uniform`), `classification_chunk_sec=100`, bandpass 500–12000, plus
  `phase_shift` and bad-channel removal.

Sources: `mountainsort5/schemes/Scheme{1,2,3}SortingParameters.py`;
`spikeinterface/sorters/external/mountainsort5.py` (`_default_params`, lines ~151–184);
`mountainsort5/examples/neuropixel_quickstart/spikeglx.py`.

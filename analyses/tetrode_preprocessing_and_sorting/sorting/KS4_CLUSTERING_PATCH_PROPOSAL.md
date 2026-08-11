---
title: Proposal — bound Kilosort4 clustering GPU memory so it can sort long (48 h) recordings
updated: 2026-06-12
status: hypothesis
---

# Brief for a planning agent: cap Kilosort4's clustering GPU memory

## Your task

Plan (do not yet implement) a modification to a **fork of Kilosort4** so that its
clustering step can run on per-region spike sets of tens of millions of spikes on a
**32 GB GPU** without running out of memory. Detection and preprocessing already work
and must be left unchanged. The fix should be **opt-in** (a new parameter), default to
current behavior, and preserve sort quality as much as possible.

- **Fork base:** Kilosort **4.1.7**. The working copy is at
  `/Users/gfindlay@ad.wisc.edu/projects/ece/Kilosort/` (currently at upstream commit
  `17743f2`; the installed package in the workspace is the identical PyPI 4.1.7).
- **All line numbers below are in `kilosort/clustering_qr.py` of that 4.1.7 tree.**
- **Hardware:** one Tesla V100 (32 GB). Host has 1.5 TB system RAM.

## Background (why this matters)

We are sorting a 48 h, 64-channel tetrode recording (16 tetrodes × 4 ch, 30 kHz) and
comparing Kilosort4 against MountainSort5. KS4's **detection** runs fine on the GPU
(~62 min/tetrode, ~16–24 M spikes detected per tetrode over 48 h), but the **final
clustering OOMs**. The same wall hits whether sorting all tetrodes together (one 64-ch
run) or one tetrode at a time (4 ch) — it is driven by the per-region **spike count**
(∝ duration), not channel count. MS5 sorts the same 48 h in ~100 min because its
clustering runs on CPU with the 1.5 TB RAM. We want KS4 to be feasible at 48 h on the
GPU. (Full context: `KS4_FEASIBILITY_FINDINGS.md` in this directory.)

## The problem (precise)

KS4 clusters spikes **per spatial center** (it tiles the probe into `x_centers ×
y_centers` and clusters each center's spikes independently). For a tetrode there is
effectively one center holding all ~23.6 M of that tetrode's 48 h spikes.

The clustering of one center is `clustering_qr.cluster(Xd, ...)` (def at **L121**),
where `Xd` is **(n_spikes × n_features)** for that center — **all** the center's spikes.
It does:

1. `neigh_mat(Xd, ..., max_sub=25000)` (**L20**) — builds a kNN graph. It **subsamples
   the graph landmarks** to `≤ max_sub` evenly-spaced spikes (`Xsub`, the graph nodes),
   but returns `kn` of shape **(n_spikes × n_neigh)** — every spike's nearest landmarks.
2. `kmeans_plusplus(Xg, ...)` (**L~183**, called at **L155**) — centroid init over
   **all** `Xg` (= `Xd` on GPU).
3. A 200-iteration graph refinement (**L161–167**) alternating `assign_isub` and
   **`assign_iclust`**.

**The OOM is in `assign_iclust` (L69–86):**

```python
xN = coo(ij, tones2.flatten(), (n_spikes, nclust)).to_dense()      # (n_spikes × nclust) DENSE
...
xN = xN - lam/m * (ki.unsqueeze(-1) * kN.to_dense())               # broadcasts to (n_spikes × nclust)
iclust = torch.argmax(xN, 1)
```

With `n_spikes ≈ 23.6 M` and `nclust ≈ 130–200`, `(n_spikes × nclust)` float32 ≈
**12.6 GiB**. Two such intermediates live across an iteration plus the working set
(`Xg`, `kn`, `rows_neigh`, `tones2`, all ∝ n_spikes) ≈ **~30 GiB**, so the next 12.6 GiB
allocation fails: observed `torch.OutOfMemoryError: Tried to allocate 12.57 GiB … 29.98
GiB allocated`. (`assign_isub`, L89, builds `(nsub × nclust)` where `nsub ≤ 25000`, so
it is small — **only `assign_iclust` and the `kmeans_plusplus` init scale with
n_spikes**.)

## Why no existing parameter fixes it

`max_cluster_subset` (default 25000) and `cluster_downsampling` (default 20) bound only
the **landmark** subset `Xsub`/`nsub` (the graph nodes), **not** `n_spikes` (the rows of
`assign_iclust`). Confirmed by reading `neigh_mat` (L20–66) and `assign_iclust` (L69–86).
`expandable_segments`/`clear_cache` do not help — it is genuine allocated memory, not
fragmentation (reserved-but-unallocated was <100 MiB at OOM).

## The fix the KS4 authors themselves sketched

`kmeans_plusplus` ends (**L271–276**) with a commented-out block that is *exactly* the
intended approach for this situation:

```python
# NOTE: For very large datasets, we may end up needing to subsample Xg.
# If the clustering above is done on a subset of Xg,
# then we need to assign all Xgs here to get an iclust
# for ii in range((len(Xg)-1)//nblock +1):
#     vexp = 2 * Xg[ii*nblock:(ii+1)*nblock] @ mu.T - (mu**2).sum(1)
#     iclust[ii*nblock:(ii+1)*nblock] = torch.argmax(vexp, dim=-1)
```

i.e. **(1) cluster a subsample, (2) assign all spikes to the resulting clusters in
chunks.** The maintainers anticipated this; the scaffolding direction is already in the
file.

## Proposed solution

Two viable designs. **Recommend prototyping Option A first** (exact, no quality
tradeoff); fall back to **Option B** (the authors' sketch) if A's per-iteration chunk
overhead is too slow.

### Option A — chunk `assign_iclust` over the spike dimension (EXACT, no quality loss)

Each row of `xN` (one spike) is independent: it depends only on that spike's `kn`,
`ki[spike]`, and the current `isub`. So compute `iclust` in **batches of spikes**
(e.g. 1–2 M at a time), never materializing the full `(n_spikes × nclust)`:

```python
def assign_iclust(rows_neigh, isub, kn, tones2, nclust, lam, m, ki, kj, device, chunk=2_000_000):
    n_spikes = kn.shape[0]
    out = torch.empty(n_spikes, dtype=torch.long, device=device)
    for a in range(0, n_spikes, chunk):
        b = min(a+chunk, n_spikes)
        # build xN for rows [a:b] only -> (b-a, nclust), apply the lam term with ki[a:b], argmax
        out[a:b] = ...
    return out
```

- **Exact:** identical result to the current code (argmax per spike is unchanged), so it
  is a pure memory/perf change — important because this study cares about determinism
  (`[[project_ms5_pca_nondeterminism]]`).
- Peak clustering memory drops to the working set (~7 GiB for 23.6 M spikes) + one
  `(chunk × nclust)` tile. Fits 32 GB with wide margin.
- Cost: 200 iterations × (n_spikes/chunk) tiles of extra Python-loop overhead. Likely
  slower than today on small data; acceptable since today it simply cannot run on large
  data. `rows_neigh`/`tones2` (each `(n_spikes × n_neigh)`, ~1–2 GiB) can stay whole or
  be sliced per chunk.

### Option B — subsample spikes for clustering, then chunked final assignment (APPROXIMATE, faster)

Follows L271–276 directly:

1. In `clustering_qr.run` (the per-center loop; see below) or inside `cluster()`,
   **uniformly subsample** the center's `Xd` to a cap (e.g. 1–2 M spikes) using the
   existing `subsample_idx(n1, n2)` helper (**L~282**, already used by `neigh_mat` for
   landmarks — it returns an evenly-distributed boolean mask).
2. Run the full `cluster()` (kmeans init + 200-iter graph refinement) on the
   **subsample** → cluster labels for the subsample. Peak `(subsample × nclust)` ≈
   1 GiB.
3. Compute per-cluster centroids `mu` from the subsample's features.
4. Assign **all** `n_spikes` to the nearest centroid in **chunks** (the commented loop:
   `vexp = 2*Xg_chunk @ mu.T - (mu**2).sum(1); iclust_chunk = argmax(vexp)`).

- Quality tradeoff: clusters are *discovered* from a representative subsample; every
  spike is still *labeled*. A unit needs ≳ (24 M / cap) × (a few hundred) spikes to be
  well-represented — fine for essentially all real units over 48 h; only very
  low-count units are at mild risk. **Must use a seeded/deterministic subsample** so the
  sort stays reproducible.
- Faster than A: the expensive 200-iteration loop runs on the small subsample; only the
  one-shot final assignment is chunked.

## Where to make the change (call graph)

- `run_kilosort.cluster_spikes` (`run_kilosort.py` L868) → `clustering_qr.run` (L815/L910).
- **`clustering_qr.run(ops, st, tF, ...)` (L416–557)** is the per-center orchestrator:
  - loops `xcent` (L445) × `ycent` (L452);
  - per center, `get_data_cpu(...)` (L560) builds that center's `Xd` (= `(nspikes, nfeat)`);
  - **calls `cluster(Xd, ...)` at L495** → per-center `iclust`;
  - writes labels globally: **`clu[igood] = iclust + nmax` (L519)**.
- `assign_iclust` L69, `assign_isub` L89, `cluster` L121, `neigh_mat` L20, `Mstats` L108,
  `kmeans_plusplus` L183 (+ the NOTE at L271), `subsample_idx` L~282.

Option A is contained to `assign_iclust` (+ thread a `chunk`/`clustering_chunk_size`
setting through `cluster`/`run`). Option B touches `cluster()`/`run` (subsample Xd; add
the centroid-based chunked final assignment from L274–276) and must keep `clu`/`Wall`
outputs consistent.

## Subtleties / risks the plan must resolve

1. **`ki`/`kj`/`m` from `Mstats(M)` (L108).** In `assign_iclust` the `lam` term uses
   `ki` (per-spike degree). Option A: slice `ki[a:b]` per chunk — straightforward.
   Option B: the final centroid assignment bypasses the graph `lam` term entirely (uses
   `mu`), so confirm that matches KS4's intent closely enough (it is the authors' own
   sketch, but it is a *centroid* assignment, not the graph objective).
2. **Outputs.** `clustering_qr.run` returns `clu` (per-spike labels) and `Wall`
   (templates). Ensure both are computed over **all** spikes after the change (Option B:
   templates/`Wall` are built from cluster membership — verify they use the full
   assignment, not the subsample).
3. **Determinism.** The study depends on reproducible sorts. Option A is exact. Option B
   must seed the subsample (`subsample_idx` is deterministic given n1/n2; confirm no RNG
   leaks) so re-runs match.
4. **`kmeans_plusplus` (L183) also scales with n_spikes.** Under Option A it still runs
   on all spikes — check whether its `Xg`-sized intermediates (it `del`s `vexp`/`dexp`
   per iter, L260) stay within budget at 23.6 M, or whether it too needs the L274 chunked
   tail. Under Option B it runs on the subsample, so it is bounded for free.
5. **Per-center generality.** Tetrode data = ~1 center/run; Neuropixels = many centers
   with fewer spikes each. The cap/chunk must be a no-op for small centers (only kicks in
   above a threshold) so normal Neuropixels sorting is unchanged.
6. **CPU clustering path.** KS4 can run clustering on CPU (`torch_device='cpu'`); the
   change should not break it (chunking is device-agnostic).

## How it plugs into our pipeline

We invoke KS4 via SpikeInterface from `tetrode_analyses.sorting.sort_store_ks4`
(`tetrode_analyses/src/tetrode_analyses/sorting.py`) and the driver
`.../sorting/30_sort_ks4.py`. Expose the new behavior as a KS4 setting (e.g.
`clustering_chunk_size` for A or `max_cluster_spikes` for B) so it flows through the SI
wrapper as an ordinary sorter param. Install the modified fork into the workspace
editable (add a `[tool.uv.sources] kilosort = { path = "../Kilosort", editable = true }`
in `gfys_workspace/pyproject.toml`, mirroring the other sibling-package overrides) — or,
if a fork install is undesirable, the same change can be delivered as a runtime
monkeypatch of `clustering_qr.assign_iclust`/`cluster` applied inside `sort_store_ks4`.

## Validation plan

1. **Unit/regression (Option A):** on a 600 s crop (already known-good), the patched
   `assign_iclust` must produce **bit-identical** `clu` to the unpatched version
   (chunked == unchunked). For Option B, document that output is approximate and compare
   unit counts.
2. **Memory + quality on a 12 h crop** (KS4 fits 12 h today): confirm patched peak GPU
   memory is bounded (e.g. < 16 GiB) and that units/agreement vs the unpatched 12 h sort
   are unchanged (A) or within tolerance (B). Record timing overhead.
3. **Full 48 h** (1 tetrode, then all 16 by group): must **complete under 32 GB** and
   produce a sorting. Then run the staged downstream `31_build_analyzer_ks4.py` /
   `32_compare_ks4_vs_ms5.py` (curated KS4-vs-MS5 agreement).
4. **Determinism:** re-run a tetrode and confirm identical output.

## Acceptance criteria

- Full 48 h tetrode sort completes on the 32 GB V100 (no OOM), peak GPU memory bounded
  and roughly independent of recording duration.
- Default (parameter unset) behavior is byte-for-byte unchanged vs upstream 4.1.7.
- Option A: clustering output identical to upstream on data that fits today.
- Quality: curated (well-isolated) unit agreement KS4-vs-MS5 is sensible and stable.

---
title: Tetrode SortingAnalyzer sparsity — use by_property="group", not radius
status: active
updated: 2026-07-26
confirmed_by_user: not_required
---

# Tetrode SortingAnalyzer sparsity

When building a SpikeInterface `SortingAnalyzer` for a **tetrode** sort, set sparsity
with:

```python
sparsity = ChannelSparsity.from_property(sorting, recording, by_property="group")
analyzer = create_sorting_analyzer(sorting, rec, sparsity=sparsity, ...)
```

rather than the radius default (`method="radius", radius_um=100`).

## Why

The `group` property *is* the tetrode (4 channels), so `by_property` maps each unit to
exactly its 4-channel tetrode — direct and exact. Radius only approximates this via
channel geometry; it happens to land on the same 4 channels here, but it is a
heuristic.

## Notes

- Pass the sparsity explicitly; the `sparse` flag is then ignored (no conflict).
- Both the recording and the sorting must carry the `group` property. `tetrode_analyses`
  recordings and aggregated sortings both do, values `0..15`.
- Verified to give exactly 4 channels/unit on the 12 h-block sort. First used in
  `analyses/tetrode_preprocessing_and_sorting/sorting/25_build_analyzer_12hblock_train1h.py`.

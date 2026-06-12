"""Population ISI histogram (0-5 ms, 1-sample bins): is the 1.0-1.5 ms band real or artifact?

The isi/rp tier disagreement is driven by the refractory window (isi 1.5 ms vs rp 1.0 ms),
and isi@1.5 jumps because there is a lot of ISI mass in the 1.0-1.5 ms band. This asks WHAT is
in that band: a discrete double-detection / artifact spike at a fixed sample lag (-> the wider
window is catching a sorting artifact, not biology), or smooth contamination (-> real
refractory violations).

Three pooled ISI histograms at 1-sample (1/30 ms = 0.0333 ms) resolution over 0-5 ms:
  A. ALL 2204 units.
  B. The DISAGREEMENT units: rp_contamination@1.0 <= 0.1 but isi_violations_ratio@1.5 > 0.1
     (the ~398 units whose tier hinges on the window) -- this is the band in question.
  C. The AND-clean units (both <= 0.1) -- reference for a clean refractory.
Marks 1.0 and 1.5 ms. Prints per-group ISI-band fractions and the top discrete sample lags
(a fixed-lag spike = artifact signature).
"""
import pathlib

import matplotlib
import numpy as np
import spikeinterface as si

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

T = pathlib.Path("/nvme/neuropixels/tetrode_data/2026-05-27_09-07-52/sortings_seed42_pcafix/tracked_48h")
ANALYZER = T / "analyzer_clustered.zarr"
FS = 30000.0
MAX_MS = 5.0
MAX_SAMP = int(MAX_MS * FS / 1000)  # 150 samples = 5 ms


def pooled_isi(sorting, unit_ids):
    """Accumulate an ISI count histogram (sample lags 1..MAX_SAMP) over the given units."""
    h = np.zeros(MAX_SAMP + 1, dtype=np.int64)
    n_isi = 0
    for u in unit_ids:
        st = sorting.get_unit_spike_train(u)
        if st.size < 2:
            continue
        d = np.diff(np.sort(st))
        n_isi += d.size
        d = d[(d >= 1) & (d <= MAX_SAMP)]
        h += np.bincount(d, minlength=MAX_SAMP + 1)
    return h, n_isi


def band_fracs(h):
    """Fraction of <=5 ms ISIs falling in [0,0.5),[0.5,1.0),[1.0,1.5),[1.5,2.0),[2.0,5.0] ms."""
    samp = np.arange(MAX_SAMP + 1)
    ms = samp / FS * 1000
    tot = h.sum()
    edges = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 5.0001)]
    return {f"{lo}-{hi}": round(float(h[(ms >= lo) & (ms < hi)].sum()) / tot, 3) if tot else 0.0
            for lo, hi in edges}, int(tot)


def main():
    an = si.load_sorting_analyzer(str(ANALYZER))
    sorting = an.sorting
    qm = an.get_extension("quality_metrics").get_data()
    isi = qm["isi_violations_ratio"].to_numpy()  # default 1.5 ms
    rp = qm["rp_contamination"].to_numpy()        # default 1.0 ms
    fin = np.isfinite(isi) & np.isfinite(rp)
    uids = np.asarray(sorting.unit_ids)
    groups = {
        "all": uids,
        "disagreement (rp<=.1, isi>.1)": uids[(rp <= 0.1) & (isi > 0.1) & fin],
        "AND-clean (both<=.1)": uids[(rp <= 0.1) & (isi <= 0.1) & fin],
    }

    ms_axis = np.arange(MAX_SAMP + 1) / FS * 1000
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for ax, (name, ids) in zip(axes, groups.items()):
        h, n_isi = pooled_isi(sorting, ids)
        fr, tot = band_fracs(h)
        ax.bar(ms_axis, h, width=(1 / FS * 1000), align="edge", color="tab:blue", alpha=0.8)
        ax.axvline(1.0, color="tab:green", lw=1.2, ls="--", label="1.0 ms (rp)")
        ax.axvline(1.5, color="tab:red", lw=1.2, ls="--", label="1.5 ms (isi)")
        ax.set_yscale("log")
        ax.set_ylabel("ISI count (log)")
        peak_samp = int(np.argmax(h[1:]) + 1)
        top3 = np.argsort(h)[::-1][:3]
        ax.set_title(f"{name}  |  n_units={len(ids)}  ISIs≤5ms={tot:,}  "
                     f"bands(ms) {fr}  | peak@{peak_samp / FS * 1000:.2f}ms", fontsize=8.5)
        ax.legend(fontsize=7, loc="upper right")
        print(f"[{name}] n_units={len(ids)} ISIs<=5ms={tot:,} bands={fr} "
              f"peak_lag={peak_samp}samp ({peak_samp / FS * 1000:.3f}ms) "
              f"top3_lags_ms={[round(int(s) / FS * 1000, 3) for s in top3]}", flush=True)
    axes[-1].set_xlabel("ISI (ms)")
    fig.suptitle("Population ISI histograms 0-5 ms (1-sample bins). Is the 1.0-1.5 ms band a "
                 "discrete artifact or smooth contamination?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = T / "population_isi_histogram.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

"""Is WavPack-hybrid error below the spike-band noise floor?

Only the 300-6000 Hz band matters for spike sorting. Compare the in-band RMS of
the compression residual against the in-band noise floor of the data itself.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from wavpack_numcodecs import WavPack

RAW = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy").astype("f4")
N, NCH = RAW.shape
FS = 30000; BITVOLTS = 0.1949999928
sos = butter(3, [300, 6000], btype="band", fs=FS, output="sos")

def bandpass(x):  # along time, per channel
    return sosfiltfilt(sos, x, axis=0)

raw_bp = bandpass(RAW)
# robust in-band noise floor per channel (MAD -> std), in uV
noise_uV = (np.median(np.abs(raw_bp), 0) / 0.6745) * BITVOLTS
print(f"in-band (300-6000Hz) noise floor: median {np.median(noise_uV):.2f} uV "
      f"(range {noise_uV.min():.2f}-{noise_uV.max():.2f} uV)", flush=True)

raw_i16 = np.load("/nvme/neuropixels/tmp/cc_bench/sample_raw.npy")
for bps in [2.25, 3.0, 3.5]:
    comp = WavPack(bps=bps)
    dec = np.frombuffer(comp.decode(comp.encode(raw_i16)), dtype="<i2").reshape(N, NCH).astype("f4")
    resid = RAW - dec
    resid_bp = bandpass(resid)
    inband_resid_uV = resid_bp.std(0) * BITVOLTS          # per channel
    wide_resid_uV = resid.std(0) * BITVOLTS
    ratio_to_noise = inband_resid_uV / noise_uV
    print(f"bps={bps}: in-band resid RMS median {np.median(inband_resid_uV):.3f} uV "
          f"| wideband resid {np.median(wide_resid_uV):.3f} uV "
          f"| in-band resid / noise floor = {np.median(ratio_to_noise):.3f} "
          f"(max {ratio_to_noise.max():.3f})", flush=True)
print("DONE", flush=True)
